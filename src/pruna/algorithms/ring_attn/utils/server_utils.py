# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import contextlib
import multiprocessing as mp
import os
import socket

import torch
import torch.distributed as dist

from pruna.engine.utils import safe_memory_cleanup
from pruna.logging.logger import pruna_logger

# Global placeholder for shared pipeline (CPU)
global_pipe = None


@atexit.register
def _cleanup_distributed():
    """Clean up the distributed process group even in uncontrolled system exits."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedServer:
    """
    Wrapper to distribute the model across multiple GPUs.

    This is our way of avoiding to run a script with "torchrun", we do the setup ourselves and spawn processes with
    copies of the model manually.

    Parameters
    ----------
    pipe : Any
        The pipeline to distribute.
    smash_config : Any
        Configuration for smashing the pipeline.

    Examples
    --------
    >>> server = DistributedServer(pipe, wrap_fn, config)
    >>> server.start()
    >>> result = server("prompt")
    """

    def __init__(self, pipe, smash_config):
        self.pipe = pipe
        self.pool = None
        self.world_size = torch.cuda.device_count()
        self.smash_config = smash_config
        self.device = "cuda"
        if self.world_size < 2:
            raise ValueError("This algorithm requires at least 2 GPUs")

    def set_env_vars(self):
        """Set the environment variables for torch multi-processing."""
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            os.environ.setdefault("MASTER_PORT", str(s.getsockname()[1]))

    @staticmethod
    def _static_init_worker(world_size, pipe, smash_config):
        """
        Static method version of _init_worker for better pickle compatibility.

        Parameters
        ----------
        world_size : int
            Total number of processes in the distributed setup
        pipe : Any
            The pipeline object to initialize
        """
        global global_pipe
        # Determine rank
        rank = mp.current_process()._identity[0] - 1
        # Ensure each worker reports the correct device before consistency checks
        smash_config["device"] = f"cuda:{rank}"
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        # Bind this process to its GPU before CUDA operations
        torch.cuda.set_device(rank)
        # Move the unpickled CPU pipeline to the current GPU
        gpu_pipe = pipe.to(f"cuda:{rank}")
        gpu_pipe.smashed = True

        from pruna import smash

        global_pipe = smash(gpu_pipe, smash_config)

    @staticmethod
    def _static_cleanup_worker(task_data):
        """
        Static method to perform cleanup operations on workers before termination.

        This method is called on each worker process before the pool is destroyed.
        You can customize this method to perform any specific cleanup operations
        needed for your workers.

        Returns
        -------
        bool
            True if cleanup was successful, False otherwise.

        Examples
        --------
        >>> success = DistributedServer._static_cleanup_worker()
        >>> print(f"Worker cleanup successful: {success}")
        """
        global global_pipe

        if global_pipe is None:
            return

        try:
            rank = torch.cuda.current_device()
            pruna_logger.info(f"Cleaning up worker on rank {rank}")

            global_pipe.to("cpu")
            global_pipe.destroy()
            safe_memory_cleanup()

            pruna_logger.info(f"Worker cleanup completed successfully on rank {rank}")
            global_pipe = None
            return True

        except Exception as e:
            pruna_logger.error(f"Error during worker cleanup: {e}")
            return False

    @staticmethod
    def _static_process_task(task_data):
        """
        Static method version of _process_task for better pickle compatibility.

        Parameters
        ----------
        task_data : tuple
            Tuple containing (rank, args, kwargs) where:
            - rank: int, the GPU rank to use
            - args: tuple, positional arguments to pass to global_pipe
            - kwargs: dict, keyword arguments to pass to global_pipe

        Returns
        -------
        Any
            The output from the global pipe, processed based on rank.

        Examples
        --------
        >>> task_data = (0, ("prompt",), {"num_steps": 20})
        >>> result = DistributedServer._static_process_task(task_data)
        """
        rank, args, kwargs = task_data
        torch.cuda.set_device(rank)
        return global_pipe.__call__(*args, rank=rank, **kwargs)

    def start(self):
        """Launch a worker pool, sending the CPU pipeline into each child."""
        if self.pool:
            return

        self.device = "cuda"

        pruna_logger.info("Spawning distributed setup...")
        pruna_logger.info("Before terminating the current process, call smashed_model.destroy() for proper cleanup.")

        self.set_env_vars()
        ctx = mp.get_context("spawn")

        self.pool = ctx.Pool(
            processes=self.world_size,
            initializer=self._static_init_worker,
            initargs=(self.world_size, self.pipe, self.smash_config),
        )

    def __call__(self, *args, **kwargs):
        """
        Dispatch prompt across ranks and return rank-0's image.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the pipeline
        **kwargs : Any
            Keyword arguments to pass to the pipeline

        Returns
        -------
        Any
            The result from rank 0
        """
        if not self.pool:
            raise RuntimeError("Runner not started. Use start() or context manager.")
        tasks = [(rank, args, kwargs) for rank in range(self.world_size)]
        results = self.pool.map(self._static_process_task, tasks)
        return results[0]

    def destroy(self):
        """Cleanly shut down the worker pool."""
        if self.pool:
            pruna_logger.info("Triggering cleanup on all workers...")
            self.device = "cpu"
            self.pool.map(self._static_cleanup_worker, range(self.world_size))

            # Close the pool
            self.pool.close()
            for worker in self.pool._pool:
                worker.join(timeout=10)
            self.pool = None

        if dist.is_initialized():
            dist.destroy_process_group()

    def __getattr__(self, attr):
        """
        Forward all other attributes to the global pipe.

        Parameters
        ----------
        attr : str
            The attribute name to forward

        Returns
        -------
        Any
            The attribute value from the pipeline
        """
        if attr == "device":
            return "cuda" if self.pool else "cpu"
        return getattr(self.pipe, attr)

    def save_pretrained(self, path: str):
        """
        Offer interface to save the distributed model to inform the user that it is not supported.

        Parameters
        ----------
        path : str
            The path to save the distributed model.
        """
        raise NotImplementedError("Saving a distributed model is not supported at the moment.")
