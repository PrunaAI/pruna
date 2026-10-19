"""Capability checks for first-block cache beyond the image-pipeline tester."""

from types import SimpleNamespace

import pytest
import torch

from pruna import SmashConfig
from pruna.algorithms.first_block_cache import FirstBlockCache

pytestmark = pytest.mark.cpu


def _pipeline_with(transformer: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(transformer=transformer)


@pytest.mark.parametrize("model_fixture", ["opt_tiny_random", "sd_tiny_random"], indirect=True)
def test_non_transformer_models_are_rejected(model_fixture: tuple) -> None:
    """Models without a cacheable ``transformer`` are rejected."""
    model, _ = model_fixture
    assert not FirstBlockCache().model_check_fn(model)


@pytest.mark.parametrize("model_fixture", ["flux2_tiny_random"], indirect=True)
def test_flux2_pipeline_caches_each_block_list(model_fixture: tuple) -> None:
    """Flux.2 dev runs with a per-list cache, including a threshold that skips the tail."""
    model, _ = model_fixture
    algorithm = FirstBlockCache()
    assert algorithm.model_check_fn(model)

    smash_config = SmashConfig()
    smash_config.add("first_block_cache")
    smash_config.add({"first_block_cache_threshold": 1.0})
    algorithm.apply(model, smash_config)
    assert model.transformer.is_cache_enabled

    kwargs = {
        "prompt": "a red apple",
        "num_inference_steps": 4,
        "height": 64,
        "width": 64,
        "text_encoder_out_layers": (1,),
    }
    first = model(**kwargs).images[0]
    second = model(**kwargs).images[0]
    assert first.size == (64, 64)
    assert second.size == (64, 64)


def test_ltx_tiny_checkpoint_has_one_block() -> None:
    """The public tiny LTX checkpoint has a single layer, so first-block cache cannot hook a tail block."""
    from diffusers import LTXVideoTransformer3DModel

    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "katuni4ka/tiny-random-ltx-video", subfolder="transformer", torch_dtype=torch.float32
    )
    assert not FirstBlockCache().model_check_fn(_pipeline_with(transformer))


def test_z_image_tiny_transformer_is_rejected() -> None:
    """Z-Image blocks are registered, but the transformer has no ``enable_cache`` on diffusers 0.39."""
    from diffusers import ZImageTransformer2DModel

    transformer = ZImageTransformer2DModel.from_pretrained(
        "tiny-random/z-image", subfolder="transformer", torch_dtype=torch.float32
    )
    assert not hasattr(transformer, "enable_cache")
    assert not FirstBlockCache().model_check_fn(_pipeline_with(transformer))


def test_wan_tiny_transformer_enables_cache() -> None:
    """Wan blocks are registered. The tiny checkpoint is checked without loading the Wan VAE."""
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        "pruna-test/wan-t2v-tiny-random", subfolder="transformer", torch_dtype=torch.float32
    )
    pipeline = _pipeline_with(transformer)
    algorithm = FirstBlockCache()
    assert algorithm.model_check_fn(pipeline)

    smash_config = SmashConfig()
    smash_config.add("first_block_cache")
    smash_config.add({"first_block_cache_threshold": 0.05})
    algorithm.apply(pipeline, smash_config)
    assert transformer.is_cache_enabled


def test_hunyuan_video_transformer_enables_cache() -> None:
    """No public tiny HunyuanVideo pipeline is available. A tiny constructed transformer still enables the cache."""
    from diffusers import HunyuanVideoTransformer3DModel

    transformer = HunyuanVideoTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        text_embed_dim=32,
        pooled_projection_dim=16,
        rope_axes_dim=(2, 2, 4),
    )
    pipeline = _pipeline_with(transformer)
    algorithm = FirstBlockCache()
    assert algorithm.model_check_fn(pipeline)

    smash_config = SmashConfig()
    smash_config.add("first_block_cache")
    algorithm.apply(pipeline, smash_config)
    assert transformer.is_cache_enabled


def test_single_block_transformer_is_rejected() -> None:
    """First-block cache needs a head block and a tail block."""
    from diffusers import HunyuanVideoTransformer3DModel

    transformer = HunyuanVideoTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_single_layers=0,
        num_refiner_layers=1,
        text_embed_dim=32,
        pooled_projection_dim=16,
        rope_axes_dim=(2, 2, 4),
    )
    assert not FirstBlockCache().model_check_fn(_pipeline_with(transformer))
