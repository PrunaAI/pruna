import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from ImageReward import ImageReward
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

LOCAL_MODEL_PATH = r"C:\Users\dell inspiron\.cache\FaceScore\FS_model.pt"
LOCAL_CONFIG_PATH = r"C:\Users\dell inspiron\.cache\FaceScore\med_config.json"


def available_models() -> List[str]:
    """Returns the names of available FS models (local only)"""
    return ["FaceScore"]

# Always load FaceScore model from local path
def load(name: str = "FaceScore", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", med_config: str = None):
    """Load a FaceScore model from local files only."""
    if name == "FaceScore":
        model_path = LOCAL_MODEL_PATH
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"FaceScore model weights not found at {model_path}. Please download FS_model.pt and place it there.")

    print(f'Loading checkpoint from {model_path}')
    state_dict = torch.load(model_path, map_location='cpu')

    # med_config
    if med_config is None:
        med_config = LOCAL_CONFIG_PATH
    if not os.path.isfile(med_config):
        raise FileNotFoundError(f"FaceScore config not found at {med_config}. Please download med_config.json and place it there.")

    model = ImageReward(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded")
    model.eval()

    return model

