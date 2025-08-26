import numpy as np
import torch
import torchvision
from torchvision import transforms

from pruna import SmashConfig, smash

# 1. Load a small Torch model (just for testing)
model = torchvision.models.resnet18(weights=None)  # no pretrained, lightweight
model.eval()


# Initialize SmashConfig
smash_config = SmashConfig()
smash_config["quantizer"] = "quanto"

# Smash the model
smashed_model = smash(
    model=model,
    smash_config=smash_config,
)

image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.float32)
input_tensor = transforms.ToTensor()(image).unsqueeze(0)

# 5. Run inference
with torch.no_grad():
    output = smashed_model(input_tensor)

print("Output shape:", output.shape)
