

# Copyright (c) 2025 PrunaAI. All rights reserved.
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

from __future__ import annotations

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms
from batch_face import RetinaFace
import cv2
from PIL import Image
import numpy as np
from .utils import load

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB."""
    return image.convert("RGB")


def _transform(n_px: int) -> Compose:
    """Return a composed transform for preprocessing face images."""
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class MLP(nn.Module):
    """Multi-layer perceptron for face scoring."""

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if "bias" in name:
                nn.init.constant_(param, val=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)
    

class FaceScore(nn.Module):
    """FaceScore reward model for evaluating face quality in images."""
    def __init__(self, model_name: str, med_config: str = None, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.detector = RetinaFace()
        self.default_prompt = "A face."
        ir = load(model_name, med_config=med_config)
        self.blip = ir.blip
        self.mlp = ir.mlp
        self.preprocess = _transform(224)

    def score(self, faces: torch.Tensor) -> torch.Tensor:
        """Score a batch of face images."""
        bs = faces.shape[0]
        text_input = self.blip.tokenizer(
            self.default_prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        text_ids = text_input.input_ids.repeat(bs, 1, 1, 1).view(bs, -1).to(self.device)
        text_masks = text_input.attention_mask.repeat(bs, 1, 1, 1).view(bs, -1).to(self.device)
        face_embeds = self.blip.visual_encoder(faces)
        face_atts = torch.ones(face_embeds.size()[:-1], dtype=torch.long).to(self.device)
        emb = (
            self.blip.text_encoder(
                text_ids,
                attention_mask=text_masks,
                encoder_hidden_states=face_embeds,
                encoder_attention_mask=face_atts,
                return_dict=True,
            ).last_hidden_state
        )[:, 0, :].float()
        reward = self.mlp(emb)
        return reward

    def crop_face_image(self, image: Image.Image, box: list[float]) -> torch.Tensor:
        """Crop a face from an image given a bounding box."""
        image = transforms.ToTensor()(image) * 255
        x, y, w, h = (
            int(box[0] - 1),
            int(box[1] - 1),
            int(box[2] + 1) - int(box[0] - 1),
            int(box[3] + 1) - int(box[1] - 1),
        )
        cropped_image = image[None][:, :, y : y + h, x : x + w]
        return cropped_image[0]

    def get_reward(self, face_path: str, threshold: float = 0.9) -> tuple[list, list, list]:
        """Get face quality rewards for faces detected in an image file."""
        face_img = cv2.imread(face_path)
        img1 = Image.open(face_path)
        faces = self.detector(face_img, cv=False)
        get_faces = []
        boxes = []
        confidences = []
        for box, landmarks, confidence in faces:
            if confidence < threshold:
                continue
            face = self.crop_face_image(img1, box)
            try:
                face = Image.fromarray(face.numpy().transpose(1, 2, 0).astype(np.uint8))
            except Exception:
                continue
            face = self.preprocess(face).unsqueeze(0).to(self.device)
            get_faces.append(face)
            boxes.append(box.tolist())
            confidences.append(float(confidence))
        if not get_faces:
            return [], [], []
        faces = torch.cat(get_faces, 0)
        rewards = self.score(faces)
        rewards = torch.squeeze(rewards)
        return rewards.detach().cpu().numpy().tolist(), boxes, confidences