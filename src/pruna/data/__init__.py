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

from typing import Any, Callable, Tuple

from pruna.data.datasets.audio import setup_commonvoice_dataset, setup_podcast_dataset
from pruna.data.datasets.image import (
    setup_cifar10_dataset,
    setup_imagenet_dataset,
    setup_mnist_dataset,
)
from pruna.data.datasets.question_answering import setup_polyglot_dataset
from pruna.data.datasets.text_generation import (
    setup_c4_dataset,
    setup_openassistant_dataset,
    setup_pubchem_dataset,
    setup_smolsmoltalk_dataset,
    setup_smoltalk_dataset,
    setup_wikitext_dataset,
)
from pruna.data.datasets.text_to_image import (
    setup_coco_dataset,
    setup_laion256_dataset,
    setup_open_image_dataset,
)

base_datasets: dict[str, Tuple[Callable, str, dict[str, Any]]] = {
    "COCO": (setup_coco_dataset, "image_generation_collate", {"img_size": 512}),
    "LAION256": (setup_laion256_dataset, "image_generation_collate", {"img_size": 512}),
    "CommonVoice": (setup_commonvoice_dataset, "audio_collate", {}),
    "AIPodcast": (setup_podcast_dataset, "audio_collate", {}),
    "ImageNet": (setup_imagenet_dataset, "image_classification_collate", {"img_size": 224}),
    "MNIST": (setup_mnist_dataset, "image_classification_collate", {"img_size": 28}),
    "WikiText": (setup_wikitext_dataset, "text_generation_collate", {}),
    "SmolTalk": (setup_smoltalk_dataset, "text_generation_collate", {}),
    "SmolSmolTalk": (setup_smolsmoltalk_dataset, "text_generation_collate", {}),
    "PubChem": (setup_pubchem_dataset, "text_generation_collate", {}),
    "OpenAssistant": (setup_openassistant_dataset, "text_generation_collate", {}),
    "C4": (setup_c4_dataset, "text_generation_collate", {}),
    "Polyglot": (setup_polyglot_dataset, "question_answering_collate", {}),
    "OpenImage": (setup_open_image_dataset, "image_generation_collate", {"img_size": 1024}),
    "CIFAR10": (setup_cifar10_dataset, "image_classification_collate", {"img_size": 32}),
}
