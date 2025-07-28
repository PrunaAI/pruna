---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for {{ repo_id | default("Model ID", true )}}

This model was created using the [pruna](https://github.com/PrunaAI/pruna) library. Pruna is a model optimization framework built for developers, enabling you to deliver more efficient models with minimal implementation overhead.

## Usage

First things first, you need to install the pruna library:

```bash
pip install {{ pruna_library | default("pruna", true) }}
```

You can [use the {{ library_name | default("library_name", true) }} library to load the model](https://huggingface.co/{{ repo_id | default("repo_id", true) }}?library={{ library_name | default("library_name", true) }}) but this might not include all optimizations by default.

To ensure that all optimizations are applied, use the pruna library to load the model using the following code:

```python
from {{ pruna_library }} import {{ pruna_model_class }}

loaded_model = {{ pruna_model_class }}.from_pretrained(
    "{{ repo_id }}"
)
# we can then run inference using the methods supported by the base model
```

{% if base_repo_id and library_name %}
For inference, you can use the inference methods of the original model like shown in [the original model card](https://huggingface.co/{{ base_repo_id }}?library={{ library_name }}).
{% endif %} Alternatively, you can visit [the Pruna documentation](https://pruna.readthedocs.io/en/latest/index.html) for more information.

## Smash Configuration

The compression configuration of the model is stored in the `smash_config.json` file, which describes the optimization methods that were applied to the model.

```bash
{{ smash_config }}
```

## 🌍 Join the Pruna AI community!

[![Twitter](https://img.shields.io/twitter/follow/PrunaAI?style=social)](https://twitter.com/PrunaAI)
[![GitHub](https://img.shields.io/github/followers/PrunaAI?label=Follow%20%40PrunaAI&style=social)](https://github.com/PrunaAI)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/company/93832878/admin/feed/posts/?feedType=following)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?style=social&logo=discord)](https://discord.com/invite/rskEr4BZJx)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/PrunaAI?style=social)](https://www.reddit.com/r/PrunaAI/)
