---
library_name: {library_name}
tags:
- pruna-ai
---

# Model Card for {repo_id}

This model was created using the [pruna](https://github.com/PrunaAI/pruna) library. Pruna is a model optimization framework built for developers, enabling you to deliver faster, more efficient models with minimal overhead.

## Usage

You can load this model using the following code:

```python
from pruna import PrunaModel

loaded_model = PrunaModel.from_hub("{repo_id}")
```

After loading the model, you can use the inference methods of the original model.

## Model Configuration

The configuration of the model is stored in the `config.json` file.

```bash
{model_config}
```

## Smash Configuration

The configuration of the model is stored in the `smash_config.json` file.

```bash
{smash_config}
```