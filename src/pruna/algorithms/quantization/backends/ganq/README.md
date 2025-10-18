#### Guide to use GANQ quantization

**Quantize a model**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruna.config.smash_config import SmashConfig
from pruna.data.pruna_datamodule import PrunaDataModule


import torch
from transformers import AutoModelForCausalLM

import torch
from pruna.algorithms.quantization.ganq import GANQQuantizer

# -------------------------------------------------------------------------
# 1. Load model and tokenizer
# -------------------------------------------------------------------------
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# -------------------------------------------------------------------------
# 2. Build SmashConfig for Pruna Quantizer
# -------------------------------------------------------------------------
smash_config = SmashConfig(
    batch_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_dir_prefix="./cache_ganq",
)

# Add tokenizer
smash_config.add_tokenizer(tokenizer)

# Use Pruna's built-in WikiText dataset (handles train/val/test splits automatically)
data_module = PrunaDataModule.from_string(
    "WikiText",
    tokenizer=tokenizer,
    collate_fn_args=dict(max_seq_len=256),
)
data_module.limit_datasets(32)  # Limit to 32 examples per split for quick testing
smash_config.add_data(data_module)

# Configure quantizer parameters
smash_config.load_dict(
    {
        "quantizer": "ganq",
        "ganq_weight_bits": 4,
        "ganq_max_epoch": 10,
        "ganq_pre_process": True,
    }
)

# -------------------------------------------------------------------------
# 4. Run Quantization
# -------------------------------------------------------------------------
quantizer = GANQQuantizer()

quantized_model = quantizer._apply(model, smash_config)

# -------------------------------------------------------------------------
# 5. Save the quantized model
# -------------------------------------------------------------------------
quantized_model.save_pretrained("./ganq_quantized_smollm")
tokenizer.save_pretrained("./ganq_quantized_smollm")

print("✅ GANQ quantization complete and model saved at ./ganq_quantized_smollm")


def model_size_in_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


original_size = model_size_in_mb(model)
quantized_size = model_size_in_mb(quantized_model)
print(f"Original model size: {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")

```


**Verify if quantization worked**

The logic here is that since GANQ uses a codebook of size (m, L) for a weight matrix for size (m,n) where L is 2^k (k = number of bits), each row in the weight matrix W should only contain values from the corressponding row in the codebook, where selection is driven by the one hot matrix S. So number of unique values in each row of W should be exactly L.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

model_q = AutoModelForCausalLM.from_pretrained(
    "ganq_quantized_smollm"
)

def verify_unique_entries_in_row(layer, row_idx=0):
    Wq = layer.self_attn.q_proj.weight.data
    unique_entries = torch.unique(Wq[row_idx])
    print(f"Number of unique entries in row {row_idx}: {unique_entries.numel()}")

verify_unique_entries_in_row(model_q.model.layers[1], row_idx=1)
verify_unique_entries_in_row(model.model.layers[1], row_idx=1)

# In my experiments, it gave this:
# Number of unique entries in row 1: 16 (since I used 4-bit quantization)
# Number of unique entries in row 1: 471
```