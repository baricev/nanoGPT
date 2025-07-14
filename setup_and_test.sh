#!/bin/bash
set -e

# Install dependencies
pip install --no-cache-dir torch numpy transformers datasets tiktoken wandb tqdm requests black

# Choose a cache location for HuggingFace assets
export HF_HOME="$PWD/.hf_cache"
mkdir -p "$HF_HOME"

# Pre-download GPT-2 small weights and tokenizer
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
cache = os.environ["HF_HOME"]
AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=cache)
AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=cache)
print("Loaded model and tokenizer successfully")
PY

# Download and preprocess the Shakespeare dataset
python data/shakespeare_char/prepare.py
