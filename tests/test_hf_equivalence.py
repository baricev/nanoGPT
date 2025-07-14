import os
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import GPT
from hf_compare import compare_state_dict, compare_forward

HF_CACHE = Path(__file__).resolve().parents[1] / ".hf_cache"


def test_hf_equivalence(monkeypatch):
    monkeypatch.setenv("HF_HOME", str(HF_CACHE))
    local = GPT.from_pretrained("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", cache_dir=HF_CACHE)
    compare_state_dict(local, hf)
    compare_forward(local, hf)
