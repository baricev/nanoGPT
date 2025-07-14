import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model import GPT, GPTConfig


def test_generate_length():
    config = GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = GPT(config)
    prompt = torch.randint(0, config.vocab_size, (1, 1))
    out = model.generate(prompt, max_new_tokens=3)
    assert out.shape[1] == prompt.shape[1] + 3
