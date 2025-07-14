import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model import GPT, GPTConfig


def test_forward_and_generate_shapes():
    config = GPTConfig(
        vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=0.0
    )
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss.shape == ()
    prompt = torch.randint(0, config.vocab_size, (1, 2))
    out = model.generate(prompt, max_new_tokens=3)
    assert out.shape[1] == prompt.shape[1] + 3
