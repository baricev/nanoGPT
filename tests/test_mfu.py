import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GPT, GPTConfig


def test_estimate_mfu():
    config = GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = GPT(config)
    mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=1.0)
    assert isinstance(mfu, float)
    assert 0.0 <= mfu <= 1.0
