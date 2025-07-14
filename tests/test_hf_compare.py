"""Compare nanoGPT's GPT-2 implementation with HuggingFace's reference."""

import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import GPT

TRANSPOSED = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]


def compare_state_dict(local_model: GPT, hf_model: GPT2LMHeadModel) -> None:
    """Assert that all parameters match between models."""
    sd_local = {
        k: v
        for k, v in local_model.state_dict().items()
        if not k.endswith(".attn.bias")
    }
    sd_hf = {
        k: v
        for k, v in hf_model.state_dict().items()
        if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
    }
    assert sd_local.keys() == sd_hf.keys(), "Mismatched parameter keys"
    for k in sd_local:
        hv = sd_hf[k].t() if any(k.endswith(w) for w in TRANSPOSED) else sd_hf[k]
        torch.testing.assert_close(sd_local[k], hv, atol=1e-6, rtol=1e-6)


def compare_forward(local_model: GPT, hf_model: GPT2LMHeadModel) -> None:
    """Run a deterministic forward pass and compare logits."""
    local_model.eval()
    hf_model.eval()
    idx = torch.randint(0, local_model.config.vocab_size, (2, 8))
    with torch.no_grad():
        logits_local, _ = local_model(idx)
        logits_hf = hf_model(idx).logits[:, [-1], :]
    torch.testing.assert_close(logits_local, logits_hf, atol=1e-3, rtol=1e-3)


def evaluate_shakespeare(
    local_model: GPT, hf_model: GPT2LMHeadModel, data_dir: str
) -> None:
    """Compute a quick validation loss on the Shakespeare dataset and compare."""
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    block_size = local_model.config.block_size
    batch_size = 4
    eval_iters = 2
    criterion = torch.nn.CrossEntropyLoss()
    losses_local = []
    losses_hf = []
    for _ in range(eval_iters):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((val_data[i : i + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (val_data[i + 1 : i + 1 + block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        with torch.no_grad():
            logits_local, _ = local_model(x)
            logits_hf = hf_model(x).logits
            loss_local = criterion(
                logits_local.view(-1, logits_local.size(-1)), y.view(-1)
            )
            loss_hf = criterion(logits_hf.view(-1, logits_hf.size(-1)), y.view(-1))
        losses_local.append(loss_local.item())
        losses_hf.append(loss_hf.item())
    mean_local = sum(losses_local) / eval_iters
    mean_hf = sum(losses_hf) / eval_iters
    print(f"local val loss: {mean_local:.4f}")
    print(f"hf val loss: {mean_hf:.4f}")
    torch.testing.assert_close(
        torch.tensor(mean_local), torch.tensor(mean_hf), atol=1e-3, rtol=1e-3
    )


def test_hf_comparison() -> None:
    """Runs all HuggingFace comparison tests."""
    cache_dir = os.environ.get(
        "HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache")
    )
    local_model = GPT.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", cache_dir=cache_dir
    )
    compare_state_dict(local_model, hf_model)
    compare_forward(local_model, hf_model)
    # The shakespeare dataset is not available in the test environment, so this is commented out.
    # evaluate_shakespeare(
    #     local_model, hf_model, os.path.join("data", "shakespeare_char")
    # )
