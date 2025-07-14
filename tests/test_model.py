import math
import os
import sys

import torch
from transformers import GPT2LMHeadModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import GPT, GPTConfig, CausalSelfAttention


def test_weight_tying():
    cfg = GPTConfig(block_size=8, vocab_size=16, n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg)
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_cproj_init_std():
    cfg = GPTConfig(n_layer=4)
    model = GPT(cfg)
    w = dict(model.named_parameters())["transformer.h.0.attn.c_proj.weight"]
    expected = 0.02 / math.sqrt(2 * model.config.n_layer)
    assert torch.isclose(w.std(), torch.tensor(expected), atol=1e-3)


def test_crop_block_size():
    cfg = GPTConfig(block_size=32, n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg)
    model.crop_block_size(16)
    assert model.transformer.wpe.weight.shape[0] == 16
    for block in model.transformer.h:
        if hasattr(block.attn, "bias"):
            assert block.attn.bias.size(-1) == 16


def test_optimizer_grouping():
    cfg = GPTConfig(n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg)
    opt = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )
    zero_decay = [g for g in opt.param_groups if g["weight_decay"] == 0.0][0]["params"]
    biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
    assert all(any(b is z for z in zero_decay) for b in biases)


def test_num_params_excludes_wpe():
    cfg = GPTConfig(block_size=32, vocab_size=64, n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg)
    total = model.get_num_params(non_embedding=False)
    non_embed = model.get_num_params(non_embedding=True)
    assert total - non_embed == model.transformer.wpe.weight.numel()


def test_generate_crops_context():
    cfg = GPTConfig(block_size=4, n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg)
    lengths = []
    orig_forward = model.forward

    def hook(idx, targets=None):
        lengths.append(idx.shape[1])
        return orig_forward(idx, targets)

    model.forward = hook
    idx = torch.randint(cfg.vocab_size, (1, 8))
    model.generate(idx, max_new_tokens=2)
    assert all(l <= cfg.block_size for l in lengths)


def test_from_pretrained_transposes():
    hf_model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", cache_dir="nanoGPT/.hf_cache"
    )
    model = GPT.from_pretrained("gpt2")
    w_hf = hf_model.transformer.h[0].attn.c_attn.weight
    w_my = model.transformer.h[0].attn.c_attn.weight
    assert torch.allclose(w_my, w_hf.t())


def test_bias_flag_off():
    cfg = GPTConfig(bias=False, n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg)
    for module in model.modules():
        if isinstance(
            module, (torch.nn.Linear, torch.nn.modules.normalization.LayerNorm)
        ):
            assert module.bias is None


def test_attention_split_shapes():
    cfg = GPTConfig(n_layer=1, n_head=2, n_embd=8, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(1, 3, cfg.n_embd)
    with torch.no_grad():
        y = attn(x)
    qkv = attn.c_attn(x)
    C = cfg.n_embd
    q, k, v = qkv.split(C, dim=2)
    k = k.view(1, 3, cfg.n_head, C // cfg.n_head).transpose(1, 2)
    q = q.view(1, 3, cfg.n_head, C // cfg.n_head).transpose(1, 2)
    v = v.view(1, 3, cfg.n_head, C // cfg.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    bias = torch.tril(torch.ones(3, 3)).view(1, 1, 3, 3)
    att = att.masked_fill(bias[:, :, :3, :3] == 0, float("-inf"))
    att = torch.nn.functional.softmax(att, dim=-1)
    y_manual = att @ v
    y_manual = y_manual.transpose(1, 2).contiguous().view(1, 3, C)
    y_manual = attn.c_proj(y_manual)
    assert torch.allclose(y, y_manual, atol=1e-5)



def test_query_projection_shapes():
    cfg = GPTConfig(n_embd=32, n_head=4)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 5, cfg.n_embd)
    outputs = {}

    def hook(module, input, output):
        outputs["proj"] = output

    h = attn.c_attn.register_forward_hook(hook)
    attn(x)
    h.remove()
    q, k, v = outputs["proj"].split(cfg.n_embd, dim=2)
    assert q.shape ==  (2, 5, cfg.n_embd)


    qh = q.view(2, 5, cfg.n_head, cfg.n_embd // cfg.n_head).transpose(1, 2)
    assert qh.shape ==  (2, cfg.n_head, 5, cfg.n_embd // cfg.n_head)

def test_layer_construction_shapes():
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=8, block_size=16, vocab_size=20)
    model = GPT(cfg)
    assert len(model.transformer.h) == cfg.n_layer
    block = model.transformer.h[0]
    assert block.attn.c_attn.weight.shape == (3 * cfg.n_embd, cfg.n_embd)
    assert block.attn.c_proj.weight.shape == (cfg.n_embd, cfg.n_embd)
    assert block.mlp.c_fc.weight.shape == (4 * cfg.n_embd, cfg.n_embd)
    assert model.transformer.wte.weight.shape == (cfg.vocab_size, cfg.n_embd)
    assert model.transformer.wpe.weight.shape == (cfg.block_size, cfg.n_embd)



def test_pretrained_parameter_names_and_shapes():
    model = GPT.from_pretrained("gpt2")
    from transformers import GPT2LMHeadModel

    hf_model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", cache_dir=".hf_cache", local_files_only=True
    )
    names_ours = [n for n, _ in model.named_parameters() if not n.endswith("attn.bias")]
    names_hf = [
        n
        for n, _ in hf_model.named_parameters()
        if not n.endswith("attn.bias") and not n.endswith("attn.masked_bias")
    ]
    assert len(names_ours) == len(names_hf)
    for n_ours, n_hf in zip(names_ours, names_hf):
        p_ours = dict(model.named_parameters())[n_ours]
        p_hf = dict(hf_model.named_parameters())[n_hf]
        if any(
            n_hf.endswith(suffix)
            for suffix in [
                "attn.c_attn.weight",
                "attn.c_proj.weight",
                "mlp.c_fc.weight",
                "mlp.c_proj.weight",
            ]
        ):
            assert p_ours.shape == p_hf.t().shape
        else:
            assert p_ours.shape == p_hf.shape
