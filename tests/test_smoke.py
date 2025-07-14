import subprocess
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE = Path(__file__).resolve().parents[1] / ".hf_cache"


def test_gpt2_cache():
    """Ensure GPT-2 weights are cached and can be loaded."""
    model_dir = HF_CACHE / "models--openai-community--gpt2"
    assert model_dir.exists(), f"{model_dir} missing"

    AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=HF_CACHE)
    AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=HF_CACHE)


def test_shakespeare_prepare(tmp_path):
    subprocess.check_call(["python", "data/shakespeare_char/prepare.py"])
    assert Path("data/shakespeare_char/train.bin").exists()
    assert Path("data/shakespeare_char/val.bin").exists()


def test_train_smoke(tmp_path, monkeypatch):
    out_dir = tmp_path / "out-test"
    cmd = [
        "python",
        "train.py",
        "config/train_shakespeare_char.py",
        "--device=cpu",
        "--compile=False",
        "--eval_interval=1",
        "--eval_iters=2",
        "--log_interval=1",
        "--block_size=64",
        "--batch_size=12",
        "--n_layer=4",
        "--n_head=4",
        "--n_embd=128",
        "--max_iters=2",
        "--lr_decay_iters=2000",
        "--dropout=0.0",
        "--always_save_checkpoint=True",
        f"--out_dir={out_dir}",
        "--wandb_log=True",
        "--wandb_project=tests",
        "--wandb_run_name=smoke",
    ]
    monkeypatch.setenv("HF_HOME", str(HF_CACHE))
    monkeypatch.setenv("WANDB_MODE", "offline")
    subprocess.check_call(cmd)
    assert (out_dir / "ckpt.pt").exists()
