# nanoGPT – Agent Guide

## Where to work
* All source lives in the repository root (Python).
* Data artefacts (train.bin / val.bin) created by `data/shakespeare_char/prepare.py`.
* since we are running on CPU instead of GPU we must set both `--device=cpu`
* and also turn off PyTorch 2.0 compile with `--compile=False`.
  
## How to verify changes locally
1. Run unit smoke test (≈10 iters):
   ```bash
   python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=2 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2 --lr_decay_iters=2000 --dropout=0.0

   ```

2. Full (≈3 min on A100) Shakespeare training:

   ```bash
   python train.py config/train_shakespeare_char.py
   ```
3. Sampling from best checkpoint:

   ```bash
   python sample.py --out_dir=out-shakespeare-char --device=cpu
   ```

## Contribution & style

* Follow `black` (already in pre‑commit); run `black .` before committing.
* Keep functions <80 lines; extract helpers into `utils.py` if needed.
* Document any new CLI flag at the top of `train.py`.

## Useful CLI snippets

| Purpose                  | Command                                         |
| ------------------------ | ----------------------------------------------- |
| Profile one step         | `python bench.py`                               |
| Convert checkpoint to HF | `python convert_checkpoint.py --out_dir export` |
