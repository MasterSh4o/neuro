# Interferogram CNN — Yandex DataSphere (g2.1, 80GB VRAM)

Universal PyTorch project for **300,000** grayscale interferograms of size **256×256** with optional **80-d side features** and multi-class classification.
Optimized for **Yandex DataSphere** `g2.1` (A100 80GB) with mixed precision, gradient accumulation, cosine LR with warmup, channels-last, and `torch.compile` (optional).

## Highlights
- Grayscale input (1×256×256), configurable classes.
- Optional 80-d side vector per sample (fused via MLP).
- Efficient residual CNN with Squeeze-and-Excitation.
- AMP (`torch.cuda.amp`), channels-last, gradient accumulation.
- Cosine LR with warmup, EMA, label smoothing, AdamW.
- Robust DataLoader (prefetch, pin_memory, persistent_workers).
- TensorBoard + CSV logging, graceful resume, checkpoints.
- Config-driven (`configs/default.yaml`).

## Expected data layout
You can choose **any** of the following (declare in config):
1. **Image folders**: `root/cls_x/*.png` or `root/cls_x/*.jpg` (grayscale).  
2. **Numpy arrays**: `root/images/*.npy` (H×W or 1×H×W), labels in `labels.csv` with columns `path,label`.
3. **Memmap/LMDB** (advanced): adapt `dataset.py` templates.

Optional side-features (80-d): provide `side_features.csv` with columns `path,f0,...,f79`.
Paths in CSV should match relative paths used by the dataset.

## Quick start (DataSphere, single GPU g2.1 80GB)
```bash
# 1) In a terminal cell
pip install -r requirements.txt

# 2) Edit config
cp configs/default.yaml configs/local.yaml
# set data.root, data.format, data.num_classes, etc.

# 3) Train
python -u src/train.py --config configs/local.yaml

# 4) Resume
python -u src/train.py --config configs/local.yaml --resume path/to/checkpoint.pt

# 5) TensorBoard
tensorboard --logdir runs
```

## Object Storage (optional)
If your data is in Yandex Object Storage (S3-compatible), mount or download locally in a notebook cell.
You can pass absolute paths into `configs/local.yaml` after mounting.

## Notes for 300k samples
- Start with `batch_size: 256` on A100 80GB; increase if memory allows.
- Use `precision: "amp"` and `channels_last: true`.
- Enable `grad_accum_steps` for larger effective batch sizes if needed.
- Keep `num_workers` ≤ CPU cores; `persistent_workers: true` for speed.

## Project tree
```
configs/
  default.yaml
scripts/
  run_datasphere.sh
src/
  train.py
  data/dataset.py
  models/net.py
  utils/scheduler.py
  utils/metrics.py
  utils/common.py
requirements.txt
```

---

**Authoring notes:** this template incorporates issues discussed in prior chats: warmup scheduler, grayscale inputs, optional 80-d extras, F1 micro/macro, large-batch stability, and robust logging/resume.
