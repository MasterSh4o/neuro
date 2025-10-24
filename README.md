# Interferogram CNN — Yandex DataSphere (g2.1, 80GB VRAM)

Universal PyTorch project for **300,000** grayscale interferograms of size **256×256** with multi-label classification.
Optimized for **Yandex DataSphere** `g2.1` (A100 80GB) with mixed precision, advanced scheduling, enhanced architecture, and comprehensive logging.

## Highlights
- Grayscale input (1×256×256), configurable multi-label classes.
- **Enhanced architecture**: 3‑layer MLP head with configurable hidden dimensions, dropout, and normalization.
- **Advanced scheduling**: Cosine warmup OR WarmupReduceLROnPlateau for adaptive learning rates.
- **Label smoothing**: Configurable BCE with logits smoothing for better generalization.
- **Gradient clipping**: Robust training with non‑finite gradient detection and skipping.
- AMP (`torch.cuda.amp`), channels‑last, gradient accumulation.
- EMA, AdamW, comprehensive logging (TensorBoard + JSON).
- **Environment‑aware**: Dedicated DataSphere configuration with workspace paths.
- Config‑driven (`configs/default.yaml`, `configs/datasphere.yaml`).

## Quick start (DataSphere, single GPU g2.1 80GB)

### Method 1: Using the DataSphere launcher
```bash
# 1) Install dependencies
bash scripts/setup_env.sh

# 2) Run with DataSphere config (automatically sets paths)
bash scripts/run_datasphere.sh --config configs/datasphere.yaml

# 3) Resume from checkpoint
bash scripts/run_datasphere.sh --config configs/datasphere.yaml --resume runs/best.pt
```

### Method 2: Manual execution
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Train with local config
python -u src/train.py --config configs/local.yaml

# 3) Train with DataSphere config
python -u src/train.py --config configs/datasphere.yaml

# 4) Resume
python -u src/train.py --config configs/datasphere.yaml --resume runs/best.pt

# 5) TensorBoard
tensorboard --logdir runs
```

## Configuration options

### Model architecture (new parameters)
```yaml
model:
  hidden_dims: [1024, 512, 256]    # MLP head hidden layers
  head_dropout: 0.3                # Dropout between head layers
  head_norm: "layernorm"           # Normalization: layernorm/batchnorm1d/none
```

### Training enhancements
```yaml
train:
  current_epoch: 51                # Resume from epoch (progress tracking)
  label_smoothing: 0.05            # BCE smoothing ε
  loss_reduction: "mean"           # Loss reduction mode
  scheduler:
    name: "warmup_plateau"         # New adaptive scheduler
    warmup_epochs: 3
    factor: 0.5
    patience: 2
    mode: "max"
    metric: "f1_micro"
```

### DataSphere‑specific settings
```yaml
workspace_root: "/home/jupyter/work"  # DataSphere workspace
data:
  root: "/home/jupyter/work/datasets/InterfDataset"
logging:
  out_dir: "runs"                     # Relative to workspace_root
```

## Expected data layout

The system expects bit‑encoded labels in filenames with configurable regex:
```
/path/to/data/1_5_12_0_8.png
# → Extract numbers [1,5,12,0,8] → Convert to binary vector
```

Configuration options:
```yaml
data:
  label_parsing:
    filename_label_regex: "(\d{1,2})"
    numbers_total: 11
    bits_per_number: 5
    ignore_last_number: true
    value_mode: "mod"        # mod/clip/raise
```

## Advanced features

### WarmupReduceLROnPlateau scheduler
Combines linear warmup with ReduceLROnPlateau for adaptive LR scheduling:
```yaml
train:
  scheduler:
    name: "warmup_plateau"
    warmup_epochs: 3
    factor: 0.5
    patience: 2
    cooldown: 1
    min_lr: 1.0e-6
    metric: "f1_micro"
```

### Label smoothing
Reduces overconfidence in multi‑label classification:
```yaml
train:
  label_smoothing: 0.05  # ε ∈ [0,1]
```

### Gradient clipping with logging
Automatic detection and skipping of non‑finite gradients with detailed logging:
```yaml
train:
  max_grad_norm: 1.0
```

## Project tree
```
configs/
  default.yaml          # Base configuration
  local.yaml            # Local development settings
  datasphere.yaml       # DataSphere g2.1 optimized settings
docs/
  datasphere_plan.md    # Implementation plan and tracking
scripts/
  run_datasphere.sh     # DataSphere launcher with environment setup
  setup_env.sh          # Dependency installation
src/
  train.py              # Main training script with enhanced features
  data/dataset.py       # Bit‑encoded label parsing
  models/net.py         # Enhanced CNN with configurable MLP head
  utils/
    scheduler.py        # CosineWarmupLR + WarmupReduceLROnPlateau
    metrics.py          # Multi‑label metrics
    common.py           # Utilities
requirements.txt
```

## Training progress tracking

The system automatically tracks and resumes training progress:
- `current_epoch` in configuration specifies starting epoch
- Checkpoints include full state (model, optimizer, scheduler, scaler, EMA)
- Both `CosineWarmupLR` and `WarmupReduceLROnPlateau` are serializable
- Progress is logged to TensorBoard and JSON history files

---

**Updated for DataSphere g2.1** with enhanced architecture, adaptive scheduling, label smoothing, and comprehensive progress tracking.
