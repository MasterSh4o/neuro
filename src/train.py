import os, argparse, re, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Prefer torch.amp; fallback to cuda.amp
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast
    _AMP_BACKEND = 'torch.amp'
except Exception:
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast
    _AMP_BACKEND = 'torch.cuda.amp'

from utils.common import set_seed, load_config, human_int
from utils.metrics import f1_multilabel
from utils.scheduler import CosineWarmupLR
from data.dataset import InterferogramDataset
from models.net import InterferoNetMultiLabel

# ---- Utils for AMP ----
def autocast_ctx(device_type, use_amp):
    if _AMP_BACKEND == 'torch.amp':
        return _autocast(device_type, enabled=use_amp)
    else:
        return _autocast(enabled=use_amp)

def make_scaler(device_type, use_amp):
    if _AMP_BACKEND == 'torch.amp':
        return _GradScaler(device_type, enabled=use_amp)
    else:
        return _GradScaler(enabled=use_amp)

# ---- Robust collate ----
def collate_fn(batch):
    import torch as _T
    xs, ys, sides, rels = zip(*batch)
    x = _T.stack(xs, dim=0)         # (B,1,H,W)
    y = _T.stack(ys, dim=0).float() # (B,K)
    side = None
    return x, y, side, list(rels)

# ---- Checkpoint I/O ----
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def atomic_save(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

_CKPT_RE = re.compile(r"ckpt_epoch(\d+)\.pt$")

def _list_ckpts_sorted(out_dir: str):
    items = []
    for f in os.listdir(out_dir):
        m = _CKPT_RE.match(f)
        if m:
            items.append((int(m.group(1)), f))
    items.sort(key=lambda t: t[0])
    return items

# ---- Simple multilabel stratification (greedy) ----
def iterative_train_val_split(Y, val_ratio=0.05, seed=42, max_val=5000):
    rng = np.random.default_rng(seed)
    N, K = Y.shape
    target_val = int(min(max_val, max(1, round(val_ratio * N))))
    pos_counts = Y.sum(axis=0).astype(int)
    desired = np.floor(pos_counts * (target_val / max(1, N))).astype(int)
    val_mask = np.zeros(N, dtype=bool)
    order = np.argsort(Y.sum(axis=1) + rng.random(N)*1e-6)[::-1]
    for i in order:
        if val_mask.sum() >= target_val: break
        y = Y[i].astype(int)
        if (desired > 0).any() and (y & (desired > 0)).any():
            val_mask[i] = True
            desired = np.maximum(desired - y, 0)
    if val_mask.sum() < target_val:
        remaining = np.where(~val_mask)[0]
        need = target_val - val_mask.sum()
        pick = rng.choice(remaining, size=need, replace=False)
        val_mask[pick] = True
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # Device/AMP
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if (str(device).startswith('cuda') and torch.cuda.is_available()) else 'cpu'
    use_amp = bool(cfg.get("precision", "amp") == "amp" and device_type == 'cuda')
    channels_last = bool(cfg.get("channels_last", True))
    torch_compile = bool(cfg.get("torch_compile", True) and hasattr(torch, "compile"))

    # Data cfg
    d = cfg["data"]
    root = d["root"]
    img_glob = d.get("img_glob", "**/*.png,**/*.jpg,**/*.tif,**/*.bmp")
    image_size = int(d.get("image_size", 256))
    # regex may be in config; regex-less dataset ignores it if not supported
    regex = d.get("filename_label_regex", None)
    numbers_total = int(d.get("numbers_total", 11))
    ignore_last = bool(d.get("ignore_last_number", True))
    bits = int(d.get("bits_per_number", 5))
    value_mode = d.get("value_mode", "mod")
    aug_cfg = {
        "hflip": bool(d.get("hflip", True)),
        "vflip": bool(d.get("vflip", True)),
        "rotate90": bool(d.get("rotate90", True)),
        "gaussian_noise_std": float(d.get("gaussian_noise_std", 0.01)),
        "random_gamma": list(d.get("random_gamma", [0.9, 1.1])),
    }

    full_ds = InterferogramDataset(root=root, img_glob=img_glob, image_size=image_size,
                                   filename_label_regex=regex, numbers_total=numbers_total, bits_per_number=bits,
                                   value_mode=value_mode, ignore_last_number=ignore_last,
                                   augment=True, aug_cfg=aug_cfg)
    N = len(full_ds); K = int(getattr(full_ds, "K", full_ds.labels.shape[1]))
    print(f"[INFO] Loaded dataset: {human_int(N)} samples, labels dim K={K}, from {root} (dedup enabled)")

    # Split
    Y = (full_ds.labels > 0.5).astype(int)
    train_idx, val_idx = iterative_train_val_split(Y, val_ratio=0.05, seed=cfg.get("seed", 42), max_val=5000)

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    print(f"[INFO] Train size={human_int(len(train_idx))}, Val size={human_int(len(val_idx))}")

    # Loaders
    w = int(d.get("workers", 8)); pin = bool(d.get("pin_memory", True))
    drop = bool(d.get("drop_last", True)); shuf = bool(d.get("shuffle", True))
    persistent = bool(d.get("persistent_workers", True))
    prefetch = int(d.get("prefetch_factor", 4))
    if w == 0 and persistent:
        print("[WARN] persistent_workers requires num_workers>0; forcing False")
        persistent = False

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=shuf,
                              num_workers=w, pin_memory=pin, drop_last=drop, persistent_workers=persistent,
                              prefetch_factor=prefetch if w>0 else None, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=w, pin_memory=pin, drop_last=False, persistent_workers=persistent,
                            prefetch_factor=prefetch if w>0 else None, collate_fn=collate_fn)

    # Model
    model = InterferoNetMultiLabel(out_dim=K)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    if torch_compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print("torch.compile failed, continue without:", e)

    # Optimizer & Scheduler
    t = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]),
                                  betas=tuple(t.get("betas", (0.9, 0.999))))
    scheduler = CosineWarmupLR(optimizer, T_max=int(t["epochs"]),
                               warmup_epochs=int(t.get("scheduler", {}).get("warmup_epochs", 5)),
                               min_lr=float(t.get("scheduler", {}).get("min_lr", 1e-6)))
    scaler = make_scaler(device_type, use_amp)

    # Loss with optional auto pos_weight
    bce_pos = str(t.get("bce_pos_weight", "auto"))
    pos_weight = None
    if bce_pos == "auto":
        ytr = full_ds.labels[train_idx]
        pos = ytr.sum(axis=0) + 1e-6
        neg = ytr.shape[0] - ytr.sum(axis=0) + 1e-6
        pw = (neg / pos).astype(np.float32)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        print("[INFO] Using BCE pos_weight=neg/pos (auto).")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # EMA
    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        @torch.no_grad()
        def update(self, model):
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
        @torch.no_grad()
        def apply_to(self, model):
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    v.copy_(self.shadow[k])

    ema = EMA(model, decay=float(t.get("ema_decay", 0.999)))

    # Logging dir (absolute) + ensure exists
    out_dir_cfg = cfg.get("logging", {}).get("out_dir", "runs")
    out_dir = os.path.abspath(out_dir_cfg)
    ensure_dir(out_dir)

    best_micro = 0.0

    def save_ckpt(epoch, is_best=False):
        nonlocal best_micro
        ensure_dir(out_dir)
        fname = f"ckpt_epoch{epoch:04d}.pt"   # zero-padded to avoid lexicographic pitfalls
        path  = os.path.join(out_dir, fname)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_micro": best_micro
        }
        atomic_save(state, path)

        keep_k = int(cfg.get("logging", {}).get("keep_last_k", 5))
        if keep_k > 0:
            ckpts = _list_ckpts_sorted(out_dir)  # numeric sort by epoch
            if len(ckpts) > keep_k:
                for _, f in ckpts[:-keep_k]:
                    try: os.remove(os.path.join(out_dir, f))
                    except: pass

        if is_best:
            atomic_save(state, os.path.join(out_dir, "best.pt"))

    # Train
    epochs = int(t["epochs"]); accum = int(t.get("grad_accum_steps", 1))
    max_grad_norm = float(t.get("max_grad_norm", 0.0))
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0; nb = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
        for it, (x, y, side, _) in enumerate(pbar):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if channels_last: x = x.contiguous(memory_format=torch.channels_last)
            with autocast_ctx(device_type, use_amp):
                logits = model(x)              # (B,K)
                loss = criterion(logits, y) / accum
            scaler.scale(loss).backward()
            if (it + 1) % accum == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            tot_loss += loss.item() * accum; nb += 1
            pbar.set_postfix(loss=f"{tot_loss/nb:.4f}")
        scheduler.step()

        # Validate with EMA
        ema.apply_to(model); model.eval()
        val_loss = 0.0; n_val = 0; f1m = []; f1M = []
        with torch.no_grad():
            for (x, y, side, _) in tqdm(val_loader, desc="Valid", ncols=120):
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                if channels_last: x = x.contiguous(memory_format=torch.channels_last)
                with autocast_ctx(device_type, use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                fmicro, fmacro = f1_multilabel(logits, y, threshold=threshold)
                f1m.append(fmicro); f1M.append(fmacro)
                n_val += x.size(0)
        val_loss /= max(1, n_val)
        import numpy as _np
        f1_micro = float(_np.mean(f1m)) if f1m else 0.0
        f1_macro = float(_np.mean(f1M)) if f1M else 0.0
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f} f1_micro={f1_micro*100:.2f}% f1_macro={f1_macro*100:.2f}%")

        is_best = f1_micro > best_micro
        if is_best: best_micro = f1_micro
        if ((epoch + 1) % int(cfg.get('logging', {}).get('save_every',1)) == 0) or is_best:
            save_ckpt(epoch+1, is_best=is_best)

    print("Done.")

if __name__ == "__main__":
    main()
