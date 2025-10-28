import argparse
import json
import math
import os
import re
import time
import hashlib
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast
    _AMP_BACKEND = "torch.amp"
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast
    _AMP_BACKEND = "torch.cuda.amp"

from utils.common import set_seed, load_config, human_int
from utils.metrics import MultilabelMetrics
from utils.scheduler import CosineWarmupLR, WarmupReduceLROnPlateau, WarmupReduceLROnPlateauConfig
from data.dataset import InterferogramDataset
from data.dataset_reference_simple import ReferenceInterferogramDataset
from models.net import InterferoNetMultiLabel


def autocast_ctx(device_type: str, use_amp: bool):
    if _AMP_BACKEND == "torch.amp":
        return _autocast(device_type, enabled=use_amp)
    return _autocast(enabled=use_amp)


def make_scaler(device_type: str, use_amp: bool):
    if _AMP_BACKEND == "torch.amp":
        return _GradScaler(device_type, enabled=use_amp)
    return _GradScaler(enabled=use_amp)


def collate_fn(batch):
    import torch as _T
    xs, ys, sides, rels = zip(*batch)
    x = _T.stack(xs, dim=0)
    y = _T.stack(ys, dim=0).float()
    side = None
    return x, y, side, list(rels)


def reference_collate_fn(batch):
    """Collate function for reference interferogram datasets."""
    import torch as _T

    def _ensure_nchw(tensor: _T.Tensor) -> _T.Tensor:
        """Make sure a batched tensor follows [N, C, H, W] layout when possible."""
        if tensor.dim() == 5 and tensor.size(1) == 1:
            tensor = tensor.squeeze(1)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        return tensor

    def _to_label_tensor(item: Any, fallback_dim: int = 80) -> _T.Tensor:
        if isinstance(item, _T.Tensor):
            return item.float()
        if item is None:
            return _T.zeros(fallback_dim, dtype=_T.float32)
        return _T.from_numpy(np.asarray(item)).float()

    def _stack_labels(items: Sequence[Any], label_dim: int) -> _T.Tensor:
        tensors: list[_T.Tensor] = []
        for value in items:
            tensor = _to_label_tensor(value, label_dim)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        return _T.stack(tensors, dim=0)

    # Handle different return formats from dataset
    if len(batch[0]) == 3:  # (img, diff_or_dual_or_label, mode_info)
        xs, second_inputs, mode_infos = zip(*batch)
        mode_infos = [mi or {} for mi in mode_infos]

        x = _T.stack(xs, dim=0).float()
        x = _ensure_nchw(x)

        label_dim = int(mode_infos[0].get('label_dim', 80)) if mode_infos else 80

        first_second = second_inputs[0]
        is_tensor = isinstance(first_second, _T.Tensor)
        treat_as_image = is_tensor and first_second.dim() >= 3

        dual_input = None
        y: Optional[_T.Tensor] = None

        if treat_as_image:
            stacked = _T.stack([si.float() for si in second_inputs], dim=0)
            stacked = _ensure_nchw(stacked)
            if stacked.dim() == 4:
                dual_input = stacked
            else:
                # Fall back to treating the secondary input as labels if shape is unexpected
                treat_as_image = False

        if not treat_as_image:
            # Interpret secondary input as labels or label-like tensors
            y = _stack_labels(second_inputs, label_dim)
            dual_input = None
    else:
        # Standard format: (x, y, side, rels)
        xs, ys, sides, rels = zip(*batch)
        x = _T.stack(xs, dim=0).float()
        x = _ensure_nchw(x)
        mode_infos = [ri or {} for ri in rels]
        label_dim = int(mode_infos[0].get('label_dim', 80)) if mode_infos else 80
        y = _stack_labels(ys, label_dim)
        dual_input = None

    # Extract labels from mode_info if they were not supplied directly
    if (y is None or not isinstance(y, _T.Tensor)) and mode_infos:
        label_dim = int(mode_infos[0].get('label_dim', 80)) if mode_infos else 80
        extracted_labels: list[_T.Tensor] = []
        for mode_info in mode_infos:
            label = mode_info.get('label') if mode_info else None
            if label is None:
                extracted_labels = []
                break
            extracted_labels.append(_to_label_tensor(label, label_dim))

        if extracted_labels:
            y = _T.stack(extracted_labels, dim=0)

    if y is None or not isinstance(y, _T.Tensor):
        label_dim = int(mode_infos[0].get('label_dim', 80)) if mode_infos else 80
        y = _T.zeros(x.size(0), label_dim, dtype=_T.float32)

    return x, y.float(), dual_input, mode_infos


def _ensure_batch_labels(
    labels: Any,
    batch_size: int,
    mode_infos: Optional[Sequence[Dict[str, Any]]] = None,
    default_dim: int = 80,
) -> torch.Tensor:
    """Convert labels coming from a dataloader into a float tensor batch."""

    label_dim = default_dim
    if mode_infos:
        try:
            inferred = int(mode_infos[0].get("label_dim", label_dim))
            if inferred > 0:
                label_dim = inferred
        except Exception:
            pass

    def _coerce_single(item: Any) -> torch.Tensor:
        if isinstance(item, torch.Tensor):
            tensor = item.float()
        else:
            tensor = torch.from_numpy(np.asarray(item)).float()
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    if isinstance(labels, torch.Tensor):
        tensor = labels.float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    if isinstance(labels, (list, tuple)):
        tensors = [_coerce_single(item) for item in labels if item is not None]
        if tensors:
            # Ensure consistent shape across stacked tensors
            base_shape = tensors[0].shape
            tensors = [
                t.view(base_shape) if t.shape != base_shape else t for t in tensors
            ]
            stacked = torch.stack(tensors, dim=0)
            return stacked.float()
        return torch.zeros(batch_size, label_dim, dtype=torch.float32)

    tensor = _coerce_single(labels)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.size(0) != batch_size:
        tensor = tensor.expand(batch_size, -1)
    return tensor.float()


def mixup_data(x, y, alpha=0.2):
    """Реализация Mixup регуляризации"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Критерий для Mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.smoothing = float(smoothing)
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0.0:
            return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)
        targets_smooth = targets * (1.0 - self.smoothing) + self.smoothing
        loss = self.bce(logits, targets_smooth)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def atomic_save(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


_CKPT_RE = re.compile(r"ckpt_epoch(\d+)\.pt$")


def _list_ckpts_sorted(out_dir: str) -> list[Tuple[int, str]]:
    items: list[Tuple[int, str]] = []
    if not os.path.isdir(out_dir):
        return items
    for name in os.listdir(out_dir):
        match = _CKPT_RE.match(name)
        if match:
            items.append((int(match.group(1)), name))
    items.sort(key=lambda t: t[0])
    return items


def dataset_signature(paths: Sequence[str]) -> str:
    hasher = hashlib.sha1()
    for path in paths:
        hasher.update(path.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> np.ndarray:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if (ratios < 0).any():
        raise ValueError("Split ratios must be non-negative.")
    total = ratios.sum()
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    return ratios / total


def _compute_split_counts(ratios: np.ndarray, total: int, max_val: Optional[int]) -> np.ndarray:
    raw = ratios * total
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    counts = np.maximum(counts, 0)
    if max_val is not None and counts[1] > max_val:
        excess = counts[1] - max_val
        counts[1] = max_val
        counts[0] += excess
    counts[2] = max(total - counts[0] - counts[1], 0)
    if counts.sum() < total:
        counts[0] += total - counts.sum()
    return counts


def _iterative_pick(Y: np.ndarray, target: int, seed: int, max_pick: Optional[int] = None) -> np.ndarray:
    target = int(target)
    if target <= 0 or Y.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if max_pick is not None:
        target = min(target, max_pick)
    target = min(target, Y.shape[0])
    rng = np.random.default_rng(seed)
    pos_counts = Y.sum(axis=0).astype(int)
    desired = np.floor(pos_counts * (target / max(1, Y.shape[0]))).astype(int)
    mask = np.zeros(Y.shape[0], dtype=bool)
    order = np.argsort(Y.sum(axis=1) + rng.random(Y.shape[0]) * 1e-6)[::-1]
    for idx in order:
        if mask.sum() >= target:
            break
        y = Y[idx].astype(int)
        if (desired > 0).any():
            if (y & (desired > 0)).any():
                mask[idx] = True
                desired = np.maximum(desired - y, 0)
        else:
            mask[idx] = True
    if mask.sum() < target:
        remaining = np.where(~mask)[0]
        need = target - mask.sum()
        if need > 0 and remaining.size > 0:
            pick = rng.choice(remaining, size=need, replace=False)
            mask[pick] = True
    return np.sort(np.where(mask)[0])


def _random_split(num_samples: int, counts: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    train_count, val_count, test_count = counts.tolist()
    train_idx = np.sort(indices[:train_count])
    val_idx = np.sort(indices[train_count:train_count + val_count])
    test_idx = np.sort(indices[train_count + val_count:train_count + val_count + test_count])
    return train_idx, val_idx, test_idx


def stratified_train_val_test_split(
    Y: np.ndarray,
    split_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stratified = bool(split_cfg.get("stratified", True))
    ratios = _normalize_ratios(
        float(split_cfg.get("train_ratio", 0.6)),
        float(split_cfg.get("val_ratio", 0.2)),
        float(split_cfg.get("test_ratio", 0.2)),
    )
    counts = _compute_split_counts(ratios, Y.shape[0], split_cfg.get("max_val_samples", None))
    if not stratified:
        return _random_split(Y.shape[0], counts, seed)
    val_idx = _iterative_pick(Y, counts[1], seed + 101, max_pick=counts[1])
    remaining_mask = np.ones(Y.shape[0], dtype=bool)
    remaining_mask[val_idx] = False
    remaining_idx = np.where(remaining_mask)[0]
    if counts[2] > 0:
        test_rel = _iterative_pick(Y[remaining_idx], counts[2], seed + 211)
        test_idx = remaining_idx[test_rel]
        train_mask = np.ones(remaining_idx.shape[0], dtype=bool)
        train_mask[test_rel] = False
        train_idx = remaining_idx[train_mask]
    else:
        test_idx = np.empty(0, dtype=np.int64)
        train_idx = remaining_idx
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def split_with_cache(
    Y: np.ndarray,
    split_cfg: Dict[str, Any],
    seed: int,
    signature: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    cache_dir = split_cfg.get("cache_dir")
    cache_file: Optional[Path] = None
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        if not cache_path.is_absolute():
            cache_path = (Path.cwd() / cache_path).resolve()
        ensure_dir(str(cache_path))
        cache_file = cache_path / f"split_seed{seed}_{signature}.npz"
        if split_cfg.get("reuse_cached", True) and cache_file.exists():
            data = np.load(cache_file, allow_pickle=False)
            sig = data["signature"]
            sig_value = sig.item() if hasattr(sig, "item") else str(sig)
            if sig_value == signature:
                return (
                    data["train"].astype(np.int64),
                    data["val"].astype(np.int64),
                    data["test"].astype(np.int64),
                    str(cache_file),
                )
    train_idx, val_idx, test_idx = stratified_train_val_test_split(Y, split_cfg, seed)
    if cache_file:
        np.savez(
            cache_file,
            train=train_idx,
            val=val_idx,
            test=test_idx,
            signature=np.array(signature),
            seed=np.array(seed, dtype=np.int64),
        )
        return train_idx, val_idx, test_idx, str(cache_file)
    return train_idx, val_idx, test_idx, None


def collect_environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "num_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": datetime.now().isoformat(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def dump_json(data: Any, path: Path) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def dump_config_yaml(cfg: Dict[str, Any], path: Path) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def setup_tensorboard(tb_cfg: Dict[str, Any], out_dir: Path, run_name: str):
    if not tb_cfg.get("enabled", False):
        return None
    if SummaryWriter is None:
        print("[WARN] TensorBoard support unavailable (torch.utils.tensorboard not installed).")
        return None
    tb_dir = Path(tb_cfg.get("log_dir", out_dir / "tensorboard"))
    if not tb_dir.is_absolute():
        tb_dir = (out_dir / tb_dir).resolve()
    tb_dir = tb_dir / run_name
    ensure_dir(str(tb_dir))
    return SummaryWriter(log_dir=str(tb_dir))


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow and v.dtype.is_floating_point
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        for k, v in model.state_dict().items():
            if k in self.backup and v.dtype.is_floating_point:
                v.copy_(self.backup[k])
        self.backup = {}

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                v.copy_(self.shadow[k])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        self.apply_to(model)

    def average_parameters(self, model: nn.Module):
        return _EMAContext(self, model)


class _EMAContext:
    def __init__(self, ema: EMA, model: nn.Module):
        self.ema = ema
        self.model = model

    def __enter__(self):
        self.ema.store(self.model)
        self.ema.apply_to(self.model)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ema.restore(self.model)


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    device_type: str,
    use_amp: bool,
    threshold: float,
    channels_last: bool,
    use_reference: bool = False,
) -> Dict[str, Any]:
    if loader is None:
        return {}
    model.eval()
    total_loss = 0.0
    total_samples = 0
    num_classes = getattr(loader.dataset, "K", 80)  # Default to 80 for Korsch reference
    meter = MultilabelMetrics(num_classes=num_classes, threshold=threshold, device=device)

    with torch.no_grad():
        for batch_data in loader:
            if use_reference and len(batch_data) == 4:  # (x, y, dual_input, mode_info)
                x, y, dual_input, mode_info = batch_data
                y = _ensure_batch_labels(y, x.size(0), mode_info)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if dual_input is not None:
                    dual_input = dual_input.to(device, non_blocking=True)

                if channels_last:
                    # Применяем channels_last только к 4D тензорам [N, C, H, W]
                    if x.dim() == 4:
                        x = x.contiguous(memory_format=torch.channels_last)
                    else:
                        print(f"[WARN EVAL] Unexpected x tensor shape: {x.shape}, expected 4D for channels_last")

                    if dual_input is not None:
                        if dual_input.dim() == 4:
                            dual_input = dual_input.contiguous(memory_format=torch.channels_last)
                        else:
                            print(f"[WARN EVAL] Unexpected dual_input tensor shape: {dual_input.shape}, expected 4D for channels_last")
                            # Попробуем исправить размерность если возможно
                            if dual_input.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                                dual_input = dual_input.unsqueeze(0)
                                print(f"[DEBUG EVAL] Added batch dimension to dual_input: {dual_input.shape}")
                                if dual_input.dim() == 4:
                                    dual_input = dual_input.contiguous(memory_format=torch.channels_last)
                                    print(f"[DEBUG EVAL] Successfully applied channels_last to fixed dual_input")

                with autocast_ctx(device_type, use_amp):
                    # Model can handle dual input or single input
                    if dual_input is not None and hasattr(model, 'forward') and 'dual_input' in model.forward.__code__.co_varnames:
                        logits = model(x, dual_input)
                    elif dual_input is not None:
                        # Concatenate dual input if model expects single input
                        combined = torch.cat([x, dual_input], dim=1) if dual_input.shape[1] == 1 else dual_input
                        logits = model(combined)
                    else:
                        logits = model(x)

                    loss = criterion(logits, y)
            else:
                # Standard evaluation
                x, y, _, rels = batch_data
                y = _ensure_batch_labels(y, x.size(0), rels)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                with autocast_ctx(device_type, use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

            total_loss += float(loss.item()) * y.size(0)
            total_samples += y.size(0)
            meter.update(logits, y)

    metrics = meter.compute()
    metrics["loss"] = float(total_loss / max(1, total_samples))
    return metrics


def save_metrics_history(history: Sequence[Dict[str, Any]], path: Path) -> None:
    dump_json(list(history), path)


def _format_metrics(metrics: Dict[str, Any]) -> str:
    parts = [
        f"loss={metrics.get('loss', 0.0):.4f}",
        f"f1_micro={metrics.get('f1_micro', 0.0) * 100:.2f}%",
        f"f1_macro={metrics.get('f1_macro', 0.0) * 100:.2f}%",
    ]
    if "f1_weighted" in metrics:
        parts.append(f"f1_weighted={metrics['f1_weighted'] * 100:.2f}%")
    return " ".join(parts)


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    generator: Optional[torch.Generator] = None,
    use_reference: bool = False,
) -> DataLoader:
    # Choose appropriate collate function based on dataset type
    collate_func = reference_collate_fn if use_reference or isinstance(dataset, ReferenceInterferogramDataset) else collate_fn

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_func,
    )
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    if generator is not None and shuffle:
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Union[CosineWarmupLR, WarmupReduceLROnPlateau],
    scaler: _GradScaler,
    ema: EMA,
    best_micro: float,
    best_epoch: int,
    out_dir: Path,
    keep_last_k: int,
    is_best: bool,
) -> Path:
    ensure_dir(str(out_dir))
    fname = f"ckpt_epoch{epoch:04d}.pt"
    path = out_dir / fname
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "best_micro": best_micro,
        "best_epoch": best_epoch,
    }
    atomic_save(state, str(path))
    if keep_last_k > 0:
        ckpts = _list_ckpts_sorted(str(out_dir))
        if len(ckpts) > keep_last_k:
            for _, fname_old in ckpts[:-keep_last_k]:
                try:
                    os.remove(out_dir / fname_old)
                except OSError:
                    pass
    if is_best:
        best_path = out_dir / "best.pt"
        atomic_save(state, str(best_path))
        return best_path
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Use new API for TF32 precision settings (PyTorch 2.0+)
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    # Fallback to old API for compatibility
    elif hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = bool(cfg.get("precision", "amp") == "amp" and device_type == "cuda")
    channels_last = bool(cfg.get("channels_last", True))
    torch_compile = bool(cfg.get("torch_compile", True) and hasattr(torch, "compile"))

    data_cfg = cfg.get("data", {})
    root = data_cfg["root"]
    img_glob = data_cfg.get("img_glob", "**/*.png,**/*.jpg,**/*.tif,**/*.bmp")
    image_size = int(data_cfg.get("image_size", 256))

    # Check for reference interferogram configuration
    reference_cfg = cfg.get("reference_interferogram", {})
    use_reference = bool(reference_cfg.get("enabled", False))
    reference_mode = reference_cfg.get("mode", "difference")  # "difference", "dual_input", or "normal"
    reference_path = reference_cfg.get("path", None) if use_reference else None
    reference_preprocessing = reference_cfg.get("preprocessing", {})

    # Korsch configuration
    korsch_mode = bool(data_cfg.get("korsch_mode", False))
    bits_per_number = int(data_cfg.get("bits_per_number", 5))

    # For Korsch mode, we expect 10 parameters (2 mirrors × 5 parameters each)
    if korsch_mode:
        numbers_total = 10  # Fixed for Korsch: 2 mirrors × 5 parameters
    else:
        label_cfg = data_cfg.get("label_parsing", {})
        numbers_total = int(label_cfg.get("numbers_total", data_cfg.get("numbers_total", 11)))

    value_mode = data_cfg.get("value_mode", "mod")
    ignore_last_number = bool(data_cfg.get("ignore_last_number", True))
    filename_label_regex = data_cfg.get("filename_label_regex", None)
    normalization_cfg = data_cfg.get("normalization", {})
    augment_cfg = data_cfg.get("augmentations", data_cfg.get("aug_cfg", {}))
    augment_enabled = bool(augment_cfg.get("enabled", True))

    # Mixup параметры (disabled for reference mode)
    mixup_cfg = augment_cfg.get("mixup", {})
    mixup_enabled = bool(mixup_cfg.get("enabled", False)) and not use_reference  # Disable mixup for reference mode
    mixup_alpha = float(mixup_cfg.get("alpha", 0.2))
    mixup_prob = float(mixup_cfg.get("prob", 0.2))

    # Resolution enhancement configuration
    resolution_enhancement = data_cfg.get("resolution_enhancement", {})
    resolution_enhancement_enabled = bool(resolution_enhancement.get("enabled", True))
    target_resolution = resolution_enhancement.get("target_resolution", None)

    # Create appropriate dataset
    if use_reference and reference_path:
        print(f"[INFO] Using reference interferogram mode: {reference_mode}")
        print(f"[INFO] Reference path: {reference_path}")

        # Handle image_size format for reference dataset
        if isinstance(image_size, int):
            image_size_tuple = (image_size, image_size)
        else:
            image_size_tuple = image_size

        full_ds = ReferenceInterferogramDataset(
            root=root,
            img_glob=img_glob,
            image_size=image_size_tuple,
            korsch_mode=korsch_mode,
            bits_per_number=bits_per_number,
            reference_path=reference_path,
            mode=reference_mode,
            reference_preprocessing=reference_preprocessing,
            resolution_enhancement=resolution_enhancement_enabled,
            target_resolution=target_resolution,
            enable_augmentations=augment_enabled,
        )
        print(f"[INFO] Reference dataset info: {full_ds.get_dataset_info()}")

        ref_info = full_ds.get_reference_info()
        if ref_info:
            print(f"[INFO] Reference interferogram info: {ref_info}")
    else:
        print("[INFO] Using standard dataset (no reference interferogram)")
        label_cfg = data_cfg.get("label_parsing", {})

        full_ds = InterferogramDataset(
            root=root,
            img_glob=img_glob,
            image_size=image_size,
            filename_label_regex=filename_label_regex,
            numbers_total=numbers_total,
            bits_per_number=bits_per_number,
            value_mode=value_mode,
            ignore_last_number=ignore_last_number,
            augment=augment_enabled,
            augmentations=augment_cfg,
            label_cfg=label_cfg,
            normalization=normalization_cfg,
        )
    N = len(full_ds)
    K = full_ds.K
    signature = dataset_signature(full_ds.paths)
    print(f"[INFO] Loaded dataset: {human_int(N)} samples, label dim K={K}, signature={signature}")

    Y = (full_ds.labels > 0.5).astype(np.int64)
    split_cfg = cfg.get("split", {})
    train_idx, val_idx, test_idx, split_cache_path = split_with_cache(Y, split_cfg, seed, signature)
    print(f"[INFO] Train size={human_int(train_idx.size)}, Val size={human_int(val_idx.size)}, Test size={human_int(test_idx.size)}")
    if split_cache_path:
        print(f"[INFO] Split cache: {split_cache_path}")

    stats_path_val = normalization_cfg.get("stats_path")
    stats_path = Path(stats_path_val).expanduser() if stats_path_val else None
    if stats_path and not stats_path.is_absolute():
        stats_path = (Path.cwd() / stats_path).resolve()
    recompute_stats = bool(normalization_cfg.get("recompute", False))
    if full_ds.norm_type == "global_zscore":
        mean = full_ds.global_mean
        std = full_ds.global_std
        loaded_signature = None
        if stats_path and stats_path.exists() and not recompute_stats:
            array = np.load(stats_path)
            mean = float(array["mean"])
            std = float(array["std"])
            if "signature" in array:
                sig_val = array["signature"]
                loaded_signature = sig_val.item() if hasattr(sig_val, "item") else str(sig_val)
        if mean is None or std is None or recompute_stats:
            mean, std = full_ds.compute_global_stats(train_idx)
            if stats_path:
                ensure_dir(str(stats_path.parent))
                np.savez(stats_path, mean=mean, std=std, signature=np.array(signature))
                print(f"[INFO] Global normalization stats saved to {stats_path}.")
        if loaded_signature and loaded_signature != signature:
            print(f"[WARN] Stats signature {loaded_signature} differs from dataset signature {signature}.")
        full_ds.set_global_stats(mean, std)
        print(f"[INFO] Global normalization mean={mean:.5f}, std={std:.5f}.")

    train_ds = full_ds.make_subset(train_idx, augment=augment_enabled, share_stats=True)
    val_ds = full_ds.make_subset(val_idx, augment=False, share_stats=True)
    test_ds = full_ds.make_subset(test_idx, augment=False, share_stats=True) if test_idx.size > 0 else None

    loader_cfg = data_cfg.get("loader", {})
    num_workers = int(loader_cfg.get("workers", data_cfg.get("workers", 8)))
    pin_memory = bool(loader_cfg.get("pin_memory", data_cfg.get("pin_memory", True)))
    drop_last = bool(loader_cfg.get("drop_last", data_cfg.get("drop_last", True)))
    shuffle = bool(loader_cfg.get("shuffle", data_cfg.get("shuffle", True)))
    persistent_workers = bool(loader_cfg.get("persistent_workers", data_cfg.get("persistent_workers", True)))
    prefetch_factor = loader_cfg.get("prefetch_factor", data_cfg.get("prefetch_factor", 4))
    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None

    # Get training configuration with error handling
    train_cfg = cfg.get("train", {})
    if not train_cfg:
        raise KeyError(
            "Missing 'train' section in config. Please add a 'train' section with at least:\n"
            "train:\n"
            "  batch_size: 8\n"
            "  epochs: 5\n"
            "  lr: 0.001\n"
            "Example: python src/train.py --config configs/debug_config.yaml"
        )

    train_batch_size = int(train_cfg.get("batch_size", 32))
    eval_cfg = cfg.get("eval", {})
    eval_batch_size = int(eval_cfg.get("batch_size", train_batch_size))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_loader = build_dataloader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        generator=generator,
        use_reference=use_reference,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_reference=use_reference,
    )
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = build_dataloader(
            test_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            use_reference=use_reference,
        )

    model_cfg = cfg.get("model", {})
    model = InterferoNetMultiLabel(
        out_dim=K,
        base_channels=int(model_cfg.get("base_channels", 32)),
        width_multipliers=model_cfg.get("width_multipliers", [2, 4, 8, 12]),
        block_repeats=model_cfg.get("block_repeats", [1, 1, 1, 1]),
        norm_type=model_cfg.get("norm", "groupnorm"),
        gn_groups=int(model_cfg.get("gn_groups", 16)),
        stochastic_depth=float(model_cfg.get("stochastic_depth", 0.0)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        hidden_dims=model_cfg.get("hidden_dims", None),
        head_dropout=float(model_cfg.get("head_dropout", 0.3)),
        head_norm=model_cfg.get("head_norm", "layernorm"),
        # Korsch and reference support
        bits_per_parameter=bits_per_number,
        use_hybrid_head=bool(model_cfg.get("use_hybrid_head", korsch_mode)),
        regression_dim=int(model_cfg.get("regression_dim", 10)) if korsch_mode else None,
        use_reference_input=use_reference,
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    if torch_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] torch.compile failed, continuing without compilation: {exc}")

    # Use train_cfg already defined above with error handling
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
    )
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "cosine_warmup")
    if scheduler_name == "warmup_plateau":
        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            config=WarmupReduceLROnPlateauConfig(
                warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 3)),
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 2)),
                cooldown=int(scheduler_cfg.get("cooldown", 1)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
                threshold=float(scheduler_cfg.get("threshold", 1e-4)),
                threshold_mode=scheduler_cfg.get("threshold_mode", "rel"),
                mode=scheduler_cfg.get("mode", "max"),
                metric=scheduler_cfg.get("metric", "f1_micro"),
            ),
            base_lrs=[train_cfg.get("lr", 0.001)],
        )
    else:
        scheduler = CosineWarmupLR(
            optimizer,
            T_max=int(train_cfg.get("epochs", 1)),
            warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 5)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
    scaler = make_scaler(device_type, use_amp)
    ema = EMA(model, decay=float(train_cfg.get("ema_decay", 0.999)))

    pos_weight_tensor = None
    bce_pos_value = train_cfg.get("bce_pos_weight", "auto")
    if isinstance(bce_pos_value, str):
        key = bce_pos_value.lower()
        if key == "auto":
            ytr = full_ds.labels[train_idx]
            pos = ytr.sum(axis=0) + 1e-6
            neg = ytr.shape[0] - ytr.sum(axis=0) + 1e-6
            ratios = (neg / pos).astype(np.float32)
            pos_weight_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)
            print("[INFO] Using auto BCE pos_weight (neg/pos).")
        elif key in {"none", "off", "disable"}:
            pos_weight_tensor = None
        else:
            raise ValueError(f"Unsupported bce_pos_weight value: {bce_pos_value}")
    elif bce_pos_value is not None:
        pos_weight_tensor = torch.tensor(bce_pos_value, dtype=torch.float32, device=device)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    loss_reduction = train_cfg.get("loss_reduction", "mean")
    if label_smoothing > 0:
        criterion = BCEWithLogitsLossWithSmoothing(
            smoothing=label_smoothing,
            reduction=loss_reduction,
        )
        print(f"[INFO] Using BCE with label smoothing ε={label_smoothing}")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction=loss_reduction)

    logging_cfg = cfg.get("logging", {})
    workspace_root = Path(cfg.get("workspace_root", "/home/jupyter/work")).expanduser()
    base_out_dir = Path(logging_cfg.get("out_dir", "runs")).expanduser()
    if not base_out_dir.is_absolute():
        base_out_dir = workspace_root / base_out_dir
    base_out_dir = base_out_dir.resolve()
    project_name = logging_cfg.get("project_name", "experiment")
    run_name = logging_cfg.get("run_name") or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = base_out_dir / project_name / run_name
    ensure_dir(str(out_dir))
    print(f"[INFO] Outputs directory: {out_dir}")

    if logging_cfg.get("dump_config", False):
        dump_config_yaml(cfg, out_dir / "config.yaml")
    if logging_cfg.get("env_report", False):
        dump_json(collect_environment_info(), out_dir / "environment.json")

    writer = setup_tensorboard(train_cfg.get("tensorboard", {}), out_dir, run_name)
    if writer:
        print(f"[INFO] TensorBoard logging to {writer.log_dir}")

    log_every = int(train_cfg.get("log_every_n_steps", 50))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    save_every = int(logging_cfg.get("save_every", 1))
    keep_last_k = int(logging_cfg.get("keep_last_k", 5))
    threshold = float(eval_cfg.get("threshold", 0.5))

    metrics_dir = Path(eval_cfg.get("metrics_output", out_dir / "metrics"))
    if not metrics_dir.is_absolute():
        metrics_dir = (out_dir / metrics_dir).resolve()
    ensure_dir(str(metrics_dir))

    history: list[Dict[str, Any]] = []
    global_step = 0
    epochs = int(train_cfg.get("epochs", 1))
    current_epoch = int(train_cfg.get("current_epoch", 0))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    best_micro = 0.0
    best_epoch = 0
    best_checkpoint_path: Optional[Path] = None
    patience_counter = 0
    start_epoch = current_epoch

    if args.resume:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        best_micro = float(ckpt.get("best_micro", best_micro))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        start_epoch = int(ckpt.get("epoch", 0))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"[INFO] Resumed from {resume_path}. Starting epoch={start_epoch + 1}.")

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        num_batches = 0
        skipped_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=120)
        for step, batch_data in enumerate(pbar):
            # Handle different batch formats
            if use_reference and len(batch_data) == 4:  # (x, y, dual_input, mode_info)
                x, y, dual_input, mode_info = batch_data
                y = _ensure_batch_labels(y, x.size(0), mode_info)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if dual_input is not None:
                    dual_input = dual_input.to(device, non_blocking=True)

                if channels_last:
                    # Применяем channels_last только к 4D тензорам [N, C, H, W]
                    if x.dim() == 4:
                        x = x.contiguous(memory_format=torch.channels_last)
                    else:
                        print(f"[WARN] Unexpected x tensor shape: {x.shape}, expected 4D for channels_last")

                    if dual_input is not None:
                        if dual_input.dim() == 4:
                            dual_input = dual_input.contiguous(memory_format=torch.channels_last)
                        else:
                            print(f"[WARN] Unexpected dual_input tensor shape: {dual_input.shape}, expected 4D for channels_last")
                            print(f"[DEBUG] dual_input dtype: {dual_input.dtype}, device: {dual_input.device}")
                            # Попробуем исправить размерность если возможно
                            if dual_input.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                                dual_input = dual_input.unsqueeze(0)
                                print(f"[DEBUG] Added batch dimension to dual_input: {dual_input.shape}")
                                if dual_input.dim() == 4:
                                    dual_input = dual_input.contiguous(memory_format=torch.channels_last)
                                    print(f"[DEBUG] Successfully applied channels_last to fixed dual_input")

                # No mixup for reference mode
                with autocast_ctx(device_type, use_amp):
                    # Model can handle dual input or single input
                    if dual_input is not None and hasattr(model, 'forward') and 'dual_input' in model.forward.__code__.co_varnames:
                        logits = model(x, dual_input)
                    elif dual_input is not None:
                        # Concatenate dual input if model expects single input
                        combined = torch.cat([x, dual_input], dim=1) if dual_input.shape[1] == 1 else dual_input
                        logits = model(combined)
                    else:
                        logits = model(x)

                    loss = criterion(logits, y) / grad_accum
            else:
                # Standard training
                x, y, _, rels = batch_data
                y = _ensure_batch_labels(y, x.size(0), rels)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)

                # Применяем Mixup с вероятностью mixup_prob
                use_mixup = mixup_enabled and np.random.rand() < mixup_prob
                if use_mixup:
                    x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)

                with autocast_ctx(device_type, use_amp):
                    logits = model(x)
                    if use_mixup:
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / grad_accum
                    else:
                        loss = criterion(logits, y) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if not math.isfinite(float(grad_norm)):
                        print(f"[WARN] Non-finite gradient norm {grad_norm:.4f}. Skipping optimizer step.")
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            epoch_loss += float(loss.item()) * grad_accum
            num_batches += 1
            global_step += 1
            if writer and log_every > 0 and global_step % log_every == 0:
                writer.add_scalar("train/loss_iter", epoch_loss / num_batches, global_step)
            pbar.set_postfix({"loss": f"{epoch_loss / max(1, num_batches):.4f}"})
        
        train_loss = epoch_loss / max(1, num_batches)
        if skipped_steps > 0:
            print(f"[INFO] Skipped {skipped_steps} optimizer steps due to non-finite gradients.")
        
        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)
            if skipped_steps > 0:
                writer.add_scalar("train/skipped_steps", skipped_steps, epoch + 1)

        with ema.average_parameters(model):
            val_metrics = evaluate(model, val_loader, criterion, device, device_type, use_amp, threshold, channels_last, use_reference=use_reference)

        print(f"[INFO] Epoch {epoch + 1}: train_loss={train_loss:.4f} | Val {_format_metrics(val_metrics)}")
        if writer and val_metrics:
            writer.add_scalar("val/loss", val_metrics["loss"], epoch + 1)
            writer.add_scalar("val/f1_micro", val_metrics.get("f1_micro", 0.0), epoch + 1)
            writer.add_scalar("val/f1_macro", val_metrics.get("f1_macro", 0.0), epoch + 1)
            if "f1_weighted" in val_metrics:
                writer.add_scalar("val/f1_weighted", val_metrics["f1_weighted"], epoch + 1)

        if isinstance(scheduler, WarmupReduceLROnPlateau):
            metric_value = val_metrics.get(scheduler.cfg.metric, val_metrics.get("f1_micro", 0.0))
            scheduler.step(metric_value)
        else:
            scheduler.step()

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - epoch_start,
            "global_step": global_step,
        }
        history.append(history_entry)

        current_micro = val_metrics.get("f1_micro", 0.0)
        improved = current_micro > best_micro
        if improved:
            best_micro = current_micro
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        is_best = improved
        if is_best or ((epoch + 1) % save_every == 0):
            ckpt_path = save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                best_micro=best_micro,
                best_epoch=best_epoch,
                out_dir=out_dir,
                keep_last_k=keep_last_k,
                is_best=is_best,
            )
            if is_best:
                best_checkpoint_path = ckpt_path

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch + 1}.")
            break

    if writer:
        writer.flush()
        writer.close()

    history_path = metrics_dir / f"history_{run_name}.json"
    save_metrics_history(history, history_path)

    best_path = best_checkpoint_path or (out_dir / "best.pt")
    final_metrics: Dict[str, Any] = {}
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            ema.copy_to(model)
        with torch.no_grad():
            final_val = evaluate(model, val_loader, criterion, device, device_type, use_amp, threshold, channels_last, use_reference=use_reference)
            final_metrics["val_best"] = final_val
            print(f"[RESULT] Best validation metrics: {_format_metrics(final_val)}")
            if test_loader is not None:
                final_test = evaluate(model, test_loader, criterion, device, device_type, use_amp, threshold, channels_last, use_reference=use_reference)
                final_metrics["test"] = final_test
                print(f"[RESULT] Test metrics: {_format_metrics(final_test)}")

    if final_metrics:
        dump_json(final_metrics, metrics_dir / f"final_{run_name}.json")

    summary = {
        "train_size": int(train_idx.size),
        "val_size": int(val_idx.size),
        "test_size": int(test_idx.size),
        "best_f1_micro": best_micro,
        "best_epoch": best_epoch,
        "out_dir": str(out_dir),
        "split_cache": split_cache_path,
        "duration_sec": time.time() - start_time,
        "history_path": str(history_path),
    }
    if final_metrics:
        summary["final_metrics"] = final_metrics
    dump_json(summary, metrics_dir / f"summary_{run_name}.json")

    print("Done.")


if __name__ == "__main__":
    main()
