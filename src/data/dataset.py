import os
import glob
from copy import deepcopy
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_ints_no_regex(base: str) -> list[int]:
    """
    Извлекает целые числа из строки без использования регулярных выражений.
    Любой нецифровой символ рассматривается как разделитель.
    """
    nums: list[int] = []
    cur: int | None = None
    for ch in base:
        if "0" <= ch <= "9":
            d = ord(ch) - 48
            cur = d if cur is None else cur * 10 + d
        else:
            if cur is not None:
                nums.append(cur)
                cur = None
    if cur is not None:
        nums.append(cur)
    return nums


def parse_bits_from_basename(
    base: str,
    numbers_total: int = 11,
    bits_per_number: int = 5,
    value_mode: str = "mod",
    ignore_last_number: bool = True,
) -> np.ndarray:
    ints = extract_ints_no_regex(base)
    if len(ints) < numbers_total:
        raise ValueError(
            f"'{base}': found {len(ints)} numbers, expected >= {numbers_total}. Parsed={ints}"
        )
    picked = ints[:numbers_total]
    if ignore_last_number and picked:
        picked = picked[:-1]

    maxv = (1 << bits_per_number) - 1
    match value_mode:
        case "mod":
            picked = [v & maxv for v in picked]
        case "clip":
            picked = [max(0, min(maxv, v)) for v in picked]
        case "raise":
            for v in picked:
                if not (0 <= v <= maxv):
                    raise ValueError(f"value {v} out of range [0,{maxv}] in '{base}'")
        case _:
            picked = [v & maxv for v in picked]

    out = np.zeros(len(picked) * bits_per_number, dtype=np.float32)
    for i, value in enumerate(picked):
        for k in range(bits_per_number):
            out[i * bits_per_number + k] = (value >> k) & 1
    return out


DEFAULT_AUG_CFG = {
    "enabled": True,
    "hflip": {"enabled": True, "prob": 0.5},
    "vflip": {"enabled": True, "prob": 0.5},
    "rotate90": {"enabled": True, "prob": 0.5},
    "gaussian_noise": {"enabled": False, "std": 0.01, "prob": 0.2},
    "random_gamma": {"enabled": False, "range": [0.9, 1.1], "prob": 0.3},
}


def _merge_nested(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            dst[key] = _merge_nested(dst[key], value)
        else:
            dst[key] = value
    return dst


def _normalize_aug_cfg(augmentations: dict | None, legacy_cfg: dict | None) -> dict:
    cfg = deepcopy(DEFAULT_AUG_CFG)
    if legacy_cfg:
        legacy_norm = {
            "enabled": True,
            "hflip": {"enabled": bool(legacy_cfg.get("hflip", True)), "prob": 0.5},
            "vflip": {"enabled": bool(legacy_cfg.get("vflip", True)), "prob": 0.5},
            "rotate90": {"enabled": bool(legacy_cfg.get("rotate90", True)), "prob": 0.5},
            "gaussian_noise": {
                "enabled": bool(legacy_cfg.get("gaussian_noise_std", 0) > 0),
                "std": float(legacy_cfg.get("gaussian_noise_std", 0.01)),
                "prob": 0.2,
            },
            "random_gamma": {
                "enabled": "random_gamma" in legacy_cfg,
                "range": list(legacy_cfg.get("random_gamma", [0.9, 1.1])),
                "prob": 0.3,
            },
        }
        cfg = _merge_nested(cfg, legacy_norm)
    if augmentations:
        cfg = _merge_nested(cfg, augmentations)
    return cfg


class InterferogramDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_glob: str = "**/*.png,**/*.jpg,**/*.tif,**/*.bmp",
        image_size: int = 256,
        *,
        filename_label_regex: str | None = None,  # для обратной совместимости
        numbers_total: int = 11,
        bits_per_number: int = 5,
        value_mode: str = "mod",
        ignore_last_number: bool = True,
        augment: bool = True,
        aug_cfg: dict | None = None,
        label_cfg: dict | None = None,
        normalization: dict | None = None,
        augmentations: dict | None = None,
        files: Sequence[str] | None = None,
        labels: np.ndarray | None = None,
    ):
        self.root = os.path.abspath(root)
        self.image_size = int(image_size)
        self.img_glob = img_glob

        if label_cfg:
            numbers_total = label_cfg.get("numbers_total", numbers_total)
            bits_per_number = label_cfg.get("bits_per_number", bits_per_number)
            value_mode = label_cfg.get("value_mode", value_mode)
            ignore_last_number = label_cfg.get("ignore_last_number", ignore_last_number)

        self.numbers_total = int(numbers_total)
        self.bits_per_number = int(bits_per_number)
        self.value_mode = value_mode
        self.ignore_last_number = bool(ignore_last_number)

        self.normalization_cfg = deepcopy(normalization) if normalization else {}
        self.norm_type = str(self.normalization_cfg.get("type", "none")).lower()
        clip_range = self.normalization_cfg.get("clip_range", None)
        self.clip_range = tuple(clip_range) if clip_range is not None else None
        self.global_mean = (
            float(self.normalization_cfg["mean"])
            if "mean" in self.normalization_cfg
            else None
        )
        self.global_std = (
            float(self.normalization_cfg["std"])
            if "std" in self.normalization_cfg
            else None
        )

        aug_cfg = augmentations if augmentations is not None else aug_cfg
        self.aug_cfg = _normalize_aug_cfg(aug_cfg, aug_cfg if augmentations is None else None)
        self.augment_flag = bool(augment) and bool(self.aug_cfg.get("enabled", True))

        if files is not None and labels is not None:
            abs_files = [os.path.abspath(os.path.join(self.root, f)) if not os.path.isabs(f) else os.path.abspath(f) for f in files]
            self.files = abs_files
            self.paths = [os.path.relpath(f, self.root) for f in self.files]
            self.labels = np.asarray(labels, dtype=np.float32)
        else:
            patterns = [
                p.strip() for p in str(img_glob).replace(";", ",").split(",") if p.strip()
            ]
            discovered: list[str] = []
            for pat in patterns:
                discovered.extend(
                    glob.glob(os.path.join(self.root, pat), recursive=True)
                )
            discovered = sorted({os.path.abspath(f) for f in discovered})
            if not discovered:
                raise RuntimeError(
                    f"No images found in '{self.root}' with patterns: {patterns}"
                )
            self.files = discovered
            self.paths = [os.path.relpath(f, self.root) for f in self.files]

            labels_list: list[np.ndarray] = []
            for path in self.files:
                base = os.path.splitext(os.path.basename(path))[0]
                labels_list.append(
                    parse_bits_from_basename(
                        base,
                        numbers_total=self.numbers_total,
                        bits_per_number=self.bits_per_number,
                        value_mode=self.value_mode,
                        ignore_last_number=self.ignore_last_number,
                    )
                )
            self.labels = np.stack(labels_list, axis=0).astype(np.float32)

        self.K = int(self.labels.shape[1])

    def __len__(self) -> int:
        return len(self.files)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(
                img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
        img = img.astype(np.float32)
        if img.max() > 1.5 or img.min() < -0.5:
            img = img / 255.0
        return img

    def _per_image_zscore(self, img: np.ndarray) -> np.ndarray:
        mean = float(img.mean())
        std = float(img.std())
        return (img - mean) / (std + 1e-6)

    def _should_apply(self, key: str, default_prob: float) -> tuple[bool, float]:
        cfg = self.aug_cfg.get(key, {})
        if isinstance(cfg, bool):
            return cfg, default_prob
        enabled = bool(cfg.get("enabled", True))
        prob = float(cfg.get("prob", default_prob))
        return enabled, prob

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if not (self.augment_flag and self.aug_cfg.get("enabled", True)):
            return img

        out = img
        rng = np.random.rand
        enabled, prob = self._should_apply("hflip", 0.5)
        if enabled and rng() < prob:
            out = np.flip(out, axis=1)

        enabled, prob = self._should_apply("vflip", 0.5)
        if enabled and rng() < prob:
            out = np.flip(out, axis=0)

        enabled, prob = self._should_apply("rotate90", 0.5)
        if enabled and rng() < prob:
            out = np.rot90(out).copy()

        gn_cfg = self.aug_cfg.get("gaussian_noise", {})
        if gn_cfg.get("enabled", False) and rng() < float(gn_cfg.get("prob", 0.2)):
            std = float(gn_cfg.get("std", 0.01))
            out = out + np.random.normal(0.0, std, size=out.shape).astype(np.float32)

        gamma_cfg = self.aug_cfg.get("random_gamma", {})
        if gamma_cfg.get("enabled", False) and rng() < float(gamma_cfg.get("prob", 0.3)):
            lo, hi = gamma_cfg.get("range", [0.9, 1.1])
            gamma = np.random.uniform(lo, hi)
            out = np.sign(out) * (np.abs(out) ** gamma)

        return out.copy()

    def set_global_stats(self, mean: float, std: float) -> None:
        self.global_mean = float(mean)
        self.global_std = float(std if std > 1e-8 else 1.0)
        self.normalization_cfg["mean"] = self.global_mean
        self.normalization_cfg["std"] = self.global_std

    def compute_global_stats(
        self,
        indices: Iterable[int] | None = None,
        *,
        use_fp64: bool = True,
    ) -> tuple[float, float]:
        idxs = list(indices) if indices is not None else range(len(self.files))
        dtype = np.float64 if use_fp64 else np.float32

        pixel_count = 0
        sum_ = 0.0
        sum_sq = 0.0
        for idx in idxs:
            path = self.files[idx]
            img = self._read_image(path).astype(dtype, copy=False)
            sum_ += float(img.sum())
            sum_sq += float(np.square(img, dtype=dtype).sum())
            pixel_count += int(img.size)

        if pixel_count == 0:
            raise RuntimeError("Cannot compute statistics on empty index set")

        mean = sum_ / pixel_count
        variance = max(sum_sq / pixel_count - mean * mean, 0.0)
        std = float(np.sqrt(variance + 1e-12))
        self.set_global_stats(mean, std)
        return float(mean), float(std)

    def make_subset(
        self,
        indices: Sequence[int],
        *,
        augment: bool | None = None,
        share_stats: bool = True,
    ) -> "InterferogramDataset":
        idx_array = np.asarray(indices, dtype=np.int64)
        if idx_array.size == 0:
            raise ValueError("Subset indices must be non-empty")

        subset_files = [self.files[i] for i in idx_array]
        subset_labels = self.labels[idx_array].copy()

        normalization = deepcopy(self.normalization_cfg)
        if share_stats and self.global_mean is not None:
            normalization["mean"] = self.global_mean
            normalization["std"] = self.global_std

        subset = InterferogramDataset(
            root=self.root,
            img_glob=self.img_glob,
            image_size=self.image_size,
            numbers_total=self.numbers_total,
            bits_per_number=self.bits_per_number,
            value_mode=self.value_mode,
            ignore_last_number=self.ignore_last_number,
            augment=self.augment_flag if augment is None else bool(augment),
            augmentations=deepcopy(self.aug_cfg),
            normalization=normalization,
            files=subset_files,
            labels=subset_labels,
        )
        return subset

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        full_path = os.path.join(self.root, rel_path)
        img = self._read_image(full_path)

        if self.norm_type == "per_image_zscore":
            img = self._per_image_zscore(img)

        img = self._augment(img)

        if self.norm_type == "global_zscore" and self.global_mean is not None:
            img = (img - self.global_mean) / (self.global_std + 1e-6)

        if self.clip_range is not None:
            lo, hi = self.clip_range
            img = np.clip(img, lo, hi)

        img = np.expand_dims(np.ascontiguousarray(img, dtype=np.float32), axis=0)
        x = torch.from_numpy(img)
        y = torch.from_numpy(self.labels[idx]).float()
        side = None
        return x, y, side, rel_path
