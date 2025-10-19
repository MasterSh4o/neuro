import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset

def extract_ints_no_regex(base: str) -> list[int]:
    """
    Extract integers by scanning characters; treats any non-digit as a separator.
    Robust to spaces/underscores and avoids regex pitfalls.
    """
    nums = []
    cur = None
    for ch in base:
        if '0' <= ch <= '9':
            d = ord(ch) - 48
            cur = d if cur is None else cur * 10 + d
        else:
            if cur is not None:
                nums.append(cur)
                cur = None
    if cur is not None:
        nums.append(cur)
    return nums

def parse_bits_from_basename(base, numbers_total=11, bits_per_number=5,
                             value_mode="mod", ignore_last_number=True):
    ints = extract_ints_no_regex(base)
    if len(ints) < numbers_total:
        raise ValueError(f"'{base}': found {len(ints)} numbers, expected >= {numbers_total}. Parsed={ints}")
    picked = ints[:numbers_total]
    if ignore_last_number and len(picked) > 0:
        picked = picked[:-1]
    maxv = (1 << bits_per_number) - 1
    if value_mode == "mod":
        picked = [v & maxv for v in picked]
    elif value_mode == "clip":
        picked = [max(0, min(maxv, v)) for v in picked]
    elif value_mode == "raise":
        for v in picked:
            if not (0 <= v <= maxv):
                raise ValueError(f"value {v} out of range [0,{maxv}] in '{base}'")
    else:
        picked = [v & maxv for v in picked]
    out = np.zeros((len(picked) * bits_per_number,), dtype=np.float32)
    for i, v in enumerate(picked):
        for k in range(bits_per_number):
            out[i * bits_per_number + k] = (v >> k) & 1
    return out

class InterferogramDataset(Dataset):
    def __init__(self, root, img_glob="**/*.png,**/*.jpg,**/*.tif,**/*.bmp", image_size=256,
                 # Backward-compat: config may pass this — we accept and ignore it.
                 filename_label_regex=None,
                 numbers_total=11, bits_per_number=5, value_mode="mod", ignore_last_number=True,
                 augment=True, aug_cfg=None):
        self.root = root
        self.image_size = int(image_size)
        # filename_label_regex is intentionally unused (regex-less parser); kept for compatibility
        self.numbers_total = int(numbers_total)
        self.bits_per_number = int(bits_per_number)
        self.value_mode = value_mode
        self.ignore_last_number = bool(ignore_last_number)
        self.augment_flag = bool(augment)
        self.aug_cfg = aug_cfg or {}

        patterns = [p.strip() for p in str(img_glob).replace(";", ",").split(",") if p.strip()]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(root, pat), recursive=True))
        files = sorted(set(files))
        if not files:
            raise RuntimeError(f"No images found in '{root}' with patterns: {patterns}")
        self.files = files
        self.paths = [os.path.relpath(f, root) for f in files]

        labels = []
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            y = parse_bits_from_basename(base,
                                         numbers_total=self.numbers_total,
                                         bits_per_number=self.bits_per_number,
                                         value_mode=self.value_mode,
                                         ignore_last_number=self.ignore_last_number)
            labels.append(y)
        self.labels = np.stack(labels, axis=0).astype(np.float32)
        self.K = self.labels.shape[1]

    def __len__(self): return len(self.files)

    def _read_image(self, path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None: raise FileNotFoundError(path)
        if im.shape[0] != self.image_size or im.shape[1] != self.image_size:
            im = cv2.resize(im, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32)
        if im.max() > 1.0 or im.min() < -1.0:
            im = (im - im.mean()) / (im.std() + 1e-6)
        im = np.clip(im, -6, 6)
        return im

    def _augment(self, im):
        if not self.augment_flag: return im
        if self.aug_cfg.get("hflip", True) and np.random.rand() < 0.5: im = np.flip(im, axis=1)
        if self.aug_cfg.get("vflip", True) and np.random.rand() < 0.5: im = np.flip(im, axis=0)
        if self.aug_cfg.get("rotate90", True) and np.random.rand() < 0.5: im = np.rot90(im).copy()
        if self.aug_cfg.get("gaussian_noise_std", 0.0) > 0 and np.random.rand() < 0.2:
            std = float(self.aug_cfg["gaussian_noise_std"])
            im = im + np.random.normal(0, std, size=im.shape).astype(np.float32)
        if "random_gamma" in self.aug_cfg and np.random.rand() < 0.3:
            lo, hi = self.aug_cfg.get("random_gamma", [0.9, 1.1])
            g = np.random.uniform(lo, hi)
            im = np.sign(im) * (np.abs(im) ** g)
        return im.copy()

    def __getitem__(self, idx):
        rel = self.paths[idx]
        full = os.path.join(self.root, rel)
        im = self._read_image(full)
        im = self._augment(im)
        import numpy as _np
        im = _np.ascontiguousarray(im)
        im = _np.expand_dims(im, 0)
        im = _np.ascontiguousarray(im)
        x = torch.from_numpy(im)
        y = torch.from_numpy(self.labels[idx]).float()
        side = None
        return x, y, side, rel
