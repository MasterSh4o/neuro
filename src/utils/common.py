from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

try:
    import importlib.metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


def set_seed(seed: int = 42, *, deterministic: bool | None = None) -> None:
    """
    Устанавливает начальные значения генераторов случайных чисел для воспроизводимости.
    deterministic:
        - True  → принудительно детерминированные алгоритмы (могут снижать производительность).
        - False → максимально возможная производительность (CuDNN benchmark включен).
        - None  → управляется переменной окружения TORCH_DETERMINISTIC (0/1).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic is None:
        deterministic = bool(int(os.environ.get("TORCH_DETERMINISTIC", "0")))

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(data: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, allow_unicode=True, sort_keys=False)


def human_int(n: int) -> str:
    s = str(n)[::-1]
    return " ".join(s[i : i + 3] for i in range(0, len(s), 3))[::-1]


def resolve_path(path: str | os.PathLike[str], *, base: str | os.PathLike[str] | None = None) -> Path:
    p = Path(path)
    if not p.is_absolute():
        base_dir = Path(base) if base is not None else Path.cwd()
        p = (base_dir / p).resolve()
    return p


def collect_dependency_versions(packages: Sequence[str] | None = None) -> Dict[str, str]:
    if packages is None:
        packages = (
            "torch",
            "torchvision",
            "torchaudio",
            "numpy",
            "pandas",
            "opencv-python",
            "scikit-learn",
            "tensorboard",
            "pyyaml",
            "tqdm",
        )
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def export_dependency_report(
    path: str | os.PathLike[str],
    *,
    packages: Sequence[str] | None = None,
    include_pip_freeze: bool = True,
) -> None:
    """
    Сохраняет информацию о версиях зависимостей в текстовом файле.
    """
    report_lines: list[str] = []
    versions = collect_dependency_versions(packages)
    if versions:
        width = max(len(k) for k in versions.keys())
        report_lines.append("# Dependency versions")
        for name in sorted(versions):
            report_lines.append(f"{name:<{width}} : {versions[name]}")
        report_lines.append("")

    if include_pip_freeze:
        try:
            freeze = subprocess.check_output(["pip", "freeze"], text=True, stderr=subprocess.STDOUT, timeout=30)
            report_lines.append("# pip freeze")
            report_lines.extend(freeze.strip().splitlines())
        except Exception as err:  # pragma: no cover
            report_lines.append(f"# pip freeze unavailable: {err}")

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


__all__ = [
    "set_seed",
    "load_config",
    "save_config",
    "human_int",
    "resolve_path",
    "collect_dependency_versions",
    "export_dependency_report",
    "ensure_dir",
    "to_numpy",
]
