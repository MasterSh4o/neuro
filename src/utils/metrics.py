from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

EPS = 1e-8


@dataclass
class PerClassMetrics:
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    support: torch.Tensor


class MultilabelMetrics:
    """
    Аккумулятор метрик для multilabel-классификации.
    Поддерживает F1 (micro/macro/weighted) и per-class статистики.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.device = device or torch.device("cpu")
        self.num_classes = num_classes
        self._tp: Optional[torch.Tensor] = None
        self._fp: Optional[torch.Tensor] = None
        self._fn: Optional[torch.Tensor] = None
        self._support: Optional[torch.Tensor] = None
        self._samples: int = 0

    def _ensure_buffers(self, num_classes: int) -> None:
        if self._tp is not None and self._tp.numel() == num_classes:
            return
        self.num_classes = num_classes
        zeros = torch.zeros(num_classes, dtype=torch.float64, device=self.device)
        self._tp = zeros.clone()
        self._fp = zeros.clone()
        self._fn = zeros.clone()
        self._support = zeros.clone()
        self._samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError("logits and targets must be 2D tensors (B, K).")
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have identical shapes.")
        num_classes = logits.shape[1]
        self._ensure_buffers(num_classes)

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).to(torch.float64)
        targets_f = targets.to(torch.float64)

        tp = (preds * targets_f).sum(dim=0)
        fp = (preds * (1.0 - targets_f)).sum(dim=0)
        fn = ((1.0 - preds) * targets_f).sum(dim=0)
        support = targets_f.sum(dim=0)

        self._tp += tp
        self._fp += fp
        self._fn += fn
        self._support += support
        self._samples += logits.shape[0]

    def reset(self) -> None:
        if self._tp is None:
            return
        self._tp.zero_()
        self._fp.zero_()
        self._fn.zero_()
        self._support.zero_()
        self._samples = 0

    def _per_class(self) -> PerClassMetrics:
        tp = self._tp
        fp = self._fp
        fn = self._fn
        support = self._support
        if tp is None or fp is None or fn is None or support is None:
            raise RuntimeError("Metric buffers are uninitialized. Call update() first.")
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2.0 * precision * recall / (precision + recall + EPS)
        return PerClassMetrics(precision=precision, recall=recall, f1=f1, support=support)

    def compute(self) -> Dict[str, float]:
        if self._tp is None or self._fp is None or self._fn is None or self._support is None:
            return {"f1_micro": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        per_class = self._per_class()
        tp = self._tp
        fp = self._fp
        fn = self._fn
        support = self._support

        macro_mask = support > 0
        if macro_mask.any():
            f1_macro = per_class.f1[macro_mask].mean().item()
        else:
            f1_macro = 0.0

        support_sum = support.sum()
        if support_sum > 0:
            f1_weighted = (per_class.f1 * support / support_sum).sum().item()
        else:
            f1_weighted = 0.0

        tp_total = tp.sum()
        fp_total = fp.sum()
        fn_total = fn.sum()
        f1_micro = (2.0 * tp_total) / (2.0 * tp_total + fp_total + fn_total + EPS)
        result = {
            "f1_micro": float(f1_micro.item()),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        }
        result["per_class_precision"] = per_class.precision.cpu().tolist()
        result["per_class_recall"] = per_class.recall.cpu().tolist()
        result["per_class_f1"] = per_class.f1.cpu().tolist()
        result["per_class_support"] = per_class.support.cpu().tolist()
        result["samples"] = self._samples
        return result


@torch.no_grad()
def f1_multilabel(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    """
    Совместимость с прежним API. Возвращает (f1_micro, f1_macro).
    """
    meter = MultilabelMetrics(num_classes=logits.shape[1], threshold=threshold, device=logits.device)
    meter.update(logits, targets)
    metrics = meter.compute()
    return metrics["f1_micro"], metrics["f1_macro"]
