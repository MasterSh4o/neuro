from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
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


class KorschPrecisionMetrics:
    """
    Специализированные метрики для оценки точности определения смещений
    трехзеркального объектива Корша с микронной/угловой точностью.
    """

    def __init__(
        self,
        bits_per_param: int = 5,
        linear_range_um: float = 1000.0,
        angular_range_arcsec: float = 60.0,
        use_hybrid: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            bits_per_param: Количество бит на параметр
            linear_range_um: Диапазон линейных смещений в микронах
            angular_range_arcsec: Диапазон угловых смещений в угловых секундах
            use_hybrid: Используется ли гибридный подход
            device: Устройство для вычислений
        """
        self.device = device or torch.device("cpu")
        self.bits_per_param = bits_per_param
        self.levels = 2 ** bits_per_param
        self.linear_range_um = linear_range_um
        self.angular_range_arcsec = angular_range_arcsec
        self.use_hybrid = use_hybrid

        # Шаг дискретизации
        self.linear_step_um = linear_range_um / self.levels
        self.angular_step_arcsec = angular_range_arcsec / self.levels

        # Аккумуляторы ошибок
        self._linear_errors = []
        self._angular_errors = []
        self._samples = 0
        self._within_1um_count = 0
        self._within_1arcsec_count = 0
        self._within_half_um_count = 0
        self._within_half_arcsec_count = 0

    @torch.no_grad()
    def update(
        self,
        predicted_logits: torch.Tensor,
        target_logits: torch.Tensor,
        predicted_regression: Optional[torch.Tensor] = None,
        target_regression: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Обновление метрик.

        Args:
            predicted_logits: Предсказанные логиты классификации
            target_logits: Целевые логиты классификации
            predicted_regression: Предсказанные регрессионные значения (опционально)
            target_regression: Целевые регрессионные значения (опционально)
        """
        if predicted_logits.shape != target_logits.shape:
            raise ValueError("Predicted and target logits must have same shape")

        if self.use_hybrid and (predicted_regression is None or target_regression is None):
            raise ValueError("Regression values required for hybrid mode")

        # Декодирование предсказаний и целей в физические единицы
        batch_size = predicted_logits.shape[0]

        for i in range(batch_size):
            pred_bits = torch.sigmoid(predicted_logits[i]) > 0.5
            target_bits = torch.sigmoid(target_logits[i]) > 0.5

            pred_displacements = self._decode_bits_to_physical(pred_bits)
            target_displacements = self._decode_bits_to_physical(target_bits)

            # Применение регрессионной коррекции
            if self.use_hybrid and predicted_regression is not None:
                pred_displacements = self._apply_regression_correction(
                    pred_displacements, predicted_regression[i]
                )

            # Вычисление ошибок
            linear_error, angular_error = self._compute_displacement_error(
                pred_displacements, target_displacements
            )

            self._linear_errors.append(linear_error)
            self._angular_errors.append(angular_error)

            # Подсчет точности с различными порогами
            if linear_error < 1.0:
                self._within_1um_count += 1
            if angular_error < 1.0:
                self._within_1arcsec_count += 1
            if linear_error < 0.5:
                self._within_half_um_count += 1
            if angular_error < 0.5:
                self._within_half_arcsec_count += 1

        self._samples += batch_size

    def _decode_bits_to_physical(self, bits: torch.Tensor) -> dict:
        """Декодирование бит в физические единицы."""
        bits_np = bits.cpu().numpy() if bits.is_cuda else bits.numpy()

        displacements = {}
        # 2 зеркала × 5 параметров = 10 групп бит
        for mirror in range(2):
            mirror_data = {}
            for param_idx in range(5):
                base_idx = (mirror * 5 + param_idx) * self.bits_per_param
                param_bits = bits_np[base_idx:base_idx + self.bits_per_param]

                # Преобразование бит в значение
                value = 0
                for j, bit in enumerate(reversed(param_bits)):
                    if bit:
                        value |= (1 << j)

                # Преобразование в физические единицы
                if param_idx < 2:  # Угловые параметры
                    physical_value = (value - self.levels // 2) * self.angular_step_arcsec
                    param_name = f'angular_{["x", "y"][param_idx]}'
                    unit = 'arcsec'
                else:  # Линейные параметры
                    physical_value = (value - self.levels // 2) * self.linear_step_um
                    param_name = f'linear_{["x", "y", "z"][param_idx - 2]}'
                    unit = 'um'

                mirror_data[param_name] = physical_value

            displacements[f'mirror_{mirror + 1}'] = mirror_data

        return displacements

    def _apply_regression_correction(self, displacements: dict, regression_values: torch.Tensor) -> dict:
        """Применение регрессионной коррекции."""
        reg_np = regression_values.cpu().numpy() if regression_values.is_cuda else regression_values.numpy()

        corrected_displacements = {}
        reg_idx = 0

        for mirror_name in sorted(displacements.keys()):
            corrected_mirror = {}
            mirror_data = displacements[mirror_name]

            param_names = ['angular_x', 'angular_y', 'linear_x', 'linear_y', 'linear_z']

            for param_name in param_names:
                base_value = mirror_data[param_name]

                # Масштаб регрессионной коррекции
                if param_name.startswith('angular'):
                    scale = self.angular_step_arcsec / 4.0  # 1/4 шага дискретизации
                else:
                    scale = self.linear_step_um / 4.0

                correction = reg_np[reg_idx] * scale
                corrected_mirror[param_name] = base_value + correction
                reg_idx += 1

            corrected_displacements[mirror_name] = corrected_mirror

        return corrected_displacements

    def _compute_displacement_error(self, pred: dict, target: dict) -> tuple[float, float]:
        """Вычисление общей ошибки смещения."""
        total_linear_error = 0.0
        total_angular_error = 0.0
        linear_count = 0
        angular_count = 0

        for mirror_name in sorted(pred.keys()):
            pred_data = pred[mirror_name]
            target_data = target[mirror_name]

            for param_name, pred_value in pred_data.items():
                target_value = target_data[param_name]
                error = abs(pred_value - target_value)

                if param_name.startswith('angular'):
                    total_angular_error += error
                    angular_count += 1
                else:
                    total_linear_error += error
                    linear_count += 1

        # Средние ошибки
        avg_linear_error = total_linear_error / max(linear_count, 1)
        avg_angular_error = total_angular_error / max(angular_count, 1)

        return avg_linear_error, avg_angular_error

    def compute(self) -> dict:
        """Вычисление итоговых метрик."""
        if self._samples == 0:
            return {
                'mae_linear_um': 0.0,
                'mae_angular_arcsec': 0.0,
                'accuracy_1um': 0.0,
                'accuracy_1arcsec': 0.0,
                'accuracy_half_um': 0.0,
                'accuracy_half_arcsec': 0.0,
                'samples': 0
            }

        linear_errors = torch.tensor(self._linear_errors, device=self.device)
        angular_errors = torch.tensor(self._angular_errors, device=self.device)

        mae_linear = linear_errors.mean().item()
        mae_angular = angular_errors.mean().item()

        stats = {
            'mae_linear_um': mae_linear,
            'mae_angular_arcsec': mae_angular,
            'rmse_linear_um': torch.sqrt((linear_errors ** 2).mean()).item(),
            'rmse_angular_arcsec': torch.sqrt((angular_errors ** 2).mean()).item(),
            'accuracy_1um': 100.0 * self._within_1um_count / self._samples,
            'accuracy_1arcsec': 100.0 * self._within_1arcsec_count / self._samples,
            'accuracy_half_um': 100.0 * self._within_half_um_count / self._samples,
            'accuracy_half_arcsec': 100.0 * self._within_half_arcsec_count / self._samples,
            'samples': self._samples,
            'bits_per_param': self.bits_per_param,
            'linear_step_um': self.linear_step_um,
            'angular_step_arcsec': self.angular_step_arcsec
        }

        return stats

    def reset(self) -> None:
        """Сброс аккумуляторов."""
        self._linear_errors.clear()
        self._angular_errors.clear()
        self._samples = 0
        self._within_1um_count = 0
        self._within_1arcsec_count = 0
        self._within_half_um_count = 0
        self._within_half_arcsec_count = 0

    def get_precision_summary(self) -> dict:
        """Возвращает сводку по достижимой точности."""
        return {
            'target_precision_um': 1.0,
            'target_precision_arcsec': 1.0,
            'current_precision_um': self.linear_step_um,
            'current_precision_arcsec': self.angular_step_arcsec,
            'sub_micron_capable': self.linear_step_um < 1.0,
            'sub_arcsec_capable': self.angular_step_arcsec < 1.0,
            'required_bits_for_1um': max(np.ceil(np.log2(1000.0)), np.ceil(np.log2(60.0))),
            'required_bits_for_submicron': max(np.ceil(np.log2(2000.0)), np.ceil(np.log2(120.0)))
        }


def create_korsch_metrics(
    bits_per_param: int = 6,
    use_hybrid: bool = False,
    device: Optional[torch.device] = None
) -> KorschPrecisionMetrics:
    """
    Создает метрики для оценки точности Кorsch объектива.
    """
    return KorschPrecisionMetrics(
        bits_per_param=bits_per_param,
        linear_range_um=1000.0,
        angular_range_arcsec=60.0,
        use_hybrid=use_hybrid,
        device=device
    )
