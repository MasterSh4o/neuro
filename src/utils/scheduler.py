import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class CosineWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs=5, min_lr=1e-6, last_epoch=-1):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            w = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * w for base_lr in self.base_lrs]
        t = self.last_epoch - self.warmup_epochs
        T = max(1, self.T_max - self.warmup_epochs)
        c = 0.5 * (1 + math.cos(math.pi * t / T))
        return [self.min_lr + (base_lr - self.min_lr) * c for base_lr in self.base_lrs]


@dataclass
class WarmupReduceLROnPlateauConfig:
    warmup_epochs: int = 5
    factor: float = 0.5
    patience: int = 2
    cooldown: int = 0
    min_lr: float = 1e-6
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    mode: str = "max"
    metric: str = "loss"


class WarmupReduceLROnPlateau:
    """Комбинированный планировщик: линейный warmup -> ReduceLROnPlateau."""

    def __init__(
        self,
        optimizer,
        *,
        config: WarmupReduceLROnPlateauConfig,
        base_lrs=None,
    ) -> None:
        self.optimizer = optimizer
        self.cfg = config
        self.warmup_epochs = max(int(config.warmup_epochs), 0)
        self.epoch = 0
        self._finished_warmup = self.warmup_epochs == 0
        self.base_lrs = base_lrs or [group["lr"] for group in optimizer.param_groups]
        self.min_lr = float(config.min_lr)
        self.metric_name = config.metric
        self._last_lr = list(self.base_lrs)

        # Инициализируемwarmup: начинаем с min_lr (или 0, если min_lr не задан).
        if not self._finished_warmup:
            start_lr = self.min_lr if self.min_lr > 0 else min(lr for lr in self.base_lrs) * 1e-3
            for group in self.optimizer.param_groups:
                group["lr"] = start_lr
            self._last_lr = [start_lr for _ in self.base_lrs]

        self.plateau = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            verbose=False,
        )

    @property
    def requires_metric(self) -> bool:
        return True

    def _warmup_step(self) -> None:
        ratio = (self.epoch + 1) / max(1, self.warmup_epochs)
        new_lrs = [
            self.min_lr + (base_lr - self.min_lr) * ratio for base_lr in self.base_lrs
        ]
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group["lr"] = lr
        self._last_lr = new_lrs
        self.epoch += 1
        if self.epoch >= self.warmup_epochs:
            self._finished_warmup = True

    def step(self, metrics: Optional[float] = None) -> None:
        if not self._finished_warmup:
            self._warmup_step()
            return
        if metrics is None:
            raise ValueError("WarmupReduceLROnPlateau requires validation metric.")
        self.plateau.step(metrics)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.epoch += 1

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "finished_warmup": self._finished_warmup,
            "base_lrs": self.base_lrs,
            "last_lr": self._last_lr,
            "plateau": self.plateau.state_dict(),
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.epoch = state.get("epoch", 0)
        self._finished_warmup = state.get("finished_warmup", False)
        self.base_lrs = state.get("base_lrs", self.base_lrs)
        self._last_lr = state.get("last_lr", self._last_lr)
        cfg_dict = state.get("cfg")
        if cfg_dict:
            self.cfg = WarmupReduceLROnPlateauConfig(**cfg_dict)
        self.plateau.load_state_dict(state["plateau"])
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr
