import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs=5, min_lr=1e-6, last_epoch=-1):
        self.T_max = T_max; self.warmup_epochs = warmup_epochs; self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            w = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * w for base_lr in self.base_lrs]
        t = self.last_epoch - self.warmup_epochs
        T = max(1, self.T_max - self.warmup_epochs)
        c = 0.5 * (1 + math.cos(math.pi * t / T))
        return [self.min_lr + (base_lr - self.min_lr) * c for base_lr in self.base_lrs]
