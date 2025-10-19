import torch

def f1_multilabel(logits, targets, threshold=0.5):
    """
    logits  : (B, K) raw logits
    targets : (B, K) in {0,1}
    Returns: (f1_micro, f1_macro)
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(targets.dtype)
    tp = (pred * targets).sum(dim=0)
    fp = (pred * (1 - targets)).sum(dim=0)
    fn = ((1 - pred) * targets).sum(dim=0)
    eps = 1e-8
    f1_per_class = 2 * tp / (2 * tp + fp + fn + eps)
    present = (targets.sum(dim=0) > 0)
    f1_macro = f1_per_class[present].mean() if present.any() else torch.tensor(0.0, device=targets.device)
    TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
    f1_micro = 2 * TP / (2 * TP + FP + FN + eps)
    return f1_micro.item(), f1_macro.item()
