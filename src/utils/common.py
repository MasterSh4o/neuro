import os, random, yaml, torch, numpy as np

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def human_int(n: int) -> str:
    s = str(n)[::-1]
    return " ".join(s[i:i+3] for i in range(0, len(s), 3))[::-1]
