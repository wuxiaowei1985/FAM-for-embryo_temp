from pathlib import Path
import random
from configs import config as cfg

def split_embryos(root, train_ratio=0.7, val_ratio=0.15, seed=cfg.SEED):
    root = Path(root)
    embryos = sorted([p.name for p in (root / "embryo_dataset").iterdir() if p.is_dir()])
    random.seed(seed)
    random.shuffle(embryos)
    n = len(embryos)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = embryos[:train_end]
    val = embryos[train_end:val_end]
    test = embryos[val_end:]
    return train, val, test