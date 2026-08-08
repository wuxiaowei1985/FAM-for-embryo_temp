from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.collate_fn import embryo_collate_fn
from dataset.utils.split import split_embryos
from dataset.utils.transforms import FocusTransform, FocusValTransform
from torch.utils.data import DataLoader
from configs import config as cfg
import random
import numpy as np
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(cfg.SEED)

train_embryos, val_embryos, test_embryos = split_embryos(cfg.DATA_ROOT_A, seed=cfg.SEED)

train_dataset = EmbryoDataset(root=cfg.DATA_ROOT_A,
                              transform=FocusTransform(),
                              embryo_list=train_embryos
                              )
val_dataset = EmbryoDataset(root=cfg.DATA_ROOT_A,
                            transform=FocusValTransform(),
                            embryo_list=val_embryos
                            )
test_dataset = EmbryoDataset(root=cfg.DATA_ROOT_A,
                             transform=FocusValTransform(),
                             embryo_list=test_embryos
                             )
train_loader = DataLoader(train_dataset,
                          batch_size=cfg.BATCH_SIZE,
                          shuffle=True,
                          num_workers=cfg.NUM_WORKERS,
                          collate_fn=embryo_collate_fn,
                          generator=g,
                          worker_init_fn=seed_worker
                          )
val_loader = DataLoader(val_dataset,
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=False,
                        num_workers=cfg.NUM_WORKERS,
                        collate_fn=embryo_collate_fn,
                        generator=g,
                        worker_init_fn=seed_worker
                        )
test_loader = DataLoader(test_dataset,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=False,
                         num_workers=cfg.NUM_WORKERS,
                         collate_fn=embryo_collate_fn,
                         generator=g,
                         worker_init_fn=seed_worker
                         )