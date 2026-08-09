from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.collate_fn import embryo_collate_fn
from dataset.utils.split import split_embryos
from dataset.utils.transforms import FocusTransform, FocusValTransform
from torch.utils.data import DataLoader
from configs import config as cfg
import random
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

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

labels = [sample["label"] for sample in train_dataset.samples]
class_counts = torch.bincount(torch.tensor(labels), minlength=cfg.NUM_CLASSES).float()
# 处理 tHB 为 0 的情况（防止除以零）
class_counts[class_counts == 0] = 1.0
# 计算每个样本的权重：样本越稀有，权重越大
sample_weights = [1.0 / class_counts[label] for label in labels]
# 归一化权重（使总和等于样本数，稳定性好）
sample_weights = torch.tensor(sample_weights, dtype=torch.float)
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset,
                          batch_size=cfg.BATCH_SIZE,
                          sampler=sampler,        # 使用 sampler，shuffle 必须设为 False
                          shuffle=False,
                          num_workers=cfg.NUM_WORKERS,
                          collate_fn=embryo_collate_fn,
                          generator=g,
                          worker_init_fn=seed_worker
                          )
# train_loader = DataLoader(train_dataset,
#                           batch_size=cfg.BATCH_SIZE,
#                           shuffle=True,
#                           num_workers=cfg.NUM_WORKERS,
#                           collate_fn=embryo_collate_fn,
#                           generator=g,
#                           worker_init_fn=seed_worker
#                           )
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