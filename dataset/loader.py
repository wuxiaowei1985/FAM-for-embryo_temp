import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.collate_fn import embryo_collate_fn
from dataset.utils.split import split_embryos
from dataset.utils.transforms import FocusTransform, FocusValTransform
from dataset.annotation_reader import AnnotationLoader
from configs import config as cfg

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(cfg.SEED)

train_embryos, val_embryos, test_embryos = split_embryos(cfg.DATA_ROOT, seed=cfg.SEED)

train_dataset = EmbryoDataset(root=cfg.DATA_ROOT,
                              transform=FocusTransform(),
                              embryo_list=train_embryos
                              )
val_dataset = EmbryoDataset(root=cfg.DATA_ROOT,
                            transform=FocusValTransform(),
                            embryo_list=val_embryos
                            )
test_dataset = EmbryoDataset(root=cfg.DATA_ROOT,
                             transform=FocusValTransform(),
                             embryo_list=test_embryos
                             )

# ========== 新增：过滤训练集 ==========
if cfg.ENABLE_CLASS_FILTER:
    df_acc = pd.read_csv(cfg.SAVE_RESULT_DIR)
    # 提取目标类别名称（注意列名 "Accuracy(%)"）
    low_acc_stages = df_acc[df_acc['Accuracy(%)'] < cfg.ACCURACY]['Stage'].tolist()
    # 获取这些类别对应的标签索引
    stage_to_label = AnnotationLoader.STAGE_TO_LABEL
    allowed_labels = [stage_to_label[stage] for stage in low_acc_stages if stage in stage_to_label]
    if allowed_labels:
        original_len = len(train_dataset.samples)
        # 过滤样本
        train_dataset.samples = [s for s in train_dataset.samples if s['label'] in allowed_labels]
        print(f"Filtered dataset: {original_len} -> {len(train_dataset.samples)} samples, "
              f"keeping classes: {low_acc_stages} (labels {allowed_labels})")
    else:
        print(f"Warning: No classes with accuracy < {cfg.ACCURACY}% found.")
else:
    print("Class filter disabled.")

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