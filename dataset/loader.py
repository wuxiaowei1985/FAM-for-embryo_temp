import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.collate_fn import embryo_collate_fn
from dataset.utils.split import split_embryos
from dataset.utils.transforms import FocusTransform, FocusValTransform
from configs import config as cfg
# ============================================================
# Worker Seed
# ============================================================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# ============================================================
# Build DataLoaders
# ============================================================
def build_dataloaders(rank=0, world_size=1):
    # ========================================================
    # Embryo-level split
    # ========================================================
    train_embryos, val_embryos, test_embryos = split_embryos(cfg.DATA_ROOT, seed=cfg.SEED)
    # ========================================================
    # Dataset
    # ========================================================
    train_dataset = EmbryoDataset(
        root=cfg.DATA_ROOT,
        transform=FocusTransform(image_size=cfg.IMAGE_SIZE, num_views=cfg.NUM_AUG_VIEWS),
        embryo_list=train_embryos
    )
    val_dataset = EmbryoDataset(
        root=cfg.DATA_ROOT,
        transform=FocusValTransform(image_size=cfg.IMAGE_SIZE),
        embryo_list=val_embryos
    )
    test_dataset = EmbryoDataset(
        root=cfg.DATA_ROOT,
        transform=FocusValTransform(image_size=cfg.IMAGE_SIZE),
        embryo_list=test_embryos
    )
    # ========================================================
    # Distributed Sampler
    # ========================================================
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.SEED, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    # ========================================================
    # Generator
    # ========================================================
    generator = torch.Generator()
    generator.manual_seed(cfg.SEED + rank)
    # ========================================================
    # Train Loader
    # ========================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=embryo_collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.NUM_WORKERS > 0
    )
    # ========================================================
    # Validation Loader
    # ========================================================
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=embryo_collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.NUM_WORKERS > 0
    )
    # ========================================================
    # Test Loader
    # ========================================================
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=embryo_collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.NUM_WORKERS > 0
    )
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler