import os
import socket
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from configs import config as cfg
# ============================================================
# Utility
# ============================================================
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
# ============================================================
# Phase 2 Freeze
# ============================================================
def freeze_for_phase2(model):
    print("\n" + "=" * 60)
    print("Applying Phase 2 freezing strategy")
    print("=" * 60)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.coarse_embedding.parameters():
        param.requires_grad = True
    for param in model.msfd_attention.parameters():
        param.requires_grad = True
    for param in model.pronuclear_head.parameters():
        param.requires_grad = True
    for param in model.cleavage_head.parameters():
        param.requires_grad = True
    for param in model.blastocyst_head.parameters():
        param.requires_grad = True

# ============================================================
# Optimizer
# ============================================================
def create_optimizer(model, lr):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    return optimizer
# ============================================================
# Scheduler
# ============================================================
def create_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
    return scheduler
# ============================================================
# Phase 1
# ============================================================
def train_phase1(model, train_loader, val_loader, train_sampler, rank, device):
    from trainer.trainer import Trainer
    from trainer.validate import Validator
    from utils.early_stopping import EarlyStopping
    from utils.history import History
    from utils.plot import plot_training_curve
    from utils.distributed import unwrap_model, broadcast_bool
    if rank == 0:
        print("\n")
        print("=" * 70)
        print("PHASE 1: COARSE STAGE TRAINING")
        print("=" * 70)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.focus_attention.parameters():
        param.requires_grad = True
    for param in model.coarse_head.parameters():
        param.requires_grad = True
    optimizer = create_optimizer(model, cfg.LR)
    scheduler = create_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, stage="coarse")
    validator = Validator(model=model, criterion=criterion, device=device,stage="coarse")
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOPPING_PATIENCE, min_delta=cfg.MIN_DELTA, save_path=cfg.PHASE1_BEST_MODEL)
    history = History()
    for epoch in range(cfg.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            print("\n" + "=" * 60)
            print(f"Phase 1 | Epoch "f"{epoch + 1}/{cfg.EPOCHS}")
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = validator.validate(val_loader)
        scheduler.step()
        stop = False
        if rank == 0:
            stop = early_stopping(val_acc=val_acc, model=unwrap_model(model), optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            lr = optimizer.param_groups[0]["lr"]
            print(f"Train Loss : "f"{train_loss:.4f}")
            print(f"Train Acc  : "f"{train_acc:.4f}")
            print(f"Val Loss   : "f"{val_loss:.4f}")
            print(f"Val Acc    : "f"{val_acc:.4f}")
            print(f"LR         : "f"{lr:.8f}")
            history.update(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_coarse_acc=train_acc,
                train_fine_acc=0.0,
                train_final_acc=0.0,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=lr
            )
        stop = broadcast_bool(stop, device)
        if stop:
            if rank == 0:
                print("\n" + "=" * 60)
                print("Phase 1 early stopping triggered.")
                print("=" * 60)
            break
    if rank == 0:
        base_model = unwrap_model(model)
        checkpoint = {
            "epoch": epoch + 1,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(checkpoint, cfg.PHASE1_LAST_MODEL)
        history.save(cfg.PHASE1_SAVE / "history.csv")
        plot_training_curve(cfg.PHASE1_SAVE / "history.csv", cfg.PHASE1_SAVE)
        print("Phase 1 last model saved.")
        print("Phase 1 history and plot saved.")
# ============================================================
# Phase 2
# ============================================================
def train_phase2(model, train_loader, val_loader, train_sampler, rank, device):
    from trainer.trainer import Trainer
    from trainer.validate import Validator
    from utils.early_stopping import EarlyStopping
    from utils.history import History
    from utils.plot import plot_training_curve
    from utils.distributed import unwrap_model, broadcast_bool
    if rank == 0:
        print("\n")
        print("=" * 70)
        print("PHASE 2: FINE-GRAINED TRAINING")
        print("=" * 70)
    # ========================================================
    # Load Phase 1 checkpoint
    # ========================================================
    checkpoint = torch.load(cfg.PHASE2_MODEL, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print("Phase 1 checkpoint loaded successfully.")
    # ========================================================
    # Freeze
    # ========================================================
    freeze_for_phase2(model)
    # ========================================================
    # Optimizer
    # ========================================================
    optimizer = create_optimizer(model, cfg.LR)
    scheduler = create_scheduler(optimizer)
    # ========================================================
    # Loss
    # ========================================================
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    # ========================================================
    # Trainer
    # ========================================================
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, stage="fine")
    validator = Validator(model=model, criterion=criterion, device=device, stage="fine")
    # ========================================================
    # Early stopping
    # ========================================================
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOPPING_PATIENCE, min_delta=cfg.MIN_DELTA, save_path=cfg.PHASE2_BEST_MODEL)
    history = History()
    # ========================================================
    # Training
    # ========================================================
    for epoch in range(cfg.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            print("\n" + "=" * 60)
            print(f"Phase 2 | Epoch "f"{epoch + 1}/{cfg.EPOCHS}")
        train_loss, train_coarse_acc, train_fine_acc, train_final_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = validator.validate(val_loader)
        scheduler.step()
        stop = False
        if rank == 0:
            stop = early_stopping(val_acc=val_acc, model=unwrap_model(model), optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            lr = optimizer.param_groups[0]["lr"]
            print(f"Train Loss : "f"{train_loss:.4f}")
            print(f"Train Coarse Acc : "f"{train_coarse_acc:.4f}")
            print(f"Train Conditional Fine Acc : "f"{train_fine_acc:.4f}")
            print(f"Train 16-class Acc : "f"{train_final_acc:.4f}")
            print(f"Val Loss : "f"{val_loss:.4f}")
            print(f"Val 16-class Acc : "f"{val_acc:.4f}")
            print(f"LR : "f"{lr:.8f}")
            history.update(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_coarse_acc=train_coarse_acc,
                train_fine_acc=train_fine_acc,
                train_final_acc=train_final_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=lr
            )
        stop = broadcast_bool(stop, device)
        if stop:
            if rank == 0:
                print("\n" + "=" * 60)
                print("Phase 2 early stopping triggered.")
                print("=" * 60)
            break
    if rank == 0:
        base_model = unwrap_model(model)
        checkpoint = {
            "epoch": epoch + 1,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, cfg.PHASE2_LAST_MODEL)
        history.save(cfg.PHASE2_SAVE / "history.csv")
        plot_training_curve(cfg.PHASE2_SAVE / "history.csv", cfg.PHASE2_SAVE)
        print("Phase 2 last model saved.")
        print("Phase 2 history and plot saved.")
# ============================================================
# Worker
# ============================================================
def worker(rank, world_size, gpu_ids, master_port):
    from utils.seed import seed_everything
    from utils.distributed import setup_distributed, cleanup_distributed, wrap_model, is_main_process
    from utils.get_model import build_model
    from dataset.loader import build_dataloaders
    setup_distributed(rank=rank, world_size=world_size, gpu_ids=gpu_ids, master_port=master_port)
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    try:
        # ====================================================
        # Seed
        # ====================================================
        seed_everything(cfg.SEED + rank)
        # ====================================================
        # Dataloader
        # ====================================================
        train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = build_dataloaders(rank=rank, world_size=world_size)
        # ====================================================
        # Build model
        # ====================================================
        model = build_model(cfg.CURRENT_MODEL)
        model = model.to(device)
        # ====================================================
        # Phase 1
        # ====================================================
        if cfg.TRAIN_STAGE in ["coarse", "both"]:
            model = wrap_model(model, device)
            train_phase1(model=model, train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler, rank=rank, device=device)
            # ------------------------------------------------
            # Unwrap before Phase 2
            # ------------------------------------------------
            model = model.module
        # ====================================================
        # Phase 2
        # ====================================================
        if cfg.TRAIN_STAGE in ["fine", "both"]:
            model = wrap_model(model, device)
            train_phase2(model=model, train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler, rank=rank, device=device)
        if is_main_process():
            print("\n" + "=" * 70)
            print("Training completed.")
            print("=" * 70)
    finally:
        cleanup_distributed()
# ============================================================
# Main
# ============================================================
def main():
    # ========================================================
    # CPU
    # ========================================================
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA.")
    # ========================================================
    # GPU validation
    # ========================================================
    available_gpu_count = (torch.cuda.device_count())
    if cfg.GPU_COUNT < 1:
        raise ValueError(f"GPU_COUNT must be >= 1, got {cfg.GPU_COUNT}.")
    if cfg.GPU_COUNT > available_gpu_count:
        raise ValueError(
            f"GPU_COUNT={cfg.GPU_COUNT}, "f"but only {available_gpu_count} GPUs are available.")
    if len(cfg.GPU_IDS) != cfg.GPU_COUNT:
        raise ValueError(f"GPU_COUNT={cfg.GPU_COUNT}, "f"but GPU_IDS={cfg.GPU_IDS}.")
    if len(set(cfg.GPU_IDS)) != len(cfg.GPU_IDS):
        raise ValueError(f"GPU_IDS contains duplicates: {cfg.GPU_IDS}")
    for gpu_id in cfg.GPU_IDS:
        if gpu_id < 0 or gpu_id >= available_gpu_count:
            raise ValueError(f"Invalid GPU ID {gpu_id}. "f"Available GPU count: "f"{available_gpu_count}.")
    # ========================================================
    # Output directory
    # ========================================================
    cfg.PHASE1_SAVE.mkdir(parents=True, exist_ok=True)
    cfg.PHASE2_SAVE.mkdir(parents=True, exist_ok=True)
    cfg.SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # ========================================================
    # Single GPU
    # ========================================================
    if cfg.GPU_COUNT == 1:
        worker(rank=0, world_size=1, gpu_ids=cfg.GPU_IDS, master_port=cfg.MASTER_PORT)
        return
    # ========================================================
    # Multi GPU
    # ========================================================
    mp.set_start_method("spawn", force=True)
    master_port = find_free_port()
    mp.spawn(worker, args=(cfg.GPU_COUNT, cfg.GPU_IDS, master_port), nprocs=cfg.GPU_COUNT, join=True)

if __name__ == "__main__":
    main()