import torch
import torch.nn as nn
from utils.seed import seed_everything
from configs import config as cfg
seed_everything(cfg.SEED)

from trainer.trainer import Trainer
from trainer.validate import Validator
from dataset.loader import *
from utils.early_stopping import EarlyStopping
from utils.history import History
from utils.plot import plot_training_curve
from utils.get_model import build_model

# ============================================================
# Phase 1：冻结策略
# ============================================================
def freeze_for_phase2(model):
    print("\n" + "=" * 60)
    print("Applying Phase 2 freezing strategy")
    print("=" * 60)
    # --------------------------------------------------------
    # 先全部冻结
    # --------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    # --------------------------------------------------------
    # Phase 2 可训练模块
    # coarse_embedding
    # MSFD Attention
    # 3个 fine heads
    # --------------------------------------------------------
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
# 创建 optimizer
# ============================================================
def create_optimizer(model, lr):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    return optimizer
# ============================================================
# 创建 scheduler
# ============================================================
def create_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=cfg.MIN_LR
    )
    return scheduler
# ============================================================
# Phase 1
# ============================================================
def train_phase1(model):
    print("\n")
    print("=" * 70)
    print("PHASE 1: COARSE STAGE TRAINING")
    print("=" * 70)
    # --------------------------------------------------------
    # Phase 1 所有模块都可以训练
    # --------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.focus_attention.parameters():
        param.requires_grad = True
    for param in model.coarse_head.parameters():
        param.requires_grad = True
    # --------------------------------------------------------
    # optimizer
    # --------------------------------------------------------
    optimizer = create_optimizer(model, cfg.LR)
    scheduler = create_scheduler(optimizer)
    # --------------------------------------------------------
    # loss
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=cfg.DEVICE, stage="coarse")
    validator = Validator(model=model, criterion=criterion, device=cfg.DEVICE, stage="coarse")
    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------
    early_stopping = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA, save_path=cfg.SAVE_MODEL_DIR / "best_model.pth")
    # --------------------------------------------------------
    # History
    # --------------------------------------------------------
    history = History()
    # --------------------------------------------------------
    # training loop
    # --------------------------------------------------------
    for epoch in range(cfg.EPOCHS):
        print("\n" + "=" * 60)
        print(f"Phase 1 | Epoch " f"{epoch + 1}/" f"{cfg.EPOCHS}")
        # ----------------------------------------------------
        # train
        # ----------------------------------------------------
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        # ----------------------------------------------------
        # validation
        # ----------------------------------------------------
        val_loss, val_acc = validator.validate(val_loader)
        # ----------------------------------------------------
        # scheduler
        # ----------------------------------------------------
        scheduler.step(val_loss)
        # ----------------------------------------------------
        # early stopping
        # 你之前已经改成根据 accuracy
        # ----------------------------------------------------
        stop = early_stopping(val_acc=val_acc, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        # ----------------------------------------------------
        # log
        # ----------------------------------------------------
        lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss : " f"{train_loss:.4f}")
        print(f"Train Acc  : " f"{train_acc:.4f}")
        print(f"Val Loss   : " f"{val_loss:.4f}")
        print(f"Val Acc    : " f"{val_acc:.4f}")
        print(f"LR         : " f"{lr:.8f}")
        history.update(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=lr,
        )
        # ----------------------------------------------------
        # early stop
        # ----------------------------------------------------
        if stop:
            print("\n" + "=" * 60)
            print("Phase 1 early stopping triggered.")
            print("=" * 60)
            break
    # --------------------------------------------------------
    # save last
    # --------------------------------------------------------
    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, cfg.SAVE_MODEL_DIR / "last_model.pth")
    print("last model saved")
    history.save(cfg.HISTORY_CSV)
    plot_training_curve(cfg.HISTORY_CSV, cfg.RUN_DIR)
    print("plot and history saved")
    return model
# ============================================================
# Phase 2
# ============================================================
def train_phase2(model):
    print("\n")
    print("=" * 70)
    print("PHASE 2: FINE-GRAINED TRAINING")
    print("=" * 70)
    # --------------------------------------------------------
    # 加载 Phase 1 best
    # --------------------------------------------------------
    checkpoint = torch.load(cfg.MODEL_DIR, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model"])
    print("Phase 1 checkpoint loaded successfully.")
    # --------------------------------------------------------
    # 冻结
    # --------------------------------------------------------
    freeze_for_phase2(model)
    # --------------------------------------------------------
    # optimizer
    # --------------------------------------------------------
    optimizer = create_optimizer(model, cfg.LR)
    scheduler = create_scheduler(optimizer)
    # --------------------------------------------------------
    # loss
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=cfg.DEVICE, stage="fine")
    validator = Validator(model=model, criterion=criterion, device=cfg.DEVICE, stage="fine")
    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------
    early_stopping = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA, save_path=cfg.SAVE_MODEL_DIR)
    # --------------------------------------------------------
    # History
    # --------------------------------------------------------
    history = History()
    # --------------------------------------------------------
    # training
    # --------------------------------------------------------
    for epoch in range(cfg.EPOCHS):
        print("\n" + "=" * 60)
        print(f"Phase 2 | Epoch " f"{epoch + 1}/" f"{cfg.EPOCHS}")
        # ----------------------------------------------------
        # train
        # ----------------------------------------------------
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        # ----------------------------------------------------
        # validation
        # 注意：这里的 val_acc 是最终16分类 accuracy
        # ----------------------------------------------------
        val_loss, val_acc = validator.validate(val_loader)
        # ----------------------------------------------------
        # scheduler
        # ----------------------------------------------------
        scheduler.step(val_loss)
        # ----------------------------------------------------
        # early stopping
        # ----------------------------------------------------
        stop = early_stopping(val_acc=val_acc, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        # ----------------------------------------------------
        # log
        # ----------------------------------------------------
        lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss : " f"{train_loss:.4f}")
        print(f"Train Fine Acc : " f"{train_acc:.4f}")
        print(f"Val Loss   : " f"{val_loss:.4f}")
        print(f"Val 16-class Acc : " f"{val_acc:.4f}")
        print(f"LR         : " f"{lr:.8f}")
        history.update(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=lr
        )
        # ----------------------------------------------------
        # early stopping
        # ----------------------------------------------------
        if stop:
            print("\n" + "=" * 60)
            print("Phase 2 early stopping triggered.")
            print("=" * 60)
            break
    # --------------------------------------------------------
    # save last
    # --------------------------------------------------------
    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, cfg.SAVE_MODEL_DIR)
    print("last model saved")
    history.save(cfg.HISTORY_CSV)
    plot_training_curve(cfg.HISTORY_CSV, cfg.RUN_DIR)
    print("plot and history saved")
    return model
# ============================================================
# Main
# ============================================================
def main():
    # --------------------------------------------------------
    # build model
    # --------------------------------------------------------
    model = build_model(cfg.CURRENT_MODEL)
    model = model.to(cfg.DEVICE)
    # --------------------------------------------------------
    # Phase 1
    # --------------------------------------------------------
    if cfg.TRAIN_STAGE in ["coarse", "both"]:
        model = train_phase1(model)
    # --------------------------------------------------------
    # Phase 2
    # --------------------------------------------------------
    if cfg.TRAIN_STAGE in ["fine", "both"]:
        # 如果只训练 Phase 2
        # 则直接加载 Phase 1
        if cfg.TRAIN_STAGE == "fine":
            model = train_phase2(model)
        else:
            model = train_phase2(model)
    print("\n" + "=" * 70)
    print("Training completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()