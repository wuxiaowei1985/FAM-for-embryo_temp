import torch
import torch.nn as nn
from trainer.trainer import Trainer
from trainer.validate import Validator
from dataset.loader import *
from configs import config as cfg
from utils.seed import seed_everything
from utils.early_stopping import EarlyStopping
from utils.history import History
from utils.plot import plot_training_curve

seed_everything(cfg.SEED)

def main():
    model = cfg.CURRENT_MODEL
    history = History()
    # ============ 新增：计算类别权重 ============
    all_labels = [sample["label"] for sample in train_dataset.samples]
    labels_tensor = torch.tensor(all_labels)
    class_counts = torch.bincount(labels_tensor, minlength=cfg.NUM_CLASSES).float()
    total_samples = len(train_dataset)
    class_weights = total_samples / (cfg.NUM_CLASSES * class_counts)
    class_weights[class_counts == 0] = 0.0
    # 可选归一化（使权重均值约为1）
    class_weights = class_weights / class_weights.sum() * cfg.NUM_CLASSES
    class_weights_tensor = class_weights.to(cfg.DEVICE)
    # ==========================================
    # 创建带权重的损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.EPOCHS,
        eta_min=cfg.MIN_LR
    )
    early_stopping = EarlyStopping(
        patience=cfg.PATIENCE,
        min_delta=cfg.MIN_DELTA,
        save_path=cfg.SAVE_MODEL_DIR / "best_model.pth",
    )
    trainer = Trainer(
        model,
        criterion=criterion,
        optimizer=optimizer,
        device=cfg.DEVICE
    )
    validator = Validator(
        model,
        criterion=criterion,
        device=cfg.DEVICE
    )
    for epoch in range(cfg.EPOCHS):
        print("=" * 60)
        print(f"Epoch {epoch+1}")
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = validator.validate(val_loader)
        if cfg.EARLY_STOPPING:
            stop = early_stopping(val_acc=val_acc, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            print(f"Train Loss : {train_loss:.4f}")
            print(f"Train Acc  : {train_acc:.4f}")
            print(f"Val Loss   : {val_loss:.4f}")
            print(f"Val Acc    : {val_acc:.4f}")
            print(f"LR         : {scheduler.get_last_lr()[0]:.8f}")
            history.update(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc, lr=scheduler.get_last_lr()[0])
            scheduler.step()
            if stop:
                print("=" * 60)
                print("Early stopping triggered.")
                print("=" * 60)
                break

    checkpoint = {
        "epoch": epoch+1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, cfg.SAVE_MODEL_DIR / "last_model.pth")
    print("last model saved")
    history.save(cfg.HISTORY_CSV)
    plot_training_curve(cfg.HISTORY_CSV, cfg.RUN_DIR)
    print("plot and history saved")


if __name__ == "__main__":
    main()