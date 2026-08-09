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
    max_weight = 10.0
    class_weights = torch.clamp(class_weights, max=max_weight)
    class_weights[class_counts == 0] = 0.0
    class_weights_tensor = class_weights.to(cfg.DEVICE)
    # ==========================================
    # 创建带权重的损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
    # ============ 精细冻结/解冻策略 ============
    for name, param in model.named_parameters():
        param.requires_grad = False  # 默认全部冻结
    # 只解冻 ResNet18 的 layer4 和最后的 BN 层
    for name, param in model.named_parameters():
        if 'encoder' in name:
            if 'layer4' in name or 'bn2' in name or 'bn3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    # FocusAttention 和 Head 全部可训练（它们在之前是 True，现在保持）
    for name, param in model.named_parameters():
        if 'fusion' in name or 'head' in name or 'focus_embedding' in name:
            param.requires_grad = True
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad], 'lr': cfg.BACKBONE_LR,
         'weight_decay': 1e-4},
        {'params': [p for n, p in model.named_parameters() if 'encoder' not in n and p.requires_grad], 'lr': cfg.LR,
         'weight_decay': 1e-4}
    ])
    # 改用 ReduceLROnPlateau：监控验证损失，若连续 5 个 epoch 不降，LR 乘 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控验证损失是否下降
        factor=0.5,  # 每次降低一半
        patience=5,  # 5 个 epoch 不降就触发
        min_lr=cfg.MIN_LR,  # 最低 1e-6
        verbose=True
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
            scheduler.step(val_loss)
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