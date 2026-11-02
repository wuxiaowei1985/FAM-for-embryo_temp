import torch
import torch.nn as nn
from utils.seed import seed_everything
from configs import config as cfg
from trainer.trainer import Trainer
from trainer.validate import Validator
from dataset.loader import train_loader, val_loader
from utils.early_stopping import EarlyStopping
from utils.history import History
from utils.plot import plot_training_curve
from utils.get_model import build_model

def main():
    seed_everything(cfg.SEED)
    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TRAIN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg.CURRENT_MODEL).to(cfg.DEVICE)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
    criterion = nn.NLLLoss()
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=cfg.DEVICE)
    validator = Validator(model=model, criterion=criterion, device=cfg.DEVICE)
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOPPING_PATIENCE, min_delta=cfg.MIN_DELTA, save_path=cfg.BEST_MODEL)
    history = History()
    for epoch in range(cfg.EPOCHS):
        print("\n" + "=" * 60)
        print(f"Training | Epoch {epoch + 1}/{cfg.EPOCHS}")
        train_metrics = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = validator.validate(val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        stop = early_stopping(val_acc=val_acc, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        print(f"Train Loss      : {train_metrics['loss']:.4f}")
        print(f"Train Final Acc : {train_metrics['accuracy']:.4f}")
        print(f"Train Coarse Acc: {train_metrics['coarse_accuracy']:.4f}")
        print(f"Val Loss        : {val_loss:.4f}")
        print(f"Val Acc         : {val_acc:.4f}")
        print(f"LR              : {lr:.8f}")
        history.update(
            epoch=epoch + 1,
            train_loss=train_metrics["loss"],
            train_coarse_acc=train_metrics["coarse_accuracy"],
            train_final_acc=train_metrics["accuracy"],
            val_loss=val_loss,
            val_acc=val_acc,
            lr=lr
        )
        torch.save(
        {"epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()},
            cfg.LAST_MODEL
        )
        if stop:
            print("Early stopping triggered.")
            break
    history.save(cfg.TRAIN_SAVE_DIR / "history.csv")
    plot_training_curve(cfg.TRAIN_SAVE_DIR / "history.csv", cfg.TRAIN_SAVE_DIR)
    print("training completed.")

if __name__ == "__main__":
    main()