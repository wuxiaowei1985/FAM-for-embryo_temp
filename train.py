import torch
import torch.nn as nn
from trainer.trainer import Trainer
from trainer.validate import Validator
from dataset.loader import *
from configs import config as cfg

def main():
    model = cfg.CURRENT_MODEL
    trainer = Trainer(
        model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg.LR),
        device=cfg.DEVICE
    )
    validator = Validator(
        model,
        criterion=nn.CrossEntropyLoss(),
        device=cfg.DEVICE
    )
    best_acc = 0
    for epoch in range(cfg.EPOCHS):
        print("=" * 60)
        print(f"Epoch {epoch+1}")
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = validator.validate(val_loader)
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.SAVE_MODEL_DIR / f"best_{epoch}.pth")
            print("Best model saved.")

if __name__ == "__main__":
    main()