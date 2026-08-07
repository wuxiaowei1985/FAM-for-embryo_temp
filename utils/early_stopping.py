import torch
from pathlib import Path


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve.
    """
    def __init__(self, patience = 10, min_delta = 0.0, save_path = None, verbose = True):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = float("-inf")
        self.early_stop = False

    def __call__( self, val_acc, model, optimizer=None, scheduler=None, epoch=None):
        # 验证集Loss有提升
        if val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            if self.save_path is not None:
                checkpoint = {
                    "epoch": epoch+1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer else None,
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "best_acc": self.best_score
                }
                torch.save(checkpoint, self.save_path)
                print("best model saved")
            if self.verbose:
                print(f"Validation accuracy improved to {val_acc:.4f}%")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop