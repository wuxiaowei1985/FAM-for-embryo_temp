import pandas as pd

class History:
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr_backbone": [],
            "lr_fh": []
        }
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr_backbone, lr_fh):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        self.history["lr_backbone"].append(lr_backbone)
        self.history["lr_fh"].append(lr_fh)

    def save(self, path):
        pd.DataFrame(self.history).to_csv(path, index=False)