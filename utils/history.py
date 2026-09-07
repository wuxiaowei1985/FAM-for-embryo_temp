import pandas as pd

class History:
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": []
        }
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(lr)

    def save(self, path):
        pd.DataFrame(self.history).to_csv(path, index=False)