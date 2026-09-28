import pandas as pd

class History:
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_coarse_acc": [],
            "train_fine_acc": [],
            "train_final_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }
    def update(self, epoch, train_loss, train_coarse_acc, train_fine_acc, train_final_acc, val_loss, val_acc, lr):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_coarse_acc"].append(train_coarse_acc)
        self.history["train_fine_acc"].append(train_fine_acc)
        self.history["train_final_acc"].append(train_final_acc)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(lr)

    def save(self, path):
        pd.DataFrame(self.history).to_csv(path, index=False)