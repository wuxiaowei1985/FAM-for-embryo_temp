import matplotlib.pyplot as plt
import pandas as pd
from configs import config as cfg

def plot_lr(save_dir, history, title):
    plt.figure(figsize=(8, 5))
    lr_name = title
    if title == "Learning Rate":
        lr_name = "lr_backbone"
    plt.plot(history["epoch"], history[lr_name], linewidth=2, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel(f"{title}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{title}.png", dpi=300)
    plt.close()

def plot_training_curve(csv_path, save_dir):
    history = pd.read_csv(csv_path)
    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=300)
    plt.close()
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_acc"], label="Train Accuracy", linewidth=2)
    plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curve.png", dpi=300)
    plt.close()
    # Learning Rate
    if cfg.FROZEN:
        plot_lr(save_dir, history, "lr_backbone")
        plot_lr(save_dir, history, "lr_fh")
    else:
        plot_lr(save_dir, history, "Learning Rate")