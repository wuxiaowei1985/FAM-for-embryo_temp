import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

scaler = GradScaler("cuda")

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def move_to_device(self, batch):
        # 将 batch 中所有 Tensor 移动到 device。
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        return batch

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_num = 0
        pbar = tqdm(loader)
        for batch in pbar:
            batch = self.move_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                output = self.model(batch)
                if isinstance(output, dict):
                    logits = output["logits"]
                else:
                    logits = output
                loss = self.criterion(logits, batch["label"])
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            pred = logits.argmax(dim=1)
            correct = (pred == batch["label"]).sum().item()
            total_correct += correct
            total_num += batch["label"].size(0)
            total_loss += loss.item()
            pbar.set_description(f"loss={loss.item():.4f}")
        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / total_num
        return epoch_loss, epoch_acc


