import torch
from tqdm import tqdm
from torch.amp import autocast
from torch.amp import GradScaler

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()

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
            with autocast():
                output = self.model(batch)
                if isinstance(output, dict):
                    logits = output["logits"]
                else:
                    logits = output
                loss = self.criterion(logits, batch["label"])
            self.scaler.scale(loss).backward()
            # 1. 先 unscale 梯度（将缩放因子还原）
            self.scaler.unscale_(self.optimizer)
            # 2. 裁剪原始梯度（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # 3. 更新参数
            self.scaler.step(self.optimizer)
            # 4. 更新缩放器
            self.scaler.update()
            pred = logits.argmax(dim=1)
            correct = (pred == batch["label"]).sum().item()
            total_correct += correct
            total_num += batch["label"].size(0)
            total_loss += loss.item()
            pbar.set_description(f"loss={loss.item():.4f}")
        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / total_num
        return epoch_loss, epoch_acc


