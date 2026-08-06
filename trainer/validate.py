import torch
from tqdm import tqdm


class Validator:
    def __init__(self, model, criterion, device,):
        self.model = model
        self.criterion = criterion
        self.device = device

    def move_to_device(self, batch):
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        return batch

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_num = 0
        pbar = tqdm(loader)
        for batch in pbar:
            batch = self.move_to_device(batch)
            logits = self.model(batch)
            loss = self.criterion(logits, batch["label"])
            pred = logits.argmax(dim=1)
            total_correct += (pred == batch["label"]).sum().item()
            total_num += batch["label"].size(0)
            total_loss += loss.item()
            pbar.set_description(f"val_loss={loss.item():.4f}")
        loss = total_loss / len(loader)
        acc = total_correct / total_num
        return loss, acc