import torch
from tqdm import tqdm

class Validator:
    def __init__(self, model, criterion, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.use_amp = device.type == "cuda"

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc="Validation")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model({"images": images, "label": labels}, return_dict=True)
                final_log_probs = output["final_log_probs"].float()
                loss = -final_log_probs.gather(1, labels.unsqueeze(1)).mean()
            preds = final_log_probs.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
        return total_loss / max(total, 1), correct / max(total, 1)
