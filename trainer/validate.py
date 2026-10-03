import torch
import torch.distributed as dist
from tqdm import tqdm
from dataset.labels import get_coarse_label
from configs import config as cfg

class Validator:
    def __init__(self, model, criterion, device, stage="coarse"):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.stage = stage
        self.rank = (dist.get_rank() if dist.is_available() and dist.is_initialized() else 0)
        self.use_amp = (cfg.USE_AMP and self.device.type == "cuda")
    # ========================================================
    # Distributed Reduction
    # ========================================================
    def reduce_statistics(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()
    # ========================================================
    # Log-space Safety Check
    # ========================================================
    @staticmethod
    def check_log_probs(final_log_probs, num_classes=16):
        assert final_log_probs.dim() == 2
        assert final_log_probs.size(1) == num_classes
        log_sum = torch.logsumexp(final_log_probs, dim=1)
        assert torch.allclose(log_sum, torch.zeros_like(log_sum), atol=1e-4), \
            "final_log_probs not normalized, "f"max |logsumexp| = "f"{log_sum.abs().max().item():.6f}"
        return final_log_probs
    # ========================================================
    # Phase 1 Validation
    # ========================================================
    @torch.no_grad()
    def validate_coarse(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"Coarse Validate [GPU {self.rank}]", disable=self.rank != 0)
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model(
                    {"images": images, "label": labels},
                    stage="coarse",
                    return_dict=True
                )
                logits = output["coarse_logits"]
                loss = self.criterion(logits, coarse_labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            correct += (pred == coarse_labels).sum().item()
            total += batch_size
            if self.rank == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{correct / max(total, 1):.4f}"
                )
        total_loss, correct, total = (self.reduce_statistics([total_loss, correct, total]))
        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc
    # ========================================================
    # Phase 2 Validation
    # ========================================================
    @torch.no_grad()
    def validate_fine(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"Fine Validate [GPU {self.rank}]", disable=self.rank != 0)
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            batch_size = labels.size(0)
            with torch.autocast( device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model(
                    {"images": images, "label": labels},
                    stage="fine",
                    return_dict=True
                )
                final_log_probs = self.check_log_probs(output["final_log_probs"])

                target_log_probs = final_log_probs[torch.arange(batch_size, device=self.device), labels]
                loss = -target_log_probs.mean()
            pred = final_log_probs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += batch_size
            total_loss += loss.item() * batch_size
            if self.rank == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{correct / max(total, 1):.4f}"
                )
        total_loss, correct, total = self.reduce_statistics([total_loss, correct, total])
        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc
    # ========================================================
    # Unified Interface
    # ========================================================
    def validate(self, loader):
        if self.stage == "coarse":
            return self.validate_coarse(loader)
        if self.stage == "fine":
            return self.validate_fine(loader)
        raise ValueError(f"Unknown validation stage: {self.stage}")