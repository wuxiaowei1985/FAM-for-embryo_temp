import torch
from tqdm import tqdm
from dataset.labels import get_coarse_label
from configs import config as cfg

class Validator:
    def __init__(self, model, criterion, device, stage="coarse"):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.stage = stage
    # ========================================================
    # Log-space 安全检查
    # ========================================================
    @staticmethod
    def check_log_probs(final_log_probs, num_classes=16):
        """
        在 log 空间做安全检查，避免 exp 后再检查带来的误差。
        """
        assert final_log_probs.dim() == 2
        assert final_log_probs.size(1) == num_classes
        # ====================================================
        # log-space:
        # logsumexp(log P_1, ..., log P_n) = log(sum P_i) = log(1) = 0
        # ====================================================
        log_sum = torch.logsumexp(final_log_probs, dim=1)
        assert torch.allclose(log_sum, torch.zeros_like(log_sum), atol=1e-4), \
            "final_log_probs not normalized, "f"max |logsumexp| = "f"{log_sum.abs().max().item():.6f}"
        return final_log_probs
    # ========================================================
    # Phase 1 validation
    # ========================================================
    @torch.no_grad()
    def validate_coarse(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc="Coarse validate")
        for batch in progress:
            # ====================================================
            # Data
            # ====================================================
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            # ====================================================
            # 16 classes -> 3 coarse classes
            # ====================================================
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            # ====================================================
            # Forward
            # ====================================================
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                output = self.model(
                    {"images": images, "label": labels},
                    stage="coarse",
                    return_dict=True
                )
                logits = output["coarse_logits"]
                # =================================================
                # Coarse Loss
                # =================================================
                loss = self.criterion(logits,coarse_labels)
            # ====================================================
            # Statistics
            # ====================================================
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            correct += (pred == coarse_labels).sum().item()
            total += batch_size
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / max(total, 1):.4f}"
            )
        # ========================================================
        # Epoch statistics
        # ========================================================
        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc
    # ========================================================
    # Phase 2 validation
    # ========================================================
    # 最终输出 16 类。
    # Loss: L_final = -log P(y | x)
    # 全程保持 log-space。
    # ========================================================
    @torch.no_grad()
    def validate_fine(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc="Fine validate")
        for batch in progress:
            # ====================================================
            # Data
            # ====================================================
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            batch_size = labels.size(0)
            # ====================================================
            # Forward
            # ====================================================
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                output = self.model(
                    {"images": images, "label": labels},
                    stage="fine",
                    return_dict=True
                )
                # =================================================
                # Log-space check
                # =================================================
                final_log_probs = self.check_log_probs(output["final_log_probs"])
                # =================================================
                # Target log probability
                # =================================================
                target_log_probs = final_log_probs[torch.arange(batch_size, device=self.device), labels]

                # =================================================
                # Final 16-class Loss
                # =================================================
                loss = -target_log_probs.mean()
            # ====================================================
            # Prediction
            # ====================================================
            # log 空间 argmax = probability 空间 argmax
            # 无需 exp。
            # ====================================================
            pred = final_log_probs.argmax(dim=1)
            # ====================================================
            # Statistics
            # ====================================================
            correct += (pred == labels).sum().item()
            total += batch_size
            total_loss += loss.item() * batch_size
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / max(total, 1):.4f}"
            )
        # ========================================================
        # Epoch statistics
        # ========================================================
        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc
    # ========================================================
    # 统一接口
    # ========================================================
    def validate(self, loader):
        if self.stage == "coarse":
            return self.validate_coarse(loader)
        elif self.stage == "fine":
            return self.validate_fine(loader)
        else:
            raise ValueError(f"Unknown validation stage: {self.stage}")