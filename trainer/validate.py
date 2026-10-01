import torch
from tqdm import tqdm
from dataset.labels import get_coarse_label

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
        # log 空间下：logsumexp 应约等于 0（即概率和为 1）
        log_sum = torch.logsumexp(final_log_probs, dim=1)
        assert torch.allclose(
            log_sum, torch.zeros_like(log_sum), atol=1e-4
        ), f"final_log_probs not normalized, max |logsumexp| = {log_sum.abs().max().item():.6f}"
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
            images = batch["images"].to(self.device)
            labels = batch["label"].to(self.device)
            coarse_labels = torch.tensor([get_coarse_label(int(label)) for label in labels], dtype=torch.long, device=self.device,)
            output = self.model({"images": images, "label": labels}, stage="coarse", return_dict=True,)
            logits = output["coarse_logits"]
            loss = self.criterion(logits, coarse_labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            correct += (pred == coarse_labels).sum().item()
            total += batch_size
        return total_loss / max(total, 1), correct / max(total, 1)
    # ========================================================
    # Phase 2 validation
    # 最终输出 16 类，损失在 log-space 计算
    # ========================================================
    @torch.no_grad()
    def validate_fine(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc="Fine validate")
        for batch in progress:
            images = batch["images"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_size = labels.size(0)
            # ================================================
            # Forward
            # ================================================
            output = self.model({"images": images, "label": labels}, stage="fine", return_dict=True,)
            # ================================================
            # 使用 log-space 检查 + log-space 损失 与 train_fine_one_epoch 完全一致： L_final = -log P(y | x)
            # ================================================
            final_log_probs = self.check_log_probs(output["final_log_probs"])
            target_log_probs = final_log_probs[torch.arange(batch_size, device=self.device), labels]
            loss = -target_log_probs.mean()
            # ================================================
            # 预测：log 空间 argmax == 概率空间 argmax
            # 无需先 exp 再 argmax
            # ================================================
            pred = final_log_probs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += batch_size
            total_loss += loss.item() * batch_size
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / max(total, 1):.4f}"
            )
        return total_loss / max(total, 1), correct / max(total, 1)
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