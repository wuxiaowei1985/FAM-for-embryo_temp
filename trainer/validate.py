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
    # Soft Routing
    # ========================================================
    @staticmethod
    def build_soft_final_probs(output):
        """
        使用模型内部的 soft routing 构造最终 16 类概率。
        final_probs: [B, 16]
        P(y | x) = P(coarse | x) * P(y | coarse, x)
        """
        final_probs = output["final_probs"]
        assert final_probs.dim() == 2
        assert final_probs.size(1) == 16
        # ----------------------------------------------------
        # Probability safety check
        # ----------------------------------------------------
        probability_sum = final_probs.sum(dim=1)
        assert torch.allclose(probability_sum, torch.ones_like(probability_sum), atol=1e-5)
        return final_probs
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
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            output = self.model(
                {"images": images, "label": labels},
                stage="coarse",
                return_dict=True
            )
            logits = output["coarse_logits"]
            loss = self.criterion(logits, coarse_labels)
            batch_size = labels.size(0)
            total_loss += (loss.item() * batch_size)
            pred = logits.argmax(dim=1)
            correct += (pred == coarse_labels).sum().item()
            total += batch_size
        return total_loss / total, correct / total
    # ========================================================
    # Phase 2 validation
    # 最终输出16类
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
            # ====================================================
            # Forward
            # ====================================================
            output = self.model({"images": images, "label": labels}, stage="fine", return_dict=True)
            # ====================================================
            # Soft routed final probability
            # ====================================================
            final_probs = self.build_soft_final_probs(output)
            # ====================================================
            # Final prediction
            # ====================================================
            pred = final_probs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            # ====================================================
            # 16-class NLL
            # ====================================================
            selected_probs = final_probs[torch.arange(labels.size(0), device=self.device), labels]
            loss = -torch.log(selected_probs.clamp_min(1e-8)).mean()
            total_loss += (loss.item() * labels.size(0))
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