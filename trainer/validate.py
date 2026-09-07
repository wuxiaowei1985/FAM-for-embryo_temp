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
            output = self.model(
                {"images": images, "label": labels},
                stage="fine",
                return_dict=True
            )
            coarse_logits = output["coarse_logits"]
            pronuclear_logits = output["pronuclear_logits"]
            cleavage_logits = output["cleavage_logits"]
            blastocyst_logits = output["blastocyst_logits"]
            # =================================================
            # coarse probability
            # =================================================
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            # =================================================
            # fine probability
            # =================================================
            pronuclear_probs = torch.softmax(pronuclear_logits, dim=1)
            cleavage_probs = torch.softmax(cleavage_logits, dim=1)
            blastocyst_probs = torch.softmax(blastocyst_logits, dim=1)
            # =================================================
            # hierarchical probability
            # P(fine) = P(coarse) × P(fine | coarse)
            # =================================================
            final_probs = torch.cat(
                [coarse_probs[:, 0:1] * pronuclear_probs, coarse_probs[:, 1:2] * cleavage_probs, coarse_probs[:, 2:3] * blastocyst_probs,],
                dim=1
            )
            # =================================================
            # 检查
            # =================================================
            assert final_probs.shape[1] == 16
            # =================================================
            # 最终预测
            # =================================================
            pred = final_probs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            # -------------------------------------------------
            # validation loss
            # 使用最终16分类概率计算 NLL
            # -------------------------------------------------
            selected_probs = final_probs[torch.arange(labels.size(0), device=self.device), labels]
            loss = -torch.log(selected_probs.clamp_min(1e-8)).mean()
            total_loss += (loss.item() * labels.size(0))
        return total_loss / total, correct / total
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