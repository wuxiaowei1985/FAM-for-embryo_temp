import torch
from tqdm import tqdm
from dataset.labels import get_coarse_label

class Validator:
    def __init__(self, model, criterion, device, stage="coarse"):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.stage = stage
    def build_routed_final_probs(self, coarse_pred, pronuclear_logits, cleavage_logits, blastocyst_logits):
        """
        根据 coarse prediction 进行 hard routing，构造最终 16 类概率。
        coarse:
            0 -> pronuclear: 3 classes
            1 -> cleavage:   8 classes
            2 -> blastocyst: 5 classes
        Output:
            [B, 16]
        """
        pronuclear_probs = torch.softmax(pronuclear_logits, dim=1)
        cleavage_probs = torch.softmax(cleavage_logits, dim=1)
        blastocyst_probs = torch.softmax(blastocyst_logits, dim=1)
        batch_size = coarse_pred.size(0)
        device = coarse_pred.device
        final_probs = torch.zeros(batch_size, 16, device=device, dtype=pronuclear_probs.dtype)
        # ========================================================
        # Pronuclear
        # ========================================================
        mask = coarse_pred == 0
        if mask.any():
            final_probs[mask, 0:3] = pronuclear_probs[mask]
        # ========================================================
        # Cleavage
        # ========================================================
        mask = coarse_pred == 1
        if mask.any():
            final_probs[mask, 3:11] = cleavage_probs[mask]
        # ========================================================
        # Blastocyst
        # ========================================================
        mask = coarse_pred == 2
        if mask.any():
            final_probs[mask, 11:16] = blastocyst_probs[mask]
        # ========================================================
        # Safety check
        # ========================================================
        assert final_probs.shape[1] == 16
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
            output = self.model(
                {"images": images, "label": labels},
                stage="fine",
                return_dict=True
            )
            coarse_pred = output["coarse_pred"]
            pronuclear_logits = output["pronuclear_logits"]
            cleavage_logits = output["cleavage_logits"]
            blastocyst_logits = output["blastocyst_logits"]
            # ====================================================
            # Hard routing
            # ====================================================
            final_probs = self.build_routed_final_probs(coarse_pred, pronuclear_logits, cleavage_logits, blastocyst_logits)
            # ====================================================
            # Safety checks
            # ====================================================
            assert final_probs.shape == (labels.size(0), 16)
            assert torch.allclose(final_probs.sum(dim=1), torch.ones(labels.size(0), device=self.device), atol=1e-5)
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
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")
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