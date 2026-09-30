import torch
from tqdm import tqdm
from dataset.labels import get_coarse_label, get_fine_label
from configs import config as cfg

class Trainer:
    def __init__(self, model, criterion, optimizer, device, stage="coarse"):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.stage = stage
    # ========================================================
    # 设置 Phase 2 冻结模块为 eval，requires_grad=False 不能阻止 BatchNorm 更新 running mean / running variance。因此 Phase 2 中必须让 Phase 1 模块保持 eval。
    # ========================================================
    def set_frozen_modules_eval(self):
        if self.stage != "fine":
            return
        self.model.encoder.eval()
        self.model.focus_attention.eval()
        self.model.coarse_head.eval()
    # ========================================================
    # Phase 1
    # Backbone -> Focus Attention -> Coarse Head -> 3 classes
    # ========================================================
    def train_coarse_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc="Coarse Training")
        for batch in progress:
            images = batch["images"].to(self.device)
            labels = batch["label"].to(self.device)
            # ------------------------------------------------
            # 16 classes → 3 coarse classes
            # ------------------------------------------------
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            # ------------------------------------------------
            # forward
            # ------------------------------------------------
            self.optimizer.zero_grad()
            output = self.model(
                {"images": images, "label": labels},
                stage="coarse",
                return_dict=True
            )
            logits = output["coarse_logits"]
            # ------------------------------------------------
            # coarse loss
            # ------------------------------------------------
            loss = self.criterion(logits, coarse_labels)
            loss.backward()
            self.optimizer.step()
            # ------------------------------------------------
            # statistics
            # ------------------------------------------------
            batch_size = labels.size(0)
            total_loss += (loss.item() * batch_size)
            pred = logits.argmax(dim=1)
            correct += (pred == coarse_labels).sum().item()
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")
        epoch_loss = (total_loss / max(total, 1))
        epoch_acc = (correct / max(total, 1))
        return epoch_loss, epoch_acc
    # ========================================================
    # Phase 2
    # Phase 1: Encoder Focus Attention Coarse Head 全部冻结。
    # Phase 2: 三个 Fine Expert 对所有样本全部计算。
    #     Fine Loss: 使用 GT coarse 找到对应 expert，但 GT coarse 只用于计算监督 loss，不参与模型 inference routing。
    #     Final Loss: 使用 Soft Routing 得到最终 16-class probability，直接计算 16-class CE。
    #     L = L_fine + lambda * L_final
    # ========================================================
    def train_fine_one_epoch(self, loader):
        self.model.train()
        # ----------------------------------------------------
        # Phase 1 modules frozen + eval
        # ----------------------------------------------------
        self.set_frozen_modules_eval()
        # ----------------------------------------------------
        # Final 16-class loss weight
        # ----------------------------------------------------
        final_loss_weight = cfg.FINAL_LOSS_WEIGHT
        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------
        total_loss = 0.0
        total_fine_loss = 0.0
        total_final_loss = 0.0
        # Conditional Fine classification accuracy
        fine_correct = 0
        fine_total = 0
        # Coarse routing accuracy
        routing_correct = 0
        routing_total = 0
        # Final 16-class accuracy
        final_correct = 0
        final_total = 0
        progress = tqdm(loader, desc="Fine Training")
        for batch in progress:
            images = batch["images"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_size = labels.size(0)
            # =================================================
            # 16-class GT
            # =================================================
            fine_labels = torch.tensor([get_fine_label(int(label)) for label in labels], dtype=torch.long, device=self.device)
            # =================================================
            # 3-class GT
            # 注意：
            # GT coarse 不参与模型 inference routing。
            # 这里只用于：
            #   1. Fine Loss
            #   2. routing accuracy
            # =================================================
            coarse_labels = torch.tensor([get_coarse_label(int(label)) for label in labels], dtype=torch.long, device=self.device)
            # =================================================
            # Forward
            # =================================================
            self.optimizer.zero_grad()
            output = self.model({"images": images, "label": labels}, stage="fine", return_dict=True)
            # =================================================
            # Coarse prediction
            # =================================================
            coarse_pred = output["coarse_pred"]
            routing_correct_mask = (coarse_pred == coarse_labels)
            routing_correct += (routing_correct_mask.sum().item())
            routing_total += batch_size
            # =================================================
            # Fine logits
            # =================================================
            pronuclear_logits = output["pronuclear_logits"]
            cleavage_logits = output["cleavage_logits"]
            blastocyst_logits = output["blastocyst_logits"]
            # =================================================
            # Final 16-class log probability
            # =================================================
            final_log_probs = output["final_log_probs"]
            # =================================================
            # Fine Loss
            # 关键：不再使用 coarse_pred。 直接根据 GT coarse 找到对应 expert，因此所有样本都会参与 Fine Loss。
            # =================================================
            fine_loss_sum = torch.tensor(0.0, device=self.device)
            fine_sample_count = 0
            batch_fine_correct = 0
            # -------------------------------------------------
            # Pronuclear
            # -------------------------------------------------
            mask_pronuclear = (coarse_labels == 0)
            if mask_pronuclear.any():
                logits = pronuclear_logits[mask_pronuclear]
                target = fine_labels[mask_pronuclear]
                assert target.min().item() >= 0
                assert target.max().item() < logits.size(1)
                loss = self.criterion(logits, target)
                n = mask_pronuclear.sum()
                fine_loss_sum = (fine_loss_sum + loss * n)
                fine_sample_count += n.item()
                pred = logits.argmax(dim=1)
                batch_fine_correct += (pred == target).sum().item()
            # -------------------------------------------------
            # Cleavage
            # -------------------------------------------------
            mask_cleavage = (coarse_labels == 1)
            if mask_cleavage.any():
                logits = cleavage_logits[mask_cleavage]
                target = fine_labels[mask_cleavage]
                assert target.min().item() >= 0
                assert target.max().item() < logits.size(1)
                loss = self.criterion(logits, target)
                n = mask_cleavage.sum()
                fine_loss_sum = (fine_loss_sum + loss * n)
                fine_sample_count += n.item()
                pred = logits.argmax(dim=1)
                batch_fine_correct += (pred == target).sum().item()
            # -------------------------------------------------
            # Blastocyst
            # -------------------------------------------------
            mask_blastocyst = (coarse_labels == 2)
            if mask_blastocyst.any():
                logits = blastocyst_logits[mask_blastocyst]
                target = fine_labels[mask_blastocyst]
                assert target.min().item() >= 0
                assert target.max().item() < logits.size(1)
                loss = self.criterion(logits, target)
                n = mask_blastocyst.sum()
                fine_loss_sum = (fine_loss_sum + loss * n)
                fine_sample_count += n.item()
                pred = logits.argmax(dim=1)
                batch_fine_correct += (pred == target).sum().item()
            # =================================================
            # Fine Loss
            # =================================================
            if fine_sample_count == 0:
                raise RuntimeError("Fine Loss has no valid samples. ""Please check coarse/fine label mapping.")
            fine_loss = (fine_loss_sum / fine_sample_count)
            # =================================================
            # Final 16-class Loss
            # 直接在 log probability 空间计算： L_final = -log P(y | x) 不再： probability -> clamp -> log
            # =================================================
            target_log_probs = final_log_probs[torch.arange(batch_size, device=self.device), labels]
            final_loss = -target_log_probs.mean()
            # =================================================
            # Total Loss
            # L = L_fine + lambda * L_final
            # =================================================
            loss = (fine_loss + final_loss_weight * final_loss)
            # =================================================
            # Backward
            # =================================================
            loss.backward()
            self.optimizer.step()
            # =================================================
            # Statistics
            # =================================================
            # Fine
            total_fine_loss += (fine_loss.item() * fine_sample_count)
            fine_correct += batch_fine_correct
            fine_total += fine_sample_count
            # Final 16-class
            final_pred = final_log_probs.argmax(dim=1)
            batch_final_correct = (final_pred == labels).sum().item()
            final_correct += batch_final_correct
            final_total += batch_size
            total_final_loss += (final_loss.item() * batch_size)
            total_loss += (loss.item() * batch_size)
            # =================================================
            # Progress
            # =================================================
            route_acc = (routing_correct / max(routing_total, 1))
            fine_acc = (fine_correct / max(fine_total, 1))
            final_acc = (final_correct / max(final_total, 1))
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                fine_loss=f"{fine_loss.item():.4f}",
                final_loss=f"{final_loss.item():.4f}",
                coarse_acc=f"{route_acc:.4f}",
                fine_acc=f"{fine_acc:.4f}",
                final_acc=f"{final_acc:.4f}"
            )
        # ====================================================
        # Epoch statistics
        # ====================================================
        epoch_loss = (total_loss / max(final_total, 1))
        epoch_coarse_acc = (routing_correct / max(routing_total, 1))
        epoch_fine_acc = (fine_correct / max(fine_total, 1))
        epoch_final_acc = (final_correct / max(final_total, 1))
        return epoch_loss, epoch_coarse_acc, epoch_fine_acc, epoch_final_acc
    # ========================================================
    # 统一接口
    # ========================================================
    def train_one_epoch(self, loader):
        if self.stage == "coarse":
            return self.train_coarse_one_epoch(loader)
        elif self.stage == "fine":
            return self.train_fine_one_epoch(loader)
        else:
            raise ValueError(f"Unknown training stage: {self.stage}")