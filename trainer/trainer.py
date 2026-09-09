import torch
from tqdm import tqdm
from dataset.labels import get_coarse_label, get_fine_label

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
    # Phase 1 modules:
    #     Encoder
    #     Focus Attention
    #     Coarse Head
    # 全部冻结。
    # Routing:
    #     Phase 1 coarse prediction
    #              ↓
    #            argmax
    #              ↓
    #     ┌────────┼────────┐
    #     ↓        ↓        ↓
    #    PN       CL       BL
    # 注意：
    # GT coarse 不参与 branch 选择。GT coarse 只用于判断：predicted coarse == GT coarse
    # 如果 routing 正确：对对应 Fine Expert 计算 loss
    # 如果 routing 错误：不计算 Fine loss
    # 这样避免不同 Fine Head 的 label space 冲突。
    # ========================================================
    def train_fine_one_epoch(self, loader):
        self.model.train()
        # ----------------------------------------------------
        # Phase 1 modules frozen + eval
        # ----------------------------------------------------
        self.set_frozen_modules_eval()
        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------
        total_loss = 0.0
        # Fine classification accuracy
        # 只统计 routing 正确的样本
        fine_correct = 0
        fine_total = 0
        # Coarse routing accuracy
        routing_correct = 0
        routing_total = 0
        # All samples
        total_samples = 0
        progress = tqdm(loader, desc="Fine Training")
        for batch in progress:
            images = batch["images"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_size = labels.size(0)
            total_samples += batch_size
            # =================================================
            # 16-class GT
            # =================================================
            fine_labels = torch.tensor(
                [get_fine_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            # =================================================
            # 3-class GT
            # 注意：这里仅用于判断 routing 是否正确，不用于选择 branch。
            # =================================================
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            # =================================================
            # Forward
            # Model 内部已经执行：Backbone -> Focus Attention -> Coarse Head -> coarse_pred -> hard routing -> selected MSFD -> selected Fine Head
            # =================================================
            self.optimizer.zero_grad()
            output = self.model(
                {"images": images, "label": labels},
                stage="fine",
                return_dict=True
            )
            # =================================================
            # Coarse prediction
            # =================================================
            coarse_pred = output["coarse_pred"]
            # -------------------------------------------------
            # Routing accuracy
            # -------------------------------------------------
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
            # 重要：只让 routing 正确的样本参与 Fine Loss。branch 仍然由 coarse_pred 决定。
            # =================================================
            # -------------------------------------------------
            # Pronuclear
            # coarse_pred == 0 AND coarse_GT == 0
            # -------------------------------------------------
            mask_pronuclear = ((coarse_pred == 0) & (coarse_labels == 0))
            # -------------------------------------------------
            # Cleavage
            # coarse_pred == 1 AND coarse_GT == 1
            # -------------------------------------------------
            mask_cleavage = ((coarse_pred == 1) & (coarse_labels == 1))
            # -------------------------------------------------
            # Blastocyst
            # coarse_pred == 2 AND coarse_GT == 2
            # -------------------------------------------------
            mask_blastocyst = ((coarse_pred == 2) & (coarse_labels == 2))
            # =================================================
            # Loss accumulator
            # =================================================
            loss_sum = torch.tensor(0.0, device=self.device)
            valid_sample_count = 0
            batch_fine_correct = 0
            # =================================================
            # Pronuclear branch
            # =================================================
            if mask_pronuclear.any():
                logits = pronuclear_logits[mask_pronuclear]
                # get_fine_label() 已经将：
                # tPB2 → 0
                # tPNa → 1
                # tPNf → 2
                # 映射到了 Pronuclear 的局部 label space。
                target = fine_labels[mask_pronuclear]
                # 安全检查
                assert (target.min().item() >= 0)
                assert (target.max().item() < logits.size(1))
                loss = self.criterion(logits, target)
                n = mask_pronuclear.sum()
                loss_sum = (loss_sum + loss * n)
                valid_sample_count += n.item()
                pred = logits.argmax(dim=1)
                batch_fine_correct += (pred == target).sum().item()
            # =================================================
            # Cleavage branch
            # =================================================
            if mask_cleavage.any():
                logits = cleavage_logits[mask_cleavage]
                # get_fine_label() 已经将：
                # t2  → 0
                # t3  → 1
                # ...
                # t9+ → 7
                # 映射到了 Cleavage 的局部 label space。
                target = fine_labels[mask_cleavage]
                # 安全检查
                assert (target.min().item() >= 0)
                assert (target.max().item() < logits.size(1))
                loss = self.criterion(logits, target)
                n = mask_cleavage.sum()
                loss_sum = (loss_sum + loss * n)
                valid_sample_count += n.item()
                pred = logits.argmax(dim=1)
                batch_fine_correct += (pred == target).sum().item()
            # =================================================
            # Blastocyst branch
            # =================================================
            if mask_blastocyst.any():
                logits = blastocyst_logits[mask_blastocyst]
                # get_fine_label() 已经将：
                # tM  → 0
                # tSB → 1
                # tB  → 2
                # tEB → 3
                # tHB → 4
                # 映射到了 Blastocyst 的局部 label space。
                target = fine_labels[mask_blastocyst]
                # 安全检查
                assert (target.min().item() >= 0)
                assert (target.max().item() < logits.size(1))
                loss = self.criterion(logits, target)
                n = mask_blastocyst.sum()
                loss_sum = (loss_sum + loss * n)
                valid_sample_count += n.item()
                pred = logits.argmax(dim=1)
                batch_fine_correct += (pred == target).sum().item()
            # =================================================
            # 如果当前 batch 没有正确 routing 的样本，则没有 Fine Loss。不能：loss.backward()，因为此时 loss 与任何 trainable parameter，没有有效的计算图连接。
            # =================================================
            if valid_sample_count == 0:
                progress.set_postfix(
                    loss="skip",
                    route_acc= f"{routing_correct / routing_total:.4f}",
                    fine_acc= f"{fine_correct / max(fine_total, 1):.4f}",
                    valid=0
                )
                continue
            # =================================================
            # 当前 batch Fine Loss，不同 branch 的样本数量可能不同，所以按照有效样本数量进行加权平均。
            # =================================================
            loss = (loss_sum / valid_sample_count)
            # =================================================
            # Backward 只有 Phase 2 模块会更新：
            # CoarseEmbedding
            # PN-MSFD
            # CL-MSFD
            # BL-MSFD
            # PN Head
            # CL Head
            # BL Head
            # =================================================
            loss.backward()
            self.optimizer.step()
            # =================================================
            # Statistics
            # =================================================
            total_loss += (loss.item() * valid_sample_count)
            fine_correct += (batch_fine_correct)
            fine_total += (valid_sample_count)
            # =================================================
            # Progress
            # =================================================
            route_acc = (routing_correct / max(routing_total, 1))
            fine_acc = (fine_correct / max(fine_total, 1))
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                route_acc=f"{route_acc:.4f}",
                fine_acc=f"{fine_acc:.4f}",
                valid=valid_sample_count,
                pn=int(mask_pronuclear.sum()),
                cl=int(mask_cleavage.sum()),
                bl=int(mask_blastocyst.sum()),
            )
        # ====================================================
        # Epoch statistics
        # ====================================================
        # Fine loss：只对 routing 正确的样本统计。
        epoch_loss = (total_loss / max(fine_total, 1))
        # Conditional Fine Accuracy：在 routing 正确的样本中，Fine Expert 的分类准确率。
        epoch_fine_acc = (fine_correct / max(fine_total, 1))
        # Coarse routing accuracy：所有样本中 coarse prediction 正确的比例。
        epoch_routing_acc = (routing_correct / max(routing_total, 1))
        print(f"\nPhase 2 Routing Accuracy : " f"{epoch_routing_acc * 100:.2f}%")
        print(f"Phase 2 Conditional Fine Accuracy : " f"{epoch_fine_acc * 100:.2f}%")
        print(f"Phase 2 Fine-valid Samples : " f"{fine_total}/{total_samples}")
        return epoch_loss, epoch_fine_acc
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