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
    # 设置 Phase 2 冻结模块为 eval
    # 非常重要：requires_grad=False 不会阻止 BatchNorm 更新 running mean
    # ========================================================
    def set_frozen_modules_eval(self):
        if self.stage != "fine":
            return
        self.model.encoder.eval()
        self.model.focus_attention.eval()
        self.model.coarse_head.eval()
    # ========================================================
    # Phase 1
    # Backbone -> FocusAttention -> Coarse Head -> 3 classes
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
            # 16 → 3
            # ------------------------------------------------
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            self.optimizer.zero_grad()
            output = self.model(
                {"images": images, "label": labels},
                stage="coarse",
                return_dict=True
            )
            logits = output["coarse_logits"]
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
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    # ========================================================
    # Phase 2
    # Backbone -> FocusAttention -> Coarse Head -> Coarse Embedding -> MSFD Attention -> 3 Fine Heads
    # ========================================================
    def train_fine_one_epoch(self, loader):
        self.model.train()
        # 防止冻结模块中的 BatchNorm 更新
        self.set_frozen_modules_eval()
        total_loss = 0.0
        total = 0
        correct = 0
        progress = tqdm(loader, desc="Fine Training")
        for batch in progress:
            images = batch["images"].to(self.device)
            labels = batch["label"].to(self.device)
            # ------------------------------------------------
            # 得到 coarse label
            # ------------------------------------------------
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            # ------------------------------------------------
            # 得到 fine label
            # ------------------------------------------------
            fine_labels = torch.tensor(
                [get_fine_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            self.optimizer.zero_grad()
            output = self.model(
                {"images": images, "label": labels},
                stage="fine",
                return_dict=True
            )
            pronuclear_logits = output["pronuclear_logits"]
            cleavage_logits = output["cleavage_logits"]
            blastocyst_logits = output["blastocyst_logits"]
            # ------------------------------------------------
            # 根据 GT coarse stage 选择对应 head
            #注意：这里只用于训练 fine classifier。不会把 GT coarse label 输入模型。
            # ------------------------------------------------
            mask_pronuclear = (coarse_labels == 0)
            mask_cleavage = (coarse_labels == 1)
            mask_blastocyst = (coarse_labels == 2)
            loss_sum = torch.tensor(0.0, device=self.device)
            sample_count = 0
            # ------------------------------------------------
            # Pronuclear
            # ------------------------------------------------
            if mask_pronuclear.any():
                loss_pronuclear = self.criterion(pronuclear_logits[mask_pronuclear], fine_labels[mask_pronuclear])
                n = mask_pronuclear.sum()
                loss_sum += (loss_pronuclear * n)
                sample_count += n.item()
            # ------------------------------------------------
            # Cleavage
            # ------------------------------------------------
            if mask_cleavage.any():
                loss_cleavage = self.criterion(cleavage_logits[mask_cleavage], fine_labels[mask_cleavage])
                n = mask_cleavage.sum()
                loss_sum += (loss_cleavage * n)
                sample_count += n.item()
            # ------------------------------------------------
            # Blastocyst
            # ------------------------------------------------
            if mask_blastocyst.any():
                loss_blastocyst = self.criterion(blastocyst_logits[mask_blastocyst], fine_labels[mask_blastocyst])
                n = mask_blastocyst.sum()
                loss_sum += (loss_blastocyst * n)
                sample_count += n.item()
            # ------------------------------------------------
            # 总 fine loss
            # ------------------------------------------------
            loss = loss_sum / max(sample_count, 1)
            loss.backward()
            self.optimizer.step()
            # ------------------------------------------------
            # 训练准确率
            # 这里计算的是 conditional fine accuracy
            # ------------------------------------------------
            batch_correct = 0
            if mask_pronuclear.any():
                pred = pronuclear_logits[mask_pronuclear].argmax(dim=1)
                batch_correct += (pred == fine_labels[mask_pronuclear]).sum().item()
            if mask_cleavage.any():
                pred = cleavage_logits[mask_cleavage].argmax(dim=1)
                batch_correct += (pred == fine_labels[mask_cleavage]).sum().item()
            if mask_blastocyst.any():
                pred = blastocyst_logits[mask_blastocyst].argmax(dim=1)
                batch_correct += (pred == fine_labels[mask_blastocyst]).sum().item()
            batch_size = labels.size(0)
            total_loss += (loss.item() * batch_size)
            correct += batch_correct
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
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