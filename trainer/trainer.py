import torch
import torch.distributed as dist
from tqdm import tqdm
from dataset.labels import get_coarse_label, get_fine_label
from configs import config as cfg
from utils.distributed import unwrap_model

class Trainer:
    def __init__(self, model, criterion, optimizer, device, stage="coarse"):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.stage = stage
        self.use_amp = (cfg.USE_AMP and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        # ====================================================
        # Distributed
        # ====================================================
        self.rank = (dist.get_rank() if dist.is_available() and dist.is_initialized() else 0)
    # ========================================================
    # Reduce Statistics
    # ========================================================
    def reduce_statistics(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()
    # ========================================================
    # Frozen modules
    # ========================================================
    def set_frozen_modules_eval(self):
        if self.stage != "fine":
            return
        model = unwrap_model(self.model)
        model.encoder.eval()
        model.focus_attention.eval()
        model.coarse_head.eval()
    # ========================================================
    # Phase 1
    # ========================================================
    def train_coarse_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"Coarse Training [GPU {self.rank}]", disable=self.rank != 0)
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model(
                    {"images": images, "label": labels},
                    stage="coarse",
                    return_dict=True
                )
                logits = output["coarse_logits"]
                loss = self.criterion(logits, coarse_labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = labels.size(0)
            total_loss += (loss.item() * batch_size)
            pred = logits.argmax(dim=1)
            correct += (pred == coarse_labels).sum().item()
            total += batch_size
            if self.rank == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{correct / max(total, 1):.4f}"
                )
        # ====================================================
        # Distributed reduction
        # ====================================================
        total_loss, correct, total = self.reduce_statistics([total_loss, correct, total])
        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc
    # ========================================================
    # Phase 2
    # ========================================================
    def train_fine_one_epoch(self, loader):
        self.model.train()
        self.set_frozen_modules_eval()
        final_loss_weight = cfg.FINAL_LOSS_WEIGHT
        total_loss = 0.0
        total_fine_loss = 0.0
        total_final_loss = 0.0
        fine_correct = 0
        fine_total = 0
        routing_correct = 0
        routing_total = 0
        final_correct = 0
        final_total = 0
        progress = tqdm(loader, desc=f"Fine Training [GPU {self.rank}]", disable=self.rank != 0)
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            batch_size = labels.size(0)
            fine_labels = torch.tensor(
                [get_fine_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            coarse_labels = torch.tensor(
                [get_coarse_label(int(label)) for label in labels],
                dtype=torch.long,
                device=self.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model(
                    {"images": images, "label": labels},
                    stage="fine",
                    return_dict=True
                )
                coarse_pred = output["coarse_pred"]
                routing_correct_mask = coarse_pred == coarse_labels
                pronuclear_logits = output["pronuclear_logits"]
                cleavage_logits = output["cleavage_logits"]
                blastocyst_logits = output["blastocyst_logits"]
                final_log_probs = output["final_log_probs"]
                fine_loss_sum = torch.tensor(0.0, device=self.device)
                fine_sample_count = 0
                batch_fine_correct = 0
                # ====================================================
                # Pronuclear
                # ====================================================
                mask_pronuclear = coarse_labels == 0
                if mask_pronuclear.any():
                    logits = pronuclear_logits[mask_pronuclear]
                    target = fine_labels[mask_pronuclear]
                    assert target.min().item() >= 0
                    assert target.max().item() < logits.size(1)
                    loss = self.criterion(logits, target)
                    n = mask_pronuclear.sum()
                    fine_loss_sum = fine_loss_sum + loss * n
                    fine_sample_count += n.item()
                    pred = logits.argmax(dim=1)
                    batch_fine_correct += (pred == target).sum().item()
                # ====================================================
                # Cleavage
                # ====================================================
                mask_cleavage = coarse_labels == 1
                if mask_cleavage.any():
                    logits = cleavage_logits[mask_cleavage]
                    target = fine_labels[mask_cleavage]
                    assert target.min().item() >= 0
                    assert target.max().item() < logits.size(1)
                    loss = self.criterion(logits, target)
                    n = mask_cleavage.sum()
                    fine_loss_sum = fine_loss_sum + loss * n
                    fine_sample_count += n.item()
                    pred = logits.argmax(dim=1)
                    batch_fine_correct += (pred == target).sum().item()
                # ====================================================
                # Blastocyst
                # ====================================================
                mask_blastocyst = coarse_labels == 2
                if mask_blastocyst.any():
                    logits = blastocyst_logits[mask_blastocyst]
                    target = fine_labels[mask_blastocyst]
                    assert target.min().item() >= 0
                    assert target.max().item() < logits.size(1)
                    loss = self.criterion(logits,  target)
                    n = mask_blastocyst.sum()
                    fine_loss_sum = fine_loss_sum + loss * n
                    fine_sample_count += n.item()
                    pred = logits.argmax(dim=1)
                    batch_fine_correct += (pred == target).sum().item()
                if fine_sample_count == 0:
                    raise RuntimeError(
                        "Fine Loss has no valid samples. ""Please check coarse/fine label mapping.")
                fine_loss = fine_loss_sum / fine_sample_count
                target_log_probs = final_log_probs[torch.arange(batch_size, device=self.device), labels]
                final_loss = -target_log_probs.mean()
                loss = fine_loss + final_loss_weight * final_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            routing_correct += routing_correct_mask.sum().item()
            routing_total += batch_size
            total_fine_loss += fine_loss.item() * fine_sample_count
            fine_correct += batch_fine_correct
            fine_total += fine_sample_count
            final_pred = final_log_probs.argmax(dim=1)
            batch_final_correct = (final_pred == labels).sum().item()
            final_correct += batch_final_correct
            final_total += batch_size
            total_final_loss += final_loss.item() * batch_size
            total_loss += loss.item() * batch_size
            if self.rank == 0:
                route_acc = routing_correct / max(routing_total, 1)
                fine_acc = fine_correct / max(fine_total, 1)
                final_acc = final_correct / max(final_total, 1)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    fine_loss=f"{fine_loss.item():.4f}",
                    final_loss=f"{final_loss.item():.4f}",
                    coarse_acc=f"{route_acc:.4f}",
                    fine_acc=f"{fine_acc:.4f}",
                    final_acc=f"{final_acc:.4f}"
                )
        # ====================================================
        # Distributed reduction
        # ====================================================
        total_loss, total_fine_loss, total_final_loss, fine_correct, fine_total, routing_correct, routing_total, final_correct, final_total = self.reduce_statistics(
            [total_loss, total_fine_loss, total_final_loss, fine_correct, fine_total, routing_correct, routing_total, final_correct, final_total]
        )
        epoch_loss = total_loss / max(final_total, 1)
        epoch_coarse_acc = routing_correct / max(routing_total, 1)
        epoch_fine_acc = fine_correct / max(fine_total, 1)
        epoch_final_acc = final_correct / max(final_total, 1)
        return epoch_loss, epoch_coarse_acc, epoch_fine_acc, epoch_final_acc
    # ========================================================
    # Unified Interface
    # ========================================================
    def train_one_epoch(self, loader):
        if self.stage == "coarse":
            return self.train_coarse_one_epoch(loader)
        if self.stage == "fine":
            return self.train_fine_one_epoch(loader)
        raise ValueError(f"Unknown training stage: {self.stage}")