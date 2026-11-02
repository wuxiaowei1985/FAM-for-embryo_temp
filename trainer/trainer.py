import torch
from tqdm import tqdm
from dataset.labels import get_coarse_label, get_fine_label

class Trainer:
    def __init__(self, model, criterion, optimizer, device, coarse_loss_weight=0.2, expert_loss_weight=0.2):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.coarse_loss_weight = coarse_loss_weight
        self.expert_loss_weight = expert_loss_weight
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_final_loss = 0.0
        total_coarse_loss = 0.0
        total_expert_loss = 0.0
        correct = 0
        coarse_correct = 0
        total = 0
        progress = tqdm(loader, desc="Joint Training")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            coarse_labels = torch.tensor([get_coarse_label(int(y)) for y in labels], device=self.device, dtype=torch.long)
            fine_labels = torch.tensor([get_fine_label(int(y)) for y in labels], device=self.device, dtype=torch.long)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model({"images": images, "label": labels}, return_dict=True)
                final_log_probs = output["final_log_probs"].float()
                final_loss = self.criterion(final_log_probs, labels) if isinstance(self.criterion, torch.nn.NLLLoss) else -final_log_probs.gather(1, labels.unsqueeze(1)).mean()
                coarse_loss = torch.nn.functional.cross_entropy(output["coarse_logits"].float(), coarse_labels)
                expert_loss_sum = final_loss.new_zeros(())
                expert_sample_count = 0
                for coarse_id, key in enumerate(("pronuclear_logits", "cleavage_logits", "blastocyst_logits")):
                    mask = coarse_labels == coarse_id
                    if mask.any():
                        expert_logits = output[key][mask].float()
                        expert_targets = fine_labels[mask]
                        expert_loss_sum = expert_loss_sum + torch.nn.functional.cross_entropy(expert_logits, expert_targets, reduction="sum")
                        expert_sample_count += int(mask.sum().item())

                expert_loss = expert_loss_sum / max(expert_sample_count, 1)
                loss = final_loss + self.coarse_loss_weight * coarse_loss + self.expert_loss_weight * expert_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = labels.size(0)
            preds = final_log_probs.argmax(dim=1)
            coarse_preds = output["coarse_logits"].argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_final_loss += final_loss.item() * batch_size
            total_coarse_loss += coarse_loss.item() * batch_size
            total_expert_loss += expert_loss.item() * batch_size
            correct += (preds == labels).sum().item()
            coarse_correct += (coarse_preds == coarse_labels).sum().item()
            total += batch_size
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                final_loss=f"{final_loss.item():.4f}",
                acc=f"{correct / max(total, 1):.4f}"
            )
        return {
            "loss": total_loss / max(total, 1),
            "final_loss": total_final_loss / max(total, 1),
            "coarse_loss": total_coarse_loss / max(total, 1),
            "expert_loss": total_expert_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "coarse_accuracy": coarse_correct / max(total, 1)
        }