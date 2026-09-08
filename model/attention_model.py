import torch
import torch.nn as nn
from model.utils.shared_encoder import SharedEncoder
from model.utils.focus_attention import FocusAttention
from model.utils.MSFD_Attention import MSFDAttention
from model.utils.classifier import ClassificationHead
from model.utils.coarse_embedding import CoarseEmbedding

class HierarchicalFocusAttentionModel(nn.Module):
    def __init__(self, pretrained=True, feature_dim=512, coarse=3, pronuclear=3, cleavage=8, blastocyst=5, num_layers=4, dropout=0.2):
        super().__init__()
        # =====================================================
        # Shared Encoder
        # =====================================================
        self.encoder = SharedEncoder(pretrained=pretrained)
        self.feature_dim = self.encoder.feature_dim
        # =====================================================
        # Focus Attention
        # =====================================================
        self.focus_attention = FocusAttention(feature_dim=self.feature_dim, num_heads=8, depth=num_layers, dropout=dropout)
        # =====================================================
        # Phase 1
        # Coarse classification
        # =====================================================
        self.coarse_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=coarse, dropout=dropout)
        # =====================================================
        # Phase 2
        # Coarse semantic embedding
        # =====================================================
        self.coarse_embedding = CoarseEmbedding(num_classes=coarse, feature_dim=self.feature_dim)
        # =====================================================
        # Phase 2
        # MSFD Attention
        # =====================================================
        self.msfd_attention = MSFDAttention(feature_dim=self.feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, dropout=dropout)
        # =====================================================
        # Phase 2
        # Fine classification
        # =====================================================
        self.pronuclear_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=pronuclear, dropout=dropout)
        self.cleavage_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=cleavage, dropout=dropout)
        self.blastocyst_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=blastocyst, dropout=dropout)
    # =========================================================
    # Backbone + Focus Attention
    # =========================================================
    def extract_focus_features(self, images):
        features = self.encoder(images)
        # [B,7,512]
        sequence, fused, attention = self.focus_attention(features, return_sequence=True)
        return sequence, fused, attention
    # =========================================================
    # Phase 1
    # =========================================================
    def forward_coarse(self, images, return_dict=False):
        sequence, fused, attention = self.extract_focus_features(images)
        logits = self.coarse_head(fused)
        if return_dict:
            return {
                "logits": logits,
                "attention": attention,
                "feature": fused,
                "sequence": sequence
            }
        return logits
    # =========================================================
    # Phase 2
    # =========================================================
    def forward_fine(self, images, return_dict=False):
        sequence, fused, focus_attention = self.extract_focus_features(images)
        # -----------------------------------------------------
        # Coarse semantic embedding
        # -----------------------------------------------------
        with torch.no_grad():
            coarse_logits = self.coarse_head(fused)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
        coarse_embedding = self.coarse_embedding(coarse_probs)
        # [B,512] → [B,1,512] → [B,7,512]
        coarse_embedding = coarse_embedding.unsqueeze(1)
        coarse_embedding = coarse_embedding.expand(-1, sequence.size(1), -1)
        # -----------------------------------------------------
        # 注入 coarse semantic information
        # -----------------------------------------------------
        sequence = sequence + coarse_embedding
        # -----------------------------------------------------
        # MSFD Attention
        # -----------------------------------------------------
        fused, msfd_attention = self.msfd_attention(sequence)
        # -----------------------------------------------------
        # Fine classification
        # -----------------------------------------------------
        pronuclear_logits = self.pronuclear_head(fused)
        cleavage_logits = self.cleavage_head(fused)
        blastocyst_logits = self.blastocyst_head(fused)
        if return_dict:
            return {
                "coarse_logits": coarse_logits,
                "coarse_probs": coarse_probs,
                "pronuclear_logits": pronuclear_logits,
                "cleavage_logits": cleavage_logits,
                "blastocyst_logits": blastocyst_logits,
                "focus_attention": focus_attention,
                "msfd_attention": msfd_attention,
                "feature": fused
            }
        return pronuclear_logits, cleavage_logits, blastocyst_logits
    # =========================================================
    # Unified forward
    # =========================================================
    def forward(self, batch, stage="coarse", return_dict=False):
        images = batch["images"]
        if stage == "coarse":
            return self.forward_coarse(images, return_dict=return_dict)
        elif stage == "fine":
            return self.forward_fine(images, return_dict=return_dict)
        else:
            raise ValueError(f"Unknown stage: {stage}")