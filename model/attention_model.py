import torch
import torch.nn as nn
from model.utils.Backbone import Focal3DBackbone
from model.utils.focus_attention import FocusAttention
from model.utils.MSFD_Attention import MSFDAttention
from model.utils.classifier import ClassificationHead
from model.utils.coarse_embedding import CoarseEmbedding

class HierarchicalFocusAttentionModel(nn.Module):
    """
    Phase 1:
        3D CNN -> hierarchical pyramid -> patch tokens -> Transformer
        -> 7-plane focus sequence -> FocusAttention -> coarse head
    Phase 2:
        frozen Phase-1 modules -> coarse routing -> branch-specific
        3D multi-scale multi-focal deformable attention -> fine head
    """
    def __init__(self, pretrained=False, feature_dim=256, coarse=3, pronuclear=3, cleavage=8, blastocyst=5, num_layers=2, dropout=0.2):
        super().__init__()
        if pretrained:
            raise ValueError("The new focal 3D backbone intentionally does not use pretrained weights.")
        self.encoder = Focal3DBackbone(num_focus=7, embed_dim=512)
        self.feature_dim = self.encoder.feature_dim
        if feature_dim != self.feature_dim:
            raise ValueError(f"feature_dim must be {self.feature_dim}, got {feature_dim}")
        self.focus_attention = FocusAttention(feature_dim=self.feature_dim, num_heads=8, depth=num_layers, dropout=dropout)
        self.coarse_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=coarse, dropout=dropout)
        self.coarse_embedding = CoarseEmbedding(num_classes=coarse, feature_dim=self.feature_dim)
        self.msfd_attention = nn.ModuleDict({
            "pronuclear": MSFDAttention(self.feature_dim, 8, num_layers, 3, 4, dropout),
            "cleavage": MSFDAttention(self.feature_dim, 8, num_layers, 3, 4, dropout),
            "blastocyst": MSFDAttention(self.feature_dim, 8, num_layers, 3, 4, dropout),
        })
        self.pronuclear_head = ClassificationHead(self.feature_dim, 256, pronuclear, dropout)
        self.cleavage_head = ClassificationHead(self.feature_dim, 256, cleavage, dropout)
        self.blastocyst_head = ClassificationHead(self.feature_dim, 256, blastocyst, dropout)

    @staticmethod
    def _select_pyramid(pyramid, mask):
        return [feature[mask] for feature in pyramid]

    def extract_backbone_features(self, images):
        return self.encoder(images)

    def extract_focus_features(self, images):
        backbone = self.extract_backbone_features(images)
        sequence, fused, attention = self.focus_attention(backbone["focus_sequence"], return_sequence=True)
        return sequence, fused, attention, backbone

    def forward_coarse(self, images, return_dict=False):
        sequence, fused, attention, backbone = self.extract_focus_features(images)
        coarse_logits = self.coarse_head(fused)
        if return_dict:
            return {
                "coarse_logits": coarse_logits,
                "attention": attention,
                "feature": fused,
                "sequence": sequence,
                "focus_sequence": backbone["focus_sequence"],
                "pyramid": backbone,
                "query_tokens": backbone["query_tokens"],
            }
        return coarse_logits

    def forward_fine(self, images, return_dict=False):
        backbone = self.extract_backbone_features(images)
        query_tokens = backbone["query_tokens"]
        pyramid = [backbone["p2"], backbone["p3"], backbone["p4"]]
        with torch.no_grad():
            _, fused, focus_attention = self.focus_attention(backbone["focus_sequence"], return_sequence=True)
            coarse_logits = self.coarse_head(fused)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            coarse_pred = coarse_probs.argmax(dim=1)
        coarse_embedding = self.coarse_embedding(coarse_probs)
        fine_query_tokens = query_tokens + coarse_embedding.unsqueeze(1)
        B = images.size(0)
        device = images.device
        dtype = query_tokens.dtype
        pronuclear_logits = torch.zeros(B, self.pronuclear_head.classifier[-1].out_features, device=device, dtype=dtype)
        cleavage_logits = torch.zeros(B, self.cleavage_head.classifier[-1].out_features, device=device, dtype=dtype)
        blastocyst_logits = torch.zeros(B, self.blastocyst_head.classifier[-1].out_features, device=device, dtype=dtype)
        pronuclear_attention = None
        cleavage_attention = None
        blastocyst_attention = None
        mask_pronuclear = coarse_pred == 0
        if mask_pronuclear.any():
            branch_pyramid = self._select_pyramid(pyramid, mask_pronuclear)
            branch_query = fine_query_tokens[mask_pronuclear]
            branch_fused, branch_attention = self.msfd_attention["pronuclear"](branch_query, branch_pyramid)
            pronuclear_logits[mask_pronuclear] = self.pronuclear_head(branch_fused)
            pronuclear_attention = branch_attention
        mask_cleavage = coarse_pred == 1
        if mask_cleavage.any():
            branch_pyramid = self._select_pyramid(pyramid, mask_cleavage)
            branch_query = fine_query_tokens[mask_cleavage]
            branch_fused, branch_attention = self.msfd_attention["cleavage"](branch_query, branch_pyramid)
            cleavage_logits[mask_cleavage] = self.cleavage_head(branch_fused)
            cleavage_attention = branch_attention
        mask_blastocyst = coarse_pred == 2
        if mask_blastocyst.any():
            branch_pyramid = self._select_pyramid(pyramid, mask_blastocyst)
            branch_query = fine_query_tokens[mask_blastocyst]
            branch_fused, branch_attention = self.msfd_attention["blastocyst"](branch_query, branch_pyramid)
            blastocyst_logits[mask_blastocyst] = self.blastocyst_head(branch_fused)
            blastocyst_attention = branch_attention
        if return_dict:
            return {
                "coarse_logits": coarse_logits,
                "coarse_probs": coarse_probs,
                "coarse_pred": coarse_pred,
                "pronuclear_logits": pronuclear_logits,
                "cleavage_logits": cleavage_logits,
                "blastocyst_logits": blastocyst_logits,
                "focus_attention": focus_attention,
                "pronuclear_msfd_attention": pronuclear_attention,
                "cleavage_msfd_attention": cleavage_attention,
                "blastocyst_msfd_attention": blastocyst_attention,
                "feature": fused,
                "sequence": fine_query_tokens,
                "coarse_embedding": coarse_embedding,
                "pyramid": pyramid,
            }
        return pronuclear_logits, cleavage_logits, blastocyst_logits

    def forward(self, batch, stage="coarse", return_dict=False):
        images = batch["images"]
        if stage == "coarse":
            return self.forward_coarse(images, return_dict=return_dict)
        if stage == "fine":
            return self.forward_fine(images, return_dict=return_dict)
        raise ValueError(f"Unknown stage: {stage}")