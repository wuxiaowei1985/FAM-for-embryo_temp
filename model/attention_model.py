import torch
import torch.nn as nn
from model.utils.Backbone import Focal3DBackbone
from model.utils.focus_attention import FocusAttention
from model.utils.MSFD_Attention import MSFDAttention
from model.utils.classifier import ClassificationHead
from model.utils.coarse_embedding import CoarseEmbedding

class HierarchicalFocusAttentionModel(nn.Module):
    """
    Phase 1: 3D CNN -> P2/P3/P4 -> P4 Patch -> [B,448,512] -> Focus Attention -> [B,448,512] -> Token Pool -> [B,512] -> Coarse Head
    Phase 2:
        Phase-1 frozen
        Focus tokens + Coarse Embedding -> [B,448,512] -> branch-specific MSFD -> [B,448,512] -> Token Pool -> [B,512] -> Fine Head
    """
    def __init__(self, pretrained=False, feature_dim=512, coarse=3, pronuclear=3, cleavage=8, blastocyst=5, num_layers=2, dropout=0.2):
        super().__init__()
        if pretrained:
            raise ValueError("Pretrained weights are disabled.")
        if feature_dim != 512:
            raise ValueError("feature_dim must be 512.")
        self.feature_dim = feature_dim
        self.encoder = Focal3DBackbone(num_focus=7, embed_dim=512)
        self.focus_attention = FocusAttention(feature_dim=512, num_heads=8, depth=num_layers, num_focus=7, patches_per_focus=64, dropout=dropout)
        self.coarse_head = ClassificationHead(in_features=512, hidden_features=256, num_classes=coarse, dropout=dropout)
        self.coarse_embedding = CoarseEmbedding(num_classes=coarse, feature_dim=512)
        self.msfd_attention = nn.ModuleDict({
            "pronuclear":
                MSFDAttention(feature_dim=512, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, query_h=8, query_w=8, dropout=dropout),
            "cleavage":
                MSFDAttention(feature_dim=512, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, query_h=8, query_w=8, dropout=dropout),
            "blastocyst":
                MSFDAttention(feature_dim=512, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, query_h=8, query_w=8, dropout=dropout)
        })
        self.pronuclear_head = ClassificationHead(512, 256, pronuclear, dropout)
        self.cleavage_head = ClassificationHead(512, 256, cleavage, dropout)
        self.blastocyst_head = ClassificationHead(512, 256, blastocyst, dropout)

    @staticmethod
    def _select_pyramid(pyramid, mask):
        return [feature[mask] for feature in pyramid]

    def extract_backbone_features(self, images):
        return self.encoder(images)

    def extract_focus_features(self, images):
        backbone = self.extract_backbone_features(images)
        sequence, fused, token_weight, focus_weight = self.focus_attention(backbone["tokens"], return_sequence=True)
        return sequence, fused, token_weight, focus_weight, backbone
    # =========================================================
    # Phase 1
    # =========================================================
    def forward_coarse(self, images, return_dict=False):
        sequence, fused, token_weight, focus_weight, backbone = self.extract_focus_features(images)
        coarse_logits = self.coarse_head(fused)
        if return_dict:
            return {
                "coarse_logits": coarse_logits,
                "attention": token_weight,
                "focus_weight": focus_weight,
                "feature": fused,
                "sequence": sequence,
                "query_tokens": sequence,
                "tokens": backbone["tokens"],
                "p2": backbone["p2"],
                "p3": backbone["p3"],
                "p4": backbone["p4"],
                "pyramid": backbone
            }
        return coarse_logits
    # =========================================================
    # Phase 2
    # =========================================================
    def forward_fine(self, images, return_dict=False):
        backbone = self.extract_backbone_features(images)
        pyramid = [backbone["p2"], backbone["p3"], backbone["p4"]]
        # -----------------------------------------------------
        # Phase 1 frozen
        # -----------------------------------------------------
        with torch.no_grad():
            focus_tokens, coarse_feature, focus_token_weight, focus_weight = self.focus_attention(backbone["tokens"], return_sequence=True)
            coarse_logits = self.coarse_head(coarse_feature)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            coarse_pred = coarse_probs.argmax(dim=1)
        # -----------------------------------------------------
        # Coarse semantic embedding
        # -----------------------------------------------------
        coarse_embedding = self.coarse_embedding(coarse_probs)
        # [B,448,512]
        fine_query_tokens = (focus_tokens + coarse_embedding.unsqueeze(1))
        B = images.size(0)
        device = images.device
        dtype = fine_query_tokens.dtype
        # -----------------------------------------------------
        # Output containers
        # -----------------------------------------------------
        pronuclear_logits = torch.zeros(B, self.pronuclear_head.classifier[-1].out_features, device=device, dtype=dtype)
        cleavage_logits = torch.zeros(B, self.cleavage_head.classifier[-1].out_features, device=device, dtype=dtype)
        blastocyst_logits = torch.zeros(B, self.blastocyst_head.classifier[-1].out_features, device=device, dtype=dtype)
        pronuclear_attention = None
        cleavage_attention = None
        blastocyst_attention = None
        # =====================================================
        # Pronuclear
        # =====================================================
        mask_pronuclear = (coarse_pred == 0)
        if mask_pronuclear.any():
            branch_pyramid = self._select_pyramid(pyramid, mask_pronuclear)
            branch_query = fine_query_tokens[mask_pronuclear]
            branch_feature, branch_attention = self.msfd_attention["pronuclear"](branch_query, branch_pyramid, return_attention=True)
            pronuclear_logits[mask_pronuclear] = self.pronuclear_head(branch_feature)
            pronuclear_attention = branch_attention
        # =====================================================
        # Cleavage
        # =====================================================
        mask_cleavage = (coarse_pred == 1)
        if mask_cleavage.any():
            branch_pyramid = self._select_pyramid(pyramid, mask_cleavage)
            branch_query = fine_query_tokens[mask_cleavage]
            branch_feature, branch_attention = self.msfd_attention["cleavage"](branch_query, branch_pyramid, return_attention=True)
            cleavage_logits[mask_cleavage] = self.cleavage_head(branch_feature)
            cleavage_attention = branch_attention
        # =====================================================
        # Blastocyst
        # =====================================================
        mask_blastocyst = (coarse_pred == 2)
        if mask_blastocyst.any():
            branch_pyramid = self._select_pyramid(pyramid, mask_blastocyst)
            branch_query = fine_query_tokens[mask_blastocyst]
            branch_feature, branch_attention = self.msfd_attention["blastocyst"](branch_query, branch_pyramid, return_attention=True)
            blastocyst_logits[mask_blastocyst] = self.blastocyst_head(branch_feature)
            blastocyst_attention = branch_attention
        if return_dict:
            return {
                "coarse_logits": coarse_logits,
                "coarse_probs": coarse_probs,
                "coarse_pred": coarse_pred,
                "pronuclear_logits": pronuclear_logits,
                "cleavage_logits": cleavage_logits,
                "blastocyst_logits": blastocyst_logits,
                "focus_attention": focus_token_weight,
                "focus_weight": focus_weight,
                "sequence": fine_query_tokens,
                "feature": coarse_feature,
                "coarse_embedding": coarse_embedding,
                "pyramid": pyramid,
                "query_tokens": fine_query_tokens,
                "pronuclear_msfd_attention": pronuclear_attention,
                "cleavage_msfd_attention": cleavage_attention,
                "blastocyst_msfd_attention": blastocyst_attention,
            }
        return pronuclear_logits, cleavage_logits, blastocyst_logits

    def forward(self, batch, stage="coarse", return_dict=False):
        images = batch["images"]
        if stage == "coarse":
            return self.forward_coarse(images, return_dict=return_dict)
        if stage == "fine":
            return self.forward_fine(images, return_dict=return_dict)
        raise ValueError(f"Unknown stage: {stage}")

if __name__ == "__main__":
    model = HierarchicalFocusAttentionModel(feature_dim=512)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: "f"{num_params / 1e6:.3f} M")