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
        pyramid = [ backbone["p2"], backbone["p3"], backbone["p4"]]
        # -----------------------------------------------------
        # Phase 1 frozen
        # -----------------------------------------------------
        with torch.no_grad():
            focus_tokens, coarse_feature, focus_token_weight, focus_weight = (self.focus_attention(backbone["tokens"], return_sequence=True))
            coarse_logits = self.coarse_head(coarse_feature)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            coarse_pred = coarse_probs.argmax(dim=1)
        # -----------------------------------------------------
        # Coarse semantic embedding
        # -----------------------------------------------------
        coarse_embedding = self.coarse_embedding(coarse_probs)
        # [B, 448, 512]
        fine_query_tokens = (focus_tokens + coarse_embedding.unsqueeze(1))
        # =====================================================
        # Pronuclear Expert
        # =====================================================
        # 训练阶段不需要保存 attention map。
        # return_attention=False 可以减少显存占用。
        branch_feature = self.msfd_attention["pronuclear"](fine_query_tokens, pyramid, return_attention=False)
        pronuclear_logits = self.pronuclear_head(branch_feature)
        # =====================================================
        # Cleavage Expert
        # =====================================================
        branch_feature = self.msfd_attention["cleavage"](fine_query_tokens, pyramid, return_attention=False)
        cleavage_logits = self.cleavage_head(branch_feature)
        # =====================================================
        # Blastocyst Expert
        # =====================================================
        branch_feature = self.msfd_attention["blastocyst"](fine_query_tokens, pyramid, return_attention=False)
        blastocyst_logits = self.blastocyst_head(branch_feature)
        # =====================================================
        # Soft Routing P(y | x) = P(coarse | x) * P(y | coarse, x)
        # ====================================================
        coarse_log_probs = torch.log_softmax(coarse_logits, dim=1)
        pronuclear_log_probs = torch.log_softmax(pronuclear_logits, dim=1)
        cleavage_log_probs = torch.log_softmax(cleavage_logits, dim=1)
        blastocyst_log_probs = torch.log_softmax(blastocyst_logits, dim=1)
        # -----------------------------------------------------
        # Construct final 16-class log probability
        # PN: 0 ~ 2
        # CL: 3 ~ 10
        # BL: 11 ~ 15
        # log P(y | x) = log P(c | x) + log P(y | c, x)
        # -----------------------------------------------------
        final_log_probs = torch.empty(images.size(0), 16, device=images.device, dtype=coarse_log_probs.dtype)
        # Pronuclear
        final_log_probs[:, 0:3] = (coarse_log_probs[:, 0:1] + pronuclear_log_probs)
        # Cleavage
        final_log_probs[:, 3:11] = (coarse_log_probs[:, 1:2] + cleavage_log_probs)
        # Blastocyst
        final_log_probs[:, 11:16] = (coarse_log_probs[:, 2:3] + blastocyst_log_probs)
        # -----------------------------------------------------
        # Convert to probability for inference / evaluation
        # -----------------------------------------------------
        final_probs = torch.exp(final_log_probs)
        # -----------------------------------------------------
        # Safety checks
        # -----------------------------------------------------
        assert final_log_probs.shape == (images.size(0), 16)
        assert final_probs.shape == (images.size(0), 16)
        if return_dict:
            return {
                "coarse_logits": coarse_logits,
                "coarse_probs": coarse_probs,
                "coarse_pred": coarse_pred,
                "pronuclear_logits": pronuclear_logits,
                "cleavage_logits": cleavage_logits,
                "blastocyst_logits": blastocyst_logits,
                "pronuclear_probs": torch.exp(pronuclear_log_probs),
                "cleavage_probs": torch.exp(cleavage_log_probs),
                "blastocyst_probs": torch.exp(blastocyst_log_probs),
                "final_log_probs": final_log_probs,
                "final_probs": final_probs,
                "focus_attention": focus_token_weight,
                "focus_weight": focus_weight,
                "sequence": fine_query_tokens,
                "feature": coarse_feature,
                "coarse_embedding": coarse_embedding,
                "pyramid": pyramid,
                "query_tokens": fine_query_tokens,
                "pronuclear_msfd_attention": None,
                "cleavage_msfd_attention": None,
                "blastocyst_msfd_attention": None,
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