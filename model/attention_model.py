import torch
import torch.nn as nn
from model.utils.shared_encoder import SharedEncoder
from model.utils.focus_attention import FocusAttention
from model.utils.MSFD_Attention import MSFDAttention
from model.utils.classifier import ClassificationHead
from model.utils.coarse_embedding import CoarseEmbedding

class HierarchicalFocusAttentionModel(nn.Module):
    """
    双阶段层级式多焦平面胚胎发育阶段分类模型。
    Phase 1: SharedEncoder -> FocusAttention -> CoarseHead -> 3-class coarse prediction
    Phase 2: Frozen Phase-1 modules -> Coarse prediction -> Hard routing -> one of:          ->            16-class prediction
                                                                            Pronuclear: MSFD + PronuclearHead
                                                                            Cleavage:   MSFD + CleavageHead
                                                                            Blastocyst: MSFD + BlastocystHead
    """
    def __init__(self, pretrained=True, feature_dim=512, coarse=3, pronuclear=3, cleavage=8, blastocyst=5, num_layers=4, dropout=0.2):
        super().__init__()
        # =====================================================
        # Shared Encoder
        # =====================================================
        self.encoder = SharedEncoder(pretrained=pretrained)
        self.feature_dim = self.encoder.feature_dim
        # =====================================================
        # Phase 1
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
        # 保留原有模块，不删除。将 coarse prediction 转换为条件信息，注入进入三个 fine branch 的 sequence。
        # =====================================================
        self.coarse_embedding = CoarseEmbedding(num_classes=coarse, feature_dim=self.feature_dim)
        # =====================================================
        # Phase 2
        # Three independent MSFD Attention branches
        # 保留原来的 msfd_attention 模块名称， 但内部变成三个相互独立的 MSFDAttention。
        # =====================================================
        self.msfd_attention = nn.ModuleDict({
            "pronuclear": MSFDAttention(feature_dim=self.feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, dropout=dropout),
            "cleavage": MSFDAttention(feature_dim=self.feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, dropout=dropout,),
            "blastocyst": MSFDAttention(feature_dim=self.feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, dropout=dropout)
        })
        # =====================================================
        # Phase 2
        # Three independent fine classification heads
        # =====================================================
        self.pronuclear_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=pronuclear, dropout=dropout)
        self.cleavage_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=cleavage, dropout=dropout)
        self.blastocyst_head = ClassificationHead(in_features=self.feature_dim, hidden_features=256, num_classes=blastocyst, dropout=dropout)
    # =========================================================
    # Backbone Feature Extraction
    # =========================================================
    def extract_backbone_features(self, images):
        """
        仅通过 SharedEncoder 提取每个焦平面的特征。
        images: [B, 7, 1, 224, 224]
        features: [B, 7, 512]
        该特征同时供：
            1. Phase 1 Focus Attention
            2. Phase 2 MSFD Attention
        使用。
        """
        features = self.encoder(images)
        return features
    # =========================================================
    # Backbone + Focus Attention
    # Phase 1 专用
    # =========================================================
    def extract_focus_features(self, images):
        """
        Phase 1: Backbone -> Focus Attention
        images: [B, 7, 1, 224, 224]
        features: [B, 7, 512]
        sequence: [B, 7, 512]
        fused: [B, 512]
        attention: Focus Attention 对 7 个焦平面的权重
        """
        features = self.extract_backbone_features(images)
        sequence, fused, attention = self.focus_attention(features, return_sequence=True)
        return sequence, fused, attention
    # =========================================================
    # Phase 1
    # =========================================================
    def forward_coarse(self, images, return_dict=False):
        sequence, fused, attention = self.extract_focus_features(images)
        coarse_logits = self.coarse_head(fused)
        if return_dict:
            return {
                "coarse_logits": coarse_logits,
                "attention": attention,
                "feature": fused,
                "sequence": sequence,
            }
        return coarse_logits
    # =========================================================
    # Phase 2
    # Backbone -> Coarse Routing
    #         -> corresponding MSFD -> Fine Head
    # =========================================================
    def forward_fine(self, images, return_dict=False):
        features = self.extract_backbone_features(images)
        # =====================================================
        # 2. Phase 1 coarse prediction
        # 注意：Focus Attention 只用于 coarse prediction，不再把它的 sequence 传给 MSFD。
        # =====================================================
        with torch.no_grad():
            _, fused, focus_attention = self.focus_attention(features, return_sequence=True)
            coarse_logits = self.coarse_head(fused)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            coarse_pred = coarse_probs.argmax(dim=1)
        # =====================================================
        # 3. Coarse semantic embedding
        # 保留原有 CoarseEmbedding。
        # coarse_probs: [B, 3]
        # coarse_embedding: [B, 512]
        # =====================================================
        coarse_embedding = self.coarse_embedding(coarse_probs)
        coarse_embedding = coarse_embedding.unsqueeze(1)
        coarse_embedding = coarse_embedding.expand(-1, features.size(1), -1)
        # =====================================================
        # 5. 注入 coarse semantic information
        # 注意：这里使用的是 Backbone features，而不是 Focus Attention sequence。
        # features: [B, 7, 512]
        # coarse_embedding: [B, 7, 512]
        # result: [B, 7, 512]
        # =====================================================
        fine_features = features + coarse_embedding
        # =====================================================
        # 6. 初始化三个 branch 的输出，为了保持现有 trainer / validator / tester 接口，仍然返回三个完整 batch-size 的 logits。未被选择的 branch 对应位置保持 0。
        # =====================================================
        batch_size = features.size(0)
        device = features.device
        dtype = features.dtype
        pronuclear_logits = torch.zeros(batch_size, self.pronuclear_head.classifier[-1].out_features, device=device, dtype=dtype)
        cleavage_logits = torch.zeros(batch_size, self.cleavage_head.classifier[-1].out_features, device=device, dtype=dtype)
        blastocyst_logits = torch.zeros(batch_size, self.blastocyst_head.classifier[-1].out_features, device=device, dtype=dtype)
        mask_pronuclear = coarse_pred == 0
        if mask_pronuclear.any():
            branch_features = fine_features[mask_pronuclear]
            branch_fused, branch_attention = self.msfd_attention["pronuclear"](branch_features)
            branch_logits = self.pronuclear_head(branch_fused)
            pronuclear_logits[mask_pronuclear] = branch_logits
            pronuclear_attention = branch_attention
        else:
            pronuclear_attention = None
        mask_cleavage = coarse_pred == 1
        if mask_cleavage.any():
            branch_features = fine_features[mask_cleavage]
            branch_fused, branch_attention = self.msfd_attention["cleavage"](branch_features)
            branch_logits = self.cleavage_head(branch_fused)
            cleavage_logits[mask_cleavage] = branch_logits
            cleavage_attention = branch_attention
        else:
            cleavage_attention = None
        mask_blastocyst = coarse_pred == 2
        if mask_blastocyst.any():
            branch_features = fine_features[mask_blastocyst]
            branch_fused, branch_attention = self.msfd_attention["blastocyst"](branch_features)
            branch_logits = self.blastocyst_head(branch_fused)
            blastocyst_logits[mask_blastocyst] = branch_logits
            blastocyst_attention = branch_attention
        else:
            blastocyst_attention = None
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
                "sequence": fine_features,
                "coarse_embedding": coarse_embedding,
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