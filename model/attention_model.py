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
    # Backbone + Focus Attention
    # =========================================================
    def extract_focus_features(self, images):
        """
        images: [B, 7, 1, 224, 224]
        features: [B, 7, 512]
        sequence: [B, 7, 512]
        fused: [B, 512]
        """
        features = self.encoder(images)
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
    # =========================================================
    def forward_fine(self, images, return_dict=False):
        """
        Phase 2 forward.
        Important: Phase-1 modules are frozen externally by train.py.
        Routing: coarse_pred = argmax(coarse_logits)
            0 -> pronuclear branch
            1 -> cleavage branch
            2 -> blastocyst branch
        Only the selected branch is executed for each sample.
        """
        # -----------------------------------------------------
        # Phase-1 feature extraction
        # -----------------------------------------------------
        sequence, fused, focus_attention = self.extract_focus_features(images)
        # -----------------------------------------------------
        # Coarse prediction
        # Phase 2 中 coarse_head 已冻结，因此这里不需要梯度。
        # -----------------------------------------------------
        with torch.no_grad():
            coarse_logits = self.coarse_head(fused)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            coarse_pred = coarse_probs.argmax(dim=1)
        # -----------------------------------------------------
        # Coarse semantic embedding
        # 保留原有 CoarseEmbedding 模块。这里使用 coarse probability 作为条件信息。coarse_embedding 本身属于 Phase 2，因此可以训练。
        # -----------------------------------------------------
        coarse_embedding = self.coarse_embedding(coarse_probs)
        # [B, 512] -> [B, 1, 512] -> [B, 7, 512]
        coarse_embedding = coarse_embedding.unsqueeze(1)
        coarse_embedding = coarse_embedding.expand(-1, sequence.size(1), -1)
        # -----------------------------------------------------
        # 注入 coarse semantic information
        # -----------------------------------------------------
        sequence = sequence + coarse_embedding
        # -----------------------------------------------------
        # 初始化输出，为了保持 batch 输出尺寸统一，三个 branch 的结果分别保存到对应位置。未被选择的 branch 保持 None。
        # -----------------------------------------------------
        batch_size = sequence.size(0)
        device = sequence.device
        pronuclear_logits = torch.zeros(batch_size, self.pronuclear_head.classifier[-1].out_features, device=device, dtype=sequence.dtype)
        cleavage_logits = torch.zeros(batch_size, self.cleavage_head.classifier[-1].out_features, device=device, dtype=sequence.dtype)
        blastocyst_logits = torch.zeros(batch_size, self.blastocyst_head.classifier[-1].out_features, device=device, dtype=sequence.dtype)
        # =====================================================
        # Branch 0: Pronuclear
        # =====================================================
        mask_pronuclear = coarse_pred == 0
        if mask_pronuclear.any():
            branch_sequence = sequence[mask_pronuclear]
            branch_fused, branch_attention = self.msfd_attention["pronuclear"](branch_sequence)
            branch_logits = self.pronuclear_head(branch_fused)
            pronuclear_logits[mask_pronuclear] = branch_logits
        else:
            branch_attention = None
        pronuclear_attention = branch_attention
        # =====================================================
        # Branch 1: Cleavage
        # =====================================================
        mask_cleavage = coarse_pred == 1
        if mask_cleavage.any():
            branch_sequence = sequence[mask_cleavage]
            branch_fused, branch_attention = self.msfd_attention["cleavage"](branch_sequence)
            branch_logits = self.cleavage_head(branch_fused)
            cleavage_logits[mask_cleavage] = branch_logits
        else:
            branch_attention = None
        cleavage_attention = branch_attention
        # =====================================================
        # Branch 2: Blastocyst
        # =====================================================
        mask_blastocyst = coarse_pred == 2
        if mask_blastocyst.any():
            branch_sequence = sequence[mask_blastocyst]
            branch_fused, branch_attention = self.msfd_attention["blastocyst"](branch_sequence)
            branch_logits = self.blastocyst_head(branch_fused)
            blastocyst_logits[mask_blastocyst] = branch_logits
        else:
            branch_attention = None
        blastocyst_attention = branch_attention
        # =====================================================
        # Final output
        # =====================================================
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
                "sequence": sequence,
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