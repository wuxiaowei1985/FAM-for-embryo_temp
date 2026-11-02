import torch
import torch.nn as nn
from model.utils.Backbone import Focal3DBackbone
from model.utils.focus_attention import FocusAttention
from model.utils.MSFD_Attention import MSFDAttention
from model.utils.classifier import ClassificationHead
from model.utils.coarse_embedding import CoarseEmbedding

class HierarchicalFocusAttentionModel(nn.Module):
    """
    Single-stage joint training:
        3D Backbone -> 3D Focus Attention -> Coarse Head
        -> 3D Coarse Embedding -> 3D MSFD Experts
        -> Hierarchical 16-class prediction
    Input: [B,7,1,224,224]
    Output: [B,16]
    """
    def __init__(self, pretrained=False, feature_dim=512, coarse=3, pronuclear=3, cleavage=8, blastocyst=5, num_layers=2, dropout=0.2, image_size=224):
        super().__init__()
        if pretrained:
            raise ValueError("Pretrained weights are disabled.")
        if feature_dim != 512:
            raise ValueError("feature_dim must be 512.")
        if image_size != 224:
            raise ValueError("image_size must be 224.")
        self.feature_dim = feature_dim
        self.encoder = Focal3DBackbone(num_focus=7, embed_dim=feature_dim, image_size=224)
        self.focus_attention = FocusAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_focus=7, query_h=7, query_w=7, dropout=dropout)
        self.coarse_head = ClassificationHead(in_features=feature_dim, hidden_features=256, num_classes=coarse, dropout=dropout)
        self.coarse_embedding = CoarseEmbedding(num_classes=coarse, feature_dim=feature_dim, num_focus=7, query_h=7, query_w=7, dropout=dropout)
        self.msfd_attention = nn.ModuleDict({
            "pronuclear": MSFDAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, query_h=7, query_w=7, dropout=dropout),
            "cleavage": MSFDAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, query_h=7, query_w=7, dropout=dropout),
            "blastocyst": MSFDAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, query_h=7, query_w=7, dropout=dropout)
        })
        self.pronuclear_head = ClassificationHead(feature_dim, 256, pronuclear, dropout)
        self.cleavage_head = ClassificationHead(feature_dim, 256, cleavage, dropout)
        self.blastocyst_head = ClassificationHead(feature_dim, 256, blastocyst, dropout)

    def forward(self, batch, stage=None, return_dict=False):
        images = batch["images"] if isinstance(batch, dict) else batch
        if images.ndim != 5 or tuple(images.shape[1:]) != (7, 1, 224, 224):
            raise ValueError(f"Expected [B,7,1,224,224], got {tuple(images.shape)}")
        backbone = self.encoder(images)
        focus_tokens, coarse_feature, focus_token_weight, focus_weight = self.focus_attention(backbone["tokens"], return_sequence=True)
        coarse_logits = self.coarse_head(coarse_feature)
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        coarse_embedding = self.coarse_embedding(coarse_probs)
        fine_query_tokens = focus_tokens + coarse_embedding
        pyramid = [backbone["p2"], backbone["p3"], backbone["p4"]]
        pronuclear_feature = self.msfd_attention["pronuclear"](fine_query_tokens, pyramid)
        cleavage_feature = self.msfd_attention["cleavage"](fine_query_tokens, pyramid)
        blastocyst_feature = self.msfd_attention["blastocyst"](fine_query_tokens, pyramid)
        pronuclear_logits = self.pronuclear_head(pronuclear_feature)
        cleavage_logits = self.cleavage_head(cleavage_feature)
        blastocyst_logits = self.blastocyst_head(blastocyst_feature)
        coarse_log_probs = torch.log_softmax(coarse_logits.float(), dim=1)
        pronuclear_log_probs = torch.log_softmax(pronuclear_logits.float(), dim=1)
        cleavage_log_probs = torch.log_softmax(cleavage_logits.float(), dim=1)
        blastocyst_log_probs = torch.log_softmax(blastocyst_logits.float(), dim=1)
        final_log_probs = torch.cat([
            coarse_log_probs[:, 0:1] + pronuclear_log_probs,
            coarse_log_probs[:, 1:2] + cleavage_log_probs,
            coarse_log_probs[:, 2:3] + blastocyst_log_probs
        ], dim=1)
        if final_log_probs.shape != (images.size(0), 16):
            raise RuntimeError(f"Expected [B,16], got {tuple(final_log_probs.shape)}")
        output = {
            "coarse_logits": coarse_logits,
            "coarse_probs": coarse_probs,
            "coarse_pred": coarse_probs.argmax(dim=1),
            "pronuclear_logits": pronuclear_logits,
            "cleavage_logits": cleavage_logits,
            "blastocyst_logits": blastocyst_logits,
            "final_log_probs": final_log_probs,
            "final_probs": torch.exp(final_log_probs),
            "focus_attention": focus_token_weight,
            "focus_weight": focus_weight,
            "sequence": fine_query_tokens,
            "feature": coarse_feature,
            "coarse_embedding": coarse_embedding,
            "pyramid": pyramid,
            "query_tokens": fine_query_tokens
        }
        return output if return_dict else final_log_probs

if __name__ == "__main__":
    model = HierarchicalFocusAttentionModel()
    model.eval()
    with torch.no_grad():
        output = model({"images": torch.randn(1, 7, 1, 224, 224)}, return_dict=True)
    print("final_log_probs:", output["final_log_probs"].shape)
    print("focus_weight:", output["focus_weight"].shape)
    print("coarse_embedding:", output["coarse_embedding"].shape)
    print("pyramid:", [tuple(x.shape) for x in output["pyramid"]])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")