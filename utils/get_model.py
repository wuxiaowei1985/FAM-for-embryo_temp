from model.attention_model import HierarchicalFocusAttentionModel
from model.mean import MeanModel
from model.baseline import BaselineModel
from configs import config as cfg

def build_model(model_name):
    """工厂函数：根据名称返回模型实例"""
    if model_name == "hierarchical_focus_attention":
        return HierarchicalFocusAttentionModel(pretrained=True, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif model_name == "baseline":
        return BaselineModel(pretrained=True,  num_classes=16)
    elif model_name == "mean":
        return MeanModel(pretrained=True, num_classes=16)
    else:
        raise ValueError(f"Unknown model name: {model_name}")