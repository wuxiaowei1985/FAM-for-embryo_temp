from model.attention_model import FocusAttentionModel
from model.msfd_attention import MSFDAttentionModel
from model.mean import MeanModel
from model.baseline import BaselineModel
from configs import config as cfg

def build_model(model_name):
    """工厂函数：根据名称返回模型实例"""
    if model_name == "focus_attention":
        return FocusAttentionModel(pretrained=True, num_classes=cfg.NUM_CLASSES, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif model_name == "baseline":
        return BaselineModel(pretrained=True,  num_classes=cfg.NUM_CLASSES)
    elif model_name == "mean":
        return MeanModel(pretrained=True, num_classes=cfg.NUM_CLASSES)
    elif model_name == "msfd_attention":
        return MSFDAttentionModel(pretrained=True,  num_classes=cfg.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name: {model_name}")