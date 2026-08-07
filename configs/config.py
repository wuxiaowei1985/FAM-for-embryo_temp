from pathlib import Path
import torch
from model.attention_model import FocusAttentionModel
from model.mean import MeanModel
from model.baseline import BaselineModel

# 训练
SEED = 42
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
MIN_LR = 1e-6
NUM_LAYERS = 4
NUM_WORKERS = 0
NUM_CLASSES = 16
EARLY_STOPPING = True
PATIENCE = 30
MIN_DELTA = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_MODEL = FocusAttentionModel(pretrained=True, num_classes=NUM_CLASSES)

# 获取 configs.py 所在目录的父目录（即与 configs 同级的目录）
CONFIG_DIR = Path(__file__).resolve().parent  # configs 文件夹
# 项目根目录
PROJECT_ROOT = CONFIG_DIR.parent  # configs 的上一级
# 数据集目录
DATA_ROOT_R = PROJECT_ROOT / "data"  # 相对路径
DATA_ROOT_A = Path(r"I:\datasets\胚胎\南特704")  # 绝对路径
# 输出目录
RUN_DIR = PROJECT_ROOT / "run"
# 保存
CURRENT_MODEL_DIR = "focus_attention_model"
SAVE_MODEL_DIR = PROJECT_ROOT / "checkpoints" / CURRENT_MODEL_DIR
TEST_MODEL_DIR = PROJECT_ROOT / "checkpoints" / CURRENT_MODEL_DIR / "best_0.pth"
# 测试结果
SAVE_RESULT_DIR = RUN_DIR / "per_class_accuracy.csv"
SAVE_CM_DIR = RUN_DIR / "confusion_matrix.csv"
# 模型性能
HISTORY_CSV = RUN_DIR / "history.csv"

if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

