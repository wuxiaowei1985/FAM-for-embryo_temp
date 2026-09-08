from pathlib import Path
import torch

# 训练
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4           # 注意力/分类头用正常学习率
MIN_LR = 1e-7
NUM_LAYERS = 2
DROPOUT = 0.4
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
FACTOR = 0.5
SCHEDULER_PATIENCE = 5
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
MIN_DELTA = 0.0

TRAIN_STAGE = "both"    #"coarse"、"fine"、"both"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_MODEL = "hierarchical_focus_attention"

# 获取 configs.py 所在目录的父目录（即与 configs 同级的目录）
CONFIG_DIR = Path(__file__).resolve().parent  # configs 文件夹
# 项目根目录
PROJECT_ROOT = CONFIG_DIR.parent  # configs 的上一级
# 数据集目录
DATA_ROOT_R = PROJECT_ROOT / "data"  # 相对路径
DATA_ROOT_A = Path(r"your dataset path")  # 绝对路径
DATA_ROOT = DATA_ROOT_R
# 输出目录
RUN_DIR = PROJECT_ROOT / "run"
# 保存
SAVE_MODEL_DIR = PROJECT_ROOT / "checkpoints" / CURRENT_MODEL
PHASE1_BEST_MODEL = SAVE_MODEL_DIR / "phase1_best_model.pth"
PHASE1_LAST_MODEL = SAVE_MODEL_DIR / "phase1_last_model.pth"
PHASE2_BEST_MODEL = SAVE_MODEL_DIR / "phase2_best_model.pth"
PHASE2_LAST_MODEL = SAVE_MODEL_DIR / "phase2_last_model.pth"
PHASE2_MODEL = PHASE1_BEST_MODEL
TEST_MODEL_DIR = PHASE2_BEST_MODEL
# 测试结果
PHASE1_SAVE = RUN_DIR / "phase1"
PHASE2_SAVE = RUN_DIR / "phase2"
SAVE_RESULT_DIR = RUN_DIR / "per_class_accuracy.csv"
SAVE_CM_DIR = RUN_DIR / "confusion_matrix.csv"
SAVE_REPORT_DIR = RUN_DIR / "classification_report.csv"