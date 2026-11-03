from pathlib import Path
import torch

SEED = 42
NUM_WORKERS = 4
NUM_AUG_VIEWS = 2
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
NUM_LAYERS = 2
DROPOUT = 0.2
COARSE_LOSS_WEIGHT = 0.5
EXPERT_LOSS_WEIGHT = 0.5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
MIN_LR = 1e-7
EARLY_STOPPING_PATIENCE = 20
MIN_DELTA = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_MODEL = "hierarchical_focus_attention"

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DATA_ROOT_R = PROJECT_ROOT / "data"
DATA_ROOT_A = Path(r"your dataset path")
DATA_ROOT = DATA_ROOT_A

RUN_DIR = PROJECT_ROOT / "run"
SAVE_MODEL_DIR = PROJECT_ROOT / "checkpoints" / CURRENT_MODEL
TRAIN_SAVE_DIR = RUN_DIR / "history"
BEST_MODEL = SAVE_MODEL_DIR / "best_model.pth"
LAST_MODEL = SAVE_MODEL_DIR / "last_model.pth"
TEST_MODEL_DIR = BEST_MODEL

SAVE_RESULT_DIR = RUN_DIR / "per_class_accuracy.csv"
SAVE_CM_DIR = RUN_DIR / "confusion_matrix.csv"
SAVE_REPORT_DIR = RUN_DIR / "classification_report.csv"