import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from configs import config as cfg
from dataset.loader import test_loader
from model.attention_model import FocusAttentionModel

CLASS_NAMES = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "tM", "tSB", "tB", "tEB", "tHB"]
CLASSES_NUM = len(CLASS_NAMES)
def main():
    model = FocusAttentionModel()
    model.load_state_dict(torch.load(cfg.TEST_MODEL_DIR, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()
    correct_per_class = torch.zeros(CLASSES_NUM, dtype=torch.int64)
    total_per_class = torch.zeros(CLASSES_NUM, dtype=torch.int64)
    overall_correct = 0
    overall_total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch["images"] = batch["images"].to(cfg.DEVICE)
            batch["label"] = batch["label"].to(cfg.DEVICE)
            labels = batch["label"]
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            overall_correct += (preds == labels).sum().item()
            overall_total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            for gt, pred in zip(labels, preds):
                gt = gt.item()
                pred = pred.item()
                total_per_class[gt] += 1
                if gt == pred:
                    correct_per_class[gt] += 1
    print("=" * 70)
    print("Per-class Accuracy")
    print("=" * 70)
    result = []
    for i in range(CLASSES_NUM):
        if total_per_class[i] == 0:
            acc = 0
        else:
            acc = (correct_per_class[i].item() / total_per_class[i].item() * 100)
        print(
            f"{CLASS_NAMES[i]:5s}"
            f"   {correct_per_class[i]:4d}/{total_per_class[i]:4d}"
            f"   Accuracy = {acc:.2f}%"
        )
        result.append({
            "Stage": CLASS_NAMES[i],
            "Correct": int(correct_per_class[i]),
            "Total": int(total_per_class[i]),
            "Accuracy(%)": round(acc, 2)
        })
    df = pd.DataFrame(result)
    df.to_csv(cfg.SAVE_RESULT_DIR, index=False, encoding="utf-8-sig")

    overall_acc = overall_correct / overall_total * 100
    print("\n")
    print("=" * 70)
    print(f"Overall Accuracy : {overall_acc:.2f}%")
    print("=" * 70)

    print("\nClassification Report\n")
    print(classification_report( all_labels, all_preds, labels=list(range(CLASSES_NUM)), target_names=CLASS_NAMES, digits=4, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(CLASSES_NUM)))
    print("\nConfusion Matrix\n")
    print(cm)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(cfg.SAVE_CM_DIR, encoding="utf-8-sig")

if __name__ == "__main__":
    main()


