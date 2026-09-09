import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from configs import config as cfg
from dataset.loader import test_loader
from utils.get_model import build_model
from dataset.labels import LABEL_NAMES
# ============================================================
# 类别数量
# ============================================================
CLASSES_NUM = len(LABEL_NAMES)
# ============================================================
# 16类最终概率
# coarse:
#   0 -> pronuclear: tPB2 / tPNa / tPNf
#   1 -> cleavage:   t2 ~ t9+
#   2 -> blastocyst: tM / tSB / tB / tEB / tHB
# final_probs:
#   [3 + 8 + 5] = 16
# ============================================================
def get_final_probs(output):
    # --------------------------------------------------------
    # Coarse probability
    # --------------------------------------------------------
    coarse_probs = output["coarse_probs"]
    # --------------------------------------------------------
    # Fine probability
    # --------------------------------------------------------
    pronuclear_logits = output["pronuclear_logits"]
    cleavage_logits = output["cleavage_logits"]
    blastocyst_logits = output["blastocyst_logits"]
    pronuclear_probs = torch.softmax(pronuclear_logits, dim=1)
    cleavage_probs = torch.softmax(cleavage_logits, dim=1)
    blastocyst_probs = torch.softmax(blastocyst_logits, dim=1)
    # --------------------------------------------------------
    # Hierarchical probability
    # P(fine) = P(coarse) × P(fine | coarse)
    # --------------------------------------------------------
    final_probs = torch.cat(
        [coarse_probs[:, 0:1] * pronuclear_probs, coarse_probs[:, 1:2] * cleavage_probs, coarse_probs[:, 2:3] * blastocyst_probs],
        dim=1
    )
    # --------------------------------------------------------
    # 安全检查
    # --------------------------------------------------------
    assert final_probs.shape[1] == CLASSES_NUM, (
        f"Expected {CLASSES_NUM} classes, " f"but got {final_probs.shape[1]}"
    )
    return final_probs
# ============================================================
# Main
# ============================================================
def main():
    # ========================================================
    # 1. 创建模型
    # ========================================================
    model = build_model(cfg.CURRENT_MODEL)
    # ========================================================
    # 2. 加载训练好的模型
    # ========================================================
    print("=" * 70)
    print("Loading model")
    print("=" * 70)
    checkpoint = torch.load(cfg.TEST_MODEL_DIR, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(cfg.DEVICE)
    model.eval()
    print("Model loaded successfully.")
    # ========================================================
    # 3. 初始化统计变量
    # ========================================================
    correct_per_class = torch.zeros(CLASSES_NUM, dtype=torch.int64)
    total_per_class = torch.zeros(CLASSES_NUM, dtype=torch.int64)
    overall_correct = 0
    overall_total = 0
    all_labels = []
    all_preds = []
    # ========================================================
    # 4. Test
    # ========================================================
    with torch.no_grad():
        for batch in test_loader:
            # ------------------------------------------------
            # 数据移动到 GPU / CPU
            # ------------------------------------------------
            batch["images"] = batch["images"].to(cfg.DEVICE)
            batch["label"] = batch["label"].to(cfg.DEVICE)
            labels = batch["label"]
            # ------------------------------------------------
            # Phase 2 forward
            # 注意：这里不能直接 model(batch)，必须指定stage="fine"，因为最终测试需要三个 fine head。
            # ------------------------------------------------
            output = model(batch, stage="fine", return_dict=True)
            # ------------------------------------------------
            # 得到最终16类概率
            # ------------------------------------------------
            final_probs = get_final_probs(output)
            # ------------------------------------------------
            # 最终16类预测
            # ------------------------------------------------
            preds = torch.argmax(final_probs, dim=1)
            # ------------------------------------------------
            # Overall statistics
            # ------------------------------------------------
            overall_correct += (preds == labels).sum().item()
            overall_total += labels.size(0)
            # ------------------------------------------------
            # 保存全部标签和预测
            # 用于 classification_report / confusion_matrix
            # ------------------------------------------------
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # ------------------------------------------------
            # Per-class statistics
            # ------------------------------------------------
            for gt, pred in zip(labels, preds):
                gt = gt.item()
                pred = pred.item()
                total_per_class[gt] += 1
                if gt == pred:
                    correct_per_class[gt] += 1
    # ========================================================
    # 5. Per-class Accuracy
    # ========================================================
    print("=" * 70)
    print("Per-class Accuracy")
    print("=" * 70)
    result = []
    for i in range(CLASSES_NUM):
        if total_per_class[i] == 0:
            acc = 0.0
        else:
            acc = (correct_per_class[i].item() / total_per_class[i].item() * 100)
        print(
            f"{LABEL_NAMES[i]:5s}"
            f"   {correct_per_class[i]:4d}/{total_per_class[i]:4d}"
            f"   Accuracy = {acc:.2f}%"
        )
        result.append(
            {
                "Stage": LABEL_NAMES[i],
                "Correct": int(correct_per_class[i]),
                "Total": int(total_per_class[i]),
                "Accuracy(%)": round(acc, 2)
            }
        )
    # ========================================================
    # 6. 保存 Per-class Accuracy
    # ========================================================
    df = pd.DataFrame(result)
    df.to_csv(cfg.SAVE_RESULT_DIR, index=False, encoding="utf-8-sig")
    # ========================================================
    # 7. Overall Accuracy
    # ========================================================
    if overall_total == 0:
        overall_acc = 0.0
    else:
        overall_acc = (overall_correct / overall_total * 100)
    print("\n")
    print("=" * 70)
    print(f"Overall Accuracy : {overall_acc:.2f}%")
    print("=" * 70)
    # ========================================================
    # 8. Classification Report
    # ========================================================
    print("\nClassification Report\n")
    # --------------------------------------------------------
    # 文本形式
    # --------------------------------------------------------
    report_text = classification_report(
        all_labels,
        all_preds,
        labels=list(range(CLASSES_NUM)),
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0
    )
    print(report_text)
    # --------------------------------------------------------
    # 字典形式
    # 用于保存 CSV
    # --------------------------------------------------------
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=list(range(CLASSES_NUM)),
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    # ========================================================
    # 9. 保存 Classification Report
    # ========================================================
    rows = []
    # --------------------------------------------------------
    # 每个类别
    # --------------------------------------------------------
    for class_name in LABEL_NAMES:
        if class_name in report_dict:
            metrics = report_dict[class_name]
            rows.append(
                {
                    "class": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"]
                }
            )
    # --------------------------------------------------------
    # macro avg / weighted avg
    # --------------------------------------------------------
    for avg_type in ["macro avg", "weighted avg"]:
        if avg_type in report_dict:
            metrics = report_dict[avg_type]
            rows.append(
                {
                    "class": avg_type,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"]
                }
            )
    # --------------------------------------------------------
    # Overall accuracy
    # 保持原来的保存方式： support 中记录 Overall Accuracy (%)
    # --------------------------------------------------------
    rows.append(
        {
            "class": "accuracy",
            "precision": None,
            "recall": None,
            "f1-score": None,
            "support": round(overall_acc, 2)
        }
    )
    df_report = pd.DataFrame(rows)
    df_report.to_csv(cfg.SAVE_REPORT_DIR, index=False, encoding="utf-8-sig")
    # ========================================================
    # 10. Confusion Matrix
    # ========================================================
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(CLASSES_NUM)))
    print("\nConfusion Matrix\n")
    print(cm)
    # --------------------------------------------------------
    # 保存 Confusion Matrix
    # --------------------------------------------------------
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_df.to_csv(cfg.SAVE_CM_DIR, encoding="utf-8-sig")
    # ========================================================
    # 11. 测试完成
    # ========================================================
    print("\n")
    print("=" * 70)
    print("Testing completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
