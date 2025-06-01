# evaluation_metrics.py
import os
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def evaluate_model(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    auroc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0
    return acc, precision, recall, f1, auroc

def save_metrics_report(results_dir, model_name, acc, precision, recall, f1, auroc):
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "metrics_report.txt")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {model_name} Classification metrics ===\n")
        f.write(f"Accuracy                 : {acc:.4f}\n")
        f.write(f"Precision (spam=1)       : {precision:.4f}\n")
        f.write(f"Recall    (spam=1)       : {recall:.4f}\n")
        f.write(f"F1-score                 : {f1:.4f}\n")
        f.write(f"AUROC                    : {auroc:.4f}\n")
