import os
import numpy as np
from sklearn.model_selection import KFold

from baselines.baseline_models import train_and_evaluate_models
from baselines.evaluation_metrics import evaluate_model, save_metrics_report
from baselines.plot_comparison import plot_roc_curve, plot_accuracy_bar
from utils.tokenizer_utils import load_tokenized_data

# 数据路径（可根据实际情况修改）
train_path = "data/cleaned/train.txt"
results_dir = "results"
plots_dir = os.path.join(results_dir, "plots")

# 确保输出文件夹存在
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# 清空旧的 metrics_report.txt（如存在）
report_path = os.path.join(results_dir, "metrics_report.txt")
if os.path.exists(report_path):
    os.remove(report_path)

# 加载完整训练数据
X_all, y_all = load_tokenized_data(train_path)
X_all = np.array(X_all, dtype=object)
y_all = np.array(y_all)

# 初始化 KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每一折的评估指标
accuracy_scores = {"SVM": [], "CNN": [], "RNN": []}
precision_scores = {"SVM": [], "CNN": [], "RNN": []}
recall_scores = {"SVM": [], "CNN": [], "RNN": []}
f1_scores = {"SVM": [], "CNN": [], "RNN": []}
auc_scores = {"SVM": [], "CNN": [], "RNN": []}
proba_agg = {"SVM": [], "CNN": [], "RNN": []}
label_agg = []

fold = 1
for train_index, val_index in kf.split(X_all):
    print(f"Fold {fold} training...")
    fold += 1

    X_train_fold = X_all[train_index].tolist()
    y_train_fold = y_all[train_index].tolist()

    X_test_fold = X_all[val_index].tolist()
    y_test_fold = y_all[val_index].tolist()

    (
        y_test,
        svm_preds, svm_proba,
        cnn_preds, cnn_proba,
        rnn_preds, rnn_proba,
    ) = train_and_evaluate_models(
        (X_train_fold, y_train_fold),
        (X_test_fold, y_test_fold)
    )

    for model_name, (preds, proba) in {
        "SVM": (svm_preds, svm_proba),
        "CNN": (cnn_preds, cnn_proba),
        "RNN": (rnn_preds, rnn_proba)
    }.items():
        acc, prec, rec, f1, auroc = evaluate_model(y_test, preds, proba)
        accuracy_scores[model_name].append(acc)
        precision_scores[model_name].append(prec)
        recall_scores[model_name].append(rec)
        f1_scores[model_name].append(f1)
        auc_scores[model_name].append(auroc)
        proba_agg[model_name].extend(proba)
    
    label_agg.extend(y_test)

# 求平均准确率并保存报告
accuracy_dict = {}
for model_name in ["SVM", "CNN", "RNN"]:
    avg_acc = np.mean(accuracy_scores[model_name])
    avg_prec = np.mean(precision_scores[model_name])
    avg_rec = np.mean(recall_scores[model_name])
    avg_f1 = np.mean(f1_scores[model_name])
    avg_auc = np.mean(auc_scores[model_name])

    accuracy_dict[model_name] = avg_acc

    save_metrics_report(
        results_dir,
        model_name,
        avg_acc,
        avg_prec,
        avg_rec,
        avg_f1,
        avg_auc
    )

# 绘制 ROC 曲线
plot_roc_curve(label_agg, proba_agg, plots_dir)

# 绘制准确率柱状图
plot_accuracy_bar(accuracy_dict, plots_dir)

print("5折交叉验证完成，结果已保存至 results/")
