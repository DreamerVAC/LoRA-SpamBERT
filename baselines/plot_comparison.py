# plot_comparison.py
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_true, proba_dict, save_path):
    """
    proba_dict: dict，如 {'SVM': svm_proba, 'CNN': cnn_proba, 'RNN': rnn_proba}
    """
    plt.figure(figsize=(8, 6))
    for label, y_proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_accuracy_bar(accuracies, save_path):
    """
    accuracies: dict，如 {'SVM': 0.95, 'CNN': 0.97, 'RNN': 0.96}
    """
    plt.figure(figsize=(6, 5))
    labels = list(accuracies.keys())
    values = list(accuracies.values())

    plt.bar(labels, values, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
