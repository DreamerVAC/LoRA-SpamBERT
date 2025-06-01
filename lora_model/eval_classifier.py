# lora_model/eval_classifier.py
import os, torch
import sys
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import re


# ---------- 工具函数 ----------
def load_texts(path):
    """
    逐行读取文本；若行末是 “\\t0” 或 “\\t1”，自动去除标签，只保留正文。
    跳过空行并返回文本列表。
    """
    texts = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.rsplit("\t", 1)
            if len(parts) == 2 and parts[1] in {"0", "1"}:
                texts.append(parts[0])
            else:
                texts.append(raw)
    return texts

parser = argparse.ArgumentParser(description="Evaluate spam classification metrics.")
parser.add_argument("--no_adapter", action="store_true",
                    help="If set, do NOT load the LoRA adapter (baseline model).")
parser.add_argument("--model_dir", default="./bert",
                    help="Local directory of pretrained BERT files.")
parser.add_argument("--offline", action="store_true",
                    help="Force transformers to work in offline mode (local files only).")
args, extra = parser.parse_known_args()

MODEL_PATH = args.model_dir if args.model_dir else "bert-base-uncased"
local_only = args.offline or bool(args.model_dir)

# ---------- 日志重定向 ----------
LOG_FILE = "./results/eval_output.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
sys.stdout = open(LOG_FILE, "w")
sys.stderr = sys.stdout

# ---------- 路径 ----------
MODEL_NAME = "bert-base-uncased"
ADAPTER_DIR = "./lora_model/lora-bert-adapter"
# paths to test ham/spam files
HAM_FILE   = "./data/cleaned/test_ham.txt"
SPAM_FILE  = "./data/cleaned/test_spam.txt"

# ---------- 加载模型 ----------
print(">>> Loading base model")
base = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2, local_files_only=local_only)
if args.no_adapter:
    model = base
    print(">>> NO adapter loaded (baseline).")
else:
    print(">>> Loading LoRA adapter")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=local_only)

# ---------- 预测单句 ----------
def predict_spam_prob(sentence: str) -> float:
    enc = tokenizer(sentence,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=128)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                       attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        spam_prob = probs[0][1].item()  # 第二个类别的概率（spam=1）
    return spam_prob

# ---------- 评估数据集 ----------

# ---------- 加载并评估 ----------
records = []
labels = []
predictions = []

ham_texts  = load_texts(HAM_FILE)
spam_texts = load_texts(SPAM_FILE)
print(f">>> Loaded ham={len(ham_texts)} spam={len(spam_texts)}")

print(">>> Evaluating test set")
for text in tqdm(ham_texts, desc="Evaluating ham"):
    spam_prob = predict_spam_prob(text)
    records.append(["ham", spam_prob, text])
    labels.append(0)
    predictions.append(spam_prob)

for text in tqdm(spam_texts, desc="Evaluating spam"):
    spam_prob = predict_spam_prob(text)
    records.append(["spam", spam_prob, text])
    labels.append(1)
    predictions.append(spam_prob)

# ---------- 计算分类指标 ----------
y_true = np.array(labels)
y_scores = np.array(predictions)
y_pred = (y_scores > 0.5).astype(int)  # 使用0.5作为分类阈值

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
auc = roc_auc_score(y_true, y_scores)

print(f"=== Classification metrics ===")
print(f"Accuracy            : {acc:.4f}")
print(f"Precision (spam=1)  : {prec:.4f}")
print(f"Recall    (spam=1)  : {rec:.4f}")
print(f"F1-score            : {f1:.4f}")
print(f"AUROC               : {auc:.4f}")

# save metrics
metrics_path = "./results/accuracy_report.txt"
with open(metrics_path, "w") as mf:
    mf.write(f"baseline\t{args.no_adapter}\n")
    mf.write(f"accuracy\t{acc}\n")
    mf.write(f"precision\t{prec}\n")
    mf.write(f"recall\t{rec}\n")
    mf.write(f"f1\t{f1}\n")
    mf.write(f"auroc\t{auc}\n")
print(f">>> Metrics saved to {metrics_path}")

sys.stdout.close()