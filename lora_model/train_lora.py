import os
import sys
import argparse
import re
import json
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold

# accumulator for ROC data across folds
roc_data = []

# ---------- 路径配置 ----------
MODEL_NAME = "./bert"
TRAIN_FILE = "./data/cleaned/train.txt"  # 只包含ham的训练数据
LORA_SAVE_DIR = "./lora_model/lora-bert-adapter"

# ---------- 日志输出到文件 ----------
LOG_FILE = "./results/train_output.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
sys.stdout = open(LOG_FILE, "w")
sys.stderr = sys.stdout
# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--cv", action="store_true", help="Run 5‑fold cross‑validation")
args, remaining = parser.parse_known_args()
# 把自定义参数移出 sys.argv，避免影响 HF TrainingArguments
sys.argv = [sys.argv[0]] + remaining

# ---------- 1. 加载数据 ----------
# 读取训练数据
train_ds = load_dataset(
    "csv",
    data_files={"train": TRAIN_FILE},
    delimiter="\t",
    column_names=["text", "label"],
    keep_default_na=False,
    quoting=3
)["train"]

# 读取测试数据：spam和ham
TEST_SPAM_FILE = "./data/cleaned/test_spam.txt"
TEST_HAM_FILE = "./data/cleaned/test_ham.txt"

spam_test_ds = load_dataset(
    "csv",
    data_files={"test": TEST_SPAM_FILE},
    delimiter="\t",
    column_names=["text", "label"],
    keep_default_na=False,
    quoting=3
)["test"]
ham_test_ds = load_dataset(
    "csv",
    data_files={"test": TEST_HAM_FILE},
    delimiter="\t",
    column_names=["text", "label"],
    keep_default_na=False,
    quoting=3
)["test"]

test_ds = concatenate_datasets([spam_test_ds, ham_test_ds])

dataset = train_ds

# ---------- 2. Tokenizer ----------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

def clean_text(text):
    """清理文本，保留有意义的内容"""
    # 移除方括号和引号
    text = re.sub(r"[\[\]']", " ", text)
    # 合并多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_fn(examples):
    texts = [clean_text(x) for x in examples["text"]]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=384
    )
    encodings["labels"] = [int(l) for l in examples["label"]]
    return encodings

# 数据预处理
tokenized_ds = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

# 将 tokenized 数据划分为训练集和测试集
train_ds = tokenized_ds

# 对测试集进行tokenize
tokenized_test_ds = test_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=test_ds.column_names,
    desc="Tokenizing test data"
)
eval_ds = tokenized_test_ds

print(f">>> Dataset sizes: {len(train_ds)} train, {len(eval_ds)} eval")

# ---------- 3. Padding collator ----------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# ---------- 若指定 --cv，则执行 5 折交叉验证并提前退出 ----------
if args.cv:

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        auc = roc_auc_score(labels, logits[:, 1])
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": auc}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    texts = dataset["text"]
    labels = dataset["label"]

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"=== Fold {fold_idx}/5 ===")
        train_fold = tokenized_ds.select(tr_idx)
        val_fold   = tokenized_ds.select(val_idx)

        base_model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, local_files_only=True
        )

        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["query", "key", "value", "output.dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )
        model = get_peft_model(base_model, lora_cfg)

        fold_args = TrainingArguments(
            output_dir=f"./results/cv/fold{fold_idx}",
            num_train_epochs=8,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_steps=500,
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            logging_steps=50,
            do_eval=True,
            fp16=True,
            report_to="none",
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=model,
            args=fold_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        # collect ROC data for this fold
        cv_pred = trainer.predict(val_fold)
        logits_fold = cv_pred.predictions
        labels_fold = cv_pred.label_ids
        probs_fold = logits_fold[:, 1]
        fpr, tpr, _ = roc_curve(labels_fold, probs_fold)
        roc_data.append((fpr, tpr))

        adapter_dir = f"./lora_model/cv/fold{fold_idx}"
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    # ---------- 汇总并保存指标 ----------
    os.makedirs("./results/cv", exist_ok=True)
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std_metrics  = {k: float(np.std([m[k] for m in fold_metrics]))  for k in fold_metrics[0]}
    with open("./results/cv/metrics_summary.json", "w") as fout:
        json.dump({"mean": mean_metrics, "std": std_metrics}, fout, indent=2)

    print(">>> 5‑fold CV finished. Mean ± std:")
    for k in mean_metrics:
        print(f"{k}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

# ---------- 4. BERT + LoRA ----------
base_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, local_files_only=True
)

# 配置LoRA
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value", "output.dense"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
)

model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

# ---------- 5. 训练参数 ----------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=8,              # 训练轮数
    per_device_train_batch_size=16,   # 批量大小
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=500,
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    logging_steps=50,
    do_eval=True,
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:,1])
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": auc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print(">>> Start LoRA fine-tuning")
trainer.train()

# ---------- 6. 保存 LoRA 适配器 ----------
os.makedirs(LORA_SAVE_DIR, exist_ok=True)
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)

os.makedirs("./results", exist_ok=True)
# plot combined ROC: CV folds + final
plt.figure()
for i, (fpr_fold, tpr_fold) in enumerate(roc_data, 1):
    plt.plot(fpr_fold, tpr_fold, label=f'Fold {i}')

eval_preds = trainer.predict(eval_ds)
logits_eval = eval_preds.predictions
labels_eval = eval_preds.label_ids
probs = logits_eval[:, 1]
fpr_final, tpr_final, _ = roc_curve(labels_eval, probs)
roc_auc = auc(fpr_final, tpr_final)
plt.plot(fpr_final, tpr_final, label=f'Final AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("./results/roc_all.png")
print(">>> Combined ROC curves saved to ./results/roc_all.png")

sys.stdout.close()

# 绘制并保存训练loss曲线
plt.figure()
training_logs = trainer.state.log_history
steps = [log['step'] for log in training_logs if 'loss' in log]
losses = [log['loss'] for log in training_logs if 'loss' in log]
plt.plot(steps, losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('./results/training_loss_curve.png')
plt.close()