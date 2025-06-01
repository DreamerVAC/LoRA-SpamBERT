# visualize_attention.py
# ---------------------
import torch, re, os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# import plotly.express as px  # no longer used after refactor
import plotly.graph_objects as go
from collections import Counter, defaultdict
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

# ----- global figure style for publication -----
import matplotlib.font_manager as fm
_FONTS = {f.name for f in fm.fontManager.ttflist}
BASE_FONT = "Times New Roman" if "Times New Roman" in _FONTS else "DejaVu Serif"

plt.rcParams.update({
    "font.family": BASE_FONT,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
})

print(f"✔ Using font: {BASE_FONT}")

BASE_DIR    = "./bert"                       # 你的 BERT 基座
ADAPTER_DIR = "./lora_model/lora-bert-adapter"
HAM_FILE    = "./data/cleaned/test_ham.txt"
SPAM_FILE   = "./data/cleaned/test_spam.txt"
N_PER_CLASS = 100                             # 每类取几封

device = "cuda" if torch.cuda.is_available() else "cpu"

print("→ loading model ...")
base  = BertForSequenceClassification.from_pretrained(BASE_DIR,
                                                      num_labels=2,
                                                      output_attentions=True)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval().to(device)
tok   = BertTokenizer.from_pretrained(BASE_DIR)

# ---------- 读取样本 ----------
def pick(path, n):
    """
    读取前 n 行样本。
    若行末含 “\t0” 或 “\t1” 标签，自动去除，只保留正文。
    """
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            # 检测并移除末尾标签
            parts = raw.rsplit("\t", 1)
            if len(parts) == 2 and parts[1] in {"0", "1"}:
                txt = parts[0]
            else:
                txt = raw
            buf.append(txt)
            if len(buf) >= n:
                break
    return buf

samples = [(t, 0) for t in pick(HAM_FILE, N_PER_CLASS)] + \
          [(t, 1) for t in pick(SPAM_FILE, N_PER_CLASS)]

# ---------- 统计 token 注意力 ----------
tot_attn = defaultdict(float)    # token_id → 累积注意力

for text, _ in tqdm(samples, desc="forward"):
    enc = tok(text, return_tensors="pt", truncation=True,
              padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        out = model(**enc)

    # out.attentions: list[L] each -> [B, H, S, S]
    # 先 stack → [L, B, H, S, S]
    attn = torch.stack(out.attentions)               # [L, B, H, S, S]
    # 对 “层 + 头” 两个维度整体平均，得到 [B, S, S]
    layer_head_avg = attn.mean(dim=(0, 2))
    # 取 batch=0，CLS 行 (索引 0) → [S]
    attn_vec = layer_head_avg[0, 0]

    # 加到全局计数器
    for idx, score in enumerate(attn_vec):
        tok_id = int(enc.input_ids[0, idx])
        if tok_id not in (tok.cls_token_id, tok.pad_token_id, tok.sep_token_id):
            tot_attn[tok_id] += score.item()

# ---------- 重新统计：分别计算 ham 与 spam 的注意力 ----------
tot_attn_ham  = defaultdict(float)
tot_attn_spam = defaultdict(float)

for text, lbl in tqdm(samples, desc="forward (split)"):
    enc = tok(text, return_tensors="pt", truncation=True,
              padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        attn = torch.stack(model(**enc).attentions)  # [L,B,H,S,S]
        layer_head_avg = attn.mean(dim=(0, 2))[0, 0]  # [S]

    for idx, score in enumerate(layer_head_avg):
        tok_id = int(enc.input_ids[0, idx])
        if tok_id in (tok.cls_token_id, tok.pad_token_id, tok.sep_token_id):
            continue
        if lbl == 0:
            tot_attn_ham[tok_id]  += score.item()
        else:
            tot_attn_spam[tok_id] += score.item()

# 取总注意力前 K，并判断来源
K = 30
all_scores = defaultdict(float)
for d in (tot_attn_ham, tot_attn_spam):
    for k, v in d.items():
        all_scores[k] += v

top = Counter(all_scores).most_common(K)
words  = [tok.convert_ids_to_tokens(i) for i, _ in top]
ham_scores  = [tot_attn_ham.get(i, 0.0) for i, _ in top]
spam_scores = [tot_attn_spam.get(i, 0.0) for i, _ in top]
# Backup original top-K tokens and their scores for subsequent charts
orig_words = words.copy()
orig_ham_scores = ham_scores.copy()
orig_spam_scores = spam_scores.copy()


# ---------- 极坐标径向条形图 (Radial Bar Chart) ----------
import numpy as np
# Prepare data
# N = len(words)
# angles = np.linspace(0.0, 2*np.pi, N, endpoint=False)
# width = 2*np.pi / N * 0.8

total_scores = np.array([h + s for h, s in zip(ham_scores, spam_scores)])
ham_arr       = np.array(ham_scores)
spam_arr      = np.array(spam_scores)
diff_arr = np.abs(spam_arr - ham_arr)
# Purple for spam-dominant, teal for ham-dominant
diff_colors = [
    '#BEB8DC' if spam_arr[i] > ham_arr[i] else '#8ECFC9'
    for i in range(len(words))
]

# Filter tokens with diff > 0.05
mask = diff_arr > 0.05
words = [w for w, m in zip(words, mask) if m]
diff_arr = diff_arr[mask]
ham_arr = ham_arr[mask]
spam_arr = spam_arr[mask]
diff_colors = [c for c, m in zip(diff_colors, mask) if m]

# Recompute positions after filtering
N = len(words)
angles = np.linspace(0.0, 2*np.pi, N, endpoint=False)
width = 2*np.pi / N * 0.8

# Build polar plot
fig = plt.figure(figsize=(8,8))
from matplotlib.patches import Patch

ax  = fig.add_subplot(111, polar=True)

# Inner ring: absolute difference between spam and ham
ax.bar(
    angles, diff_arr, width=width, bottom=0.0,
    color=diff_colors, edgecolor='white',
    linewidth=1, alpha=0.7, label='Attention Difference'
)

# Middle ring: ham attention
ax.bar(angles, ham_arr, width=width, bottom=diff_arr,
       color='#82B0D2', edgecolor='white', linewidth=1, alpha=0.8,
       label='Ham Attention')

# Outer ring: spam attention
ax.bar(angles, spam_arr, width=width, bottom=diff_arr+ham_arr,
       color='#FFBE7A', edgecolor='white', linewidth=1, alpha=0.8,
       label='Spam Attention')

# Add token labels by stacking each character along the radius
margin = (diff_arr + ham_arr + spam_arr).max() * 0.05
char_spacing = margin * 0.6  # reduced space between characters
for angle, token, d, h, s in zip(angles, words, diff_arr, ham_arr, spam_arr):
    if np.pi < angle <= 2 * np.pi:
        token = token[::-1]  # 左半部分字符顺序反转
    bottom = d + h + s
    radius0 = bottom + margin
    for i, char in enumerate(token):
        r = radius0 + i * char_spacing
        ax.text(
            angle, r, char,
            rotation=0,
            ha='center', va='center', fontsize=8
        )

# Style adjustments
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_axis_off()
legend_handles = [
    Patch(facecolor='#8ECFC9', edgecolor='white', label='Diff (Ham>Spam)'),
    Patch(facecolor='#BEB8DC', edgecolor='white', label='Diff (Spam>Ham)'),
    Patch(facecolor='#82B0D2', edgecolor='white', label='Ham Attention'),
    Patch(facecolor='#FFBE7A', edgecolor='white', label='Spam Attention'),
]
ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.tight_layout()

# Save figure
os.makedirs("results", exist_ok=True)
plt.savefig("results/token_attention_radial.png", dpi=300, bbox_inches="tight")
print("✔ Radial bar chart saved → results/token_attention_radial.png")

# ---------- 单封垃圾邮件示例 ----------
sample_spam = (
    "Congratulations! You have been selected for an exclusive gas nomination deal. "
    "Click http://super‑deal.example.com to claim your meter discount now."
)
enc = tok(sample_spam, return_tensors="pt", truncation=True,
          padding="max_length", max_length=64).to(device)
with torch.no_grad():
    attn = torch.stack(model(**enc).attentions)  # [L,B,H,S,S]
    layer_head_avg = attn.mean(dim=(0, 2))[0]    # [S,S]

# ---------- 重新计算 CDF（剔除 CLS / SEP） ----------
# 计算有效 token 的掩码（去除 [CLS]、[SEP] 与 [PAD]）
input_ids = enc.input_ids[0]                              # [S]
mask = (input_ids != tok.pad_token_id) & \
       (input_ids != tok.cls_token_id) & \
       (input_ids != tok.sep_token_id)

# 提取有效 token id 与词形
valid_ids = input_ids[mask]                               # [N]
seq_len   = valid_ids.size(0)                             # 有效 token 数
tokens    = tok.convert_ids_to_tokens(valid_ids.tolist()) # 对应词形

# 取 CLS→token 注意力并仅保留有效 token
cls_vec = layer_head_avg[0, mask].cpu().numpy()           # [N]

cdf = np.cumsum(cls_vec) / np.sum(cls_vec)

plt.figure(figsize=(max(6, 0.4 * len(tokens)), 3))  # 缩短x轴token间隔
plt.step(range(1, len(tokens) + 1), cdf, where="mid", linewidth=2, color='#3B76AF')
plt.xticks(range(1, len(tokens) + 1), tokens, rotation=90, fontsize=8)
plt.xlabel("Tokens (original order)", fontsize=10, fontweight='bold')
plt.ylabel("Cumulative Attention", fontsize=10, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.savefig("results/sample_spam_cdf.png", dpi=300, bbox_inches="tight")

print("✔ Figures saved → wordcloud.png / sankey.png / sunburst.png / sample_spam_cdf.png")

# ---------- Combined Token Influence (Mirrored Lollipop Chart) ----------
# Prepare DataFrame of token influences
df = pd.DataFrame({
    "token": orig_words,
    "ham_score": orig_ham_scores,
    "spam_score": orig_spam_scores
})
# Keep tokens with score >0.05 in either class
df = df[(df["ham_score"] > 0.05) | (df["spam_score"] > 0.05)]
# Select all tokens above threshold, sorted by descending score
top_ham  = df[df["ham_score"] > 0.05].sort_values("ham_score", ascending=False)
top_spam = df[df["spam_score"] > 0.05].sort_values("spam_score", ascending=False)

# Dynamic figure height based on number of tokens
n_ham = len(top_ham)
n_spam = len(top_spam)
max_rows = max(n_ham, n_spam)
row_height = 0.3  # inches per row
fig_height = max(4, max_rows * row_height)
fig, ax = plt.subplots(figsize=(6, fig_height))
# Precompute horizontal limits and label offset
max_h = top_ham["ham_score"].max() if not top_ham.empty else 0
max_s = top_spam["spam_score"].max() if not top_spam.empty else 0
x_left, x_right = -max_s * 1.1, max_h * 1.1
ax.set_xlim(x_left, x_right)
x_offset = (x_right - x_left) * 0.01
# Adjust margins to avoid label clipping
fig.subplots_adjust(left=0.2, right=0.8)

# Draw ham bars to the right (positive)
for i, (tok_w, score) in enumerate(zip(top_ham["token"], top_ham["ham_score"])):
    y = len(top_ham) - 1 - i  # positions from N_ham-1 down to 0
    ax.hlines(y, 0, score, color="#82B0D2", alpha=0.8, linewidth=1)
    ax.scatter(score, y, color="#82B0D2", s=40)
    ax.text(x_right + x_offset, y, tok_w, va="center", ha="left", fontsize=8, clip_on=False)

# Draw spam bars to the left (negative)
for i, (tok_w, score) in enumerate(zip(top_spam["token"], top_spam["spam_score"])):
    y = -i - 1     # positions -1,-2,-3
    ax.hlines(y, 0, -score, color="#FFBE7A", alpha=0.8, linewidth=1)
    ax.scatter(-score, y, color="#FFBE7A", s=40)
    ax.text(x_left - x_offset, y, tok_w, va="center", ha="right", fontsize=8, clip_on=False)

# Central axis
ax.axvline(0, color="black", linewidth=1)

# Clean up axes
ax.set_yticks([])
ax.set_xlabel("Influence Magnitude")
plt.tight_layout()

# Save result
ax.invert_yaxis()
os.makedirs("results", exist_ok=True)
plt.savefig("results/token_influence_combined.png", dpi=300, bbox_inches="tight")
print("✔ Saved → results/token_influence_combined.png")
