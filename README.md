# LoRA垃圾邮件检测系统

本项目使用LoRA (Low-Rank Adaptation)技术对预训练BERT模型进行微调，实现高效的垃圾邮件检测系统。

## 模型性能

在Enron-Spam测试集上，BERT模型的表现如下：

| 指标 | 性能 |
|------|------|
| Accuracy | 97.30% |
| Precision | 98.66% |
| Recall | 95.96% |
| F1-score | 97.29% |
| AUROC | 99.74% |

## 项目结构

```
.
├── lora_model/               # LoRA模型相关代码
│   ├── train_lora.py        # LoRA训练主程序
│   ├── eval_classifier.py    # 模型评估程序
│   ├── visualize_attention.py# 注意力可视化工具
│   └── cv/                  # 交叉验证相关代码
├── preprocessing/            # 数据预处理代码
│   └── email_cleaner.py     # 邮件文本清洗工具
├── data/                    # 数据目录
├── bert/                    # BERT模型相关代码（需要从ModelScope下载）
├── baselines/              # 基准模型实现
├── utils/                  # 工具函数
└── results/                # 实验结果保存目录
```

## 环境配置

本项目需要Python 3.12及以上版本。使用conda创建虚拟环境：

```bash
# 创建conda环境
conda create --name spam_detection python=3.12
conda activate spam_detection

# 安装依赖
pip install -r requirements.txt
```

## 模型与数据准备

### BERT模型下载
本项目使用的BERT模型来自ModelScope，在运行代码前需要先下载模型在./bert文件夹中：

命令行下载：
```
pip install modelscope
modelscope download --model google-bert/bert-base-uncased
```

git 下载：
```
git lfs install
git clone https://www.modelscope.cn/google-bert/bert-base-uncased.git
```

### 数据集
本项目使用Enron-Spam数据集。该数据集包含来自Enron邮件集合的垃圾邮件和正常邮件。数据集应放置在`data/`目录下。

**注意**: Enron数据集来自公开的Enron电子邮件，这些邮件是在Enron公司调查期间被美国联邦能源监管委员会公开的。使用该数据集时请遵守相关的使用条款和隐私规定。

## 运行步骤

1. 数据预处理：
```bash
python preprocessing/email_cleaner.py
```

2. 训练LoRA模型：
```bash
python lora_model/train_lora.py
```

3. 评估模型性能：
```bash
python lora_model/eval_classifier.py
```

4. 可视化注意力权重（可选）：
```bash
python lora_model/visualize_attention.py
```

结果保存在./results文件夹中

## 主要特性

- 使用LoRA技术进行高效模型微调
- 支持多种数据预处理方式
- 提供详细的模型评估指标
- 包含注意力机制可视化工具
- 实现多个基准模型用于对比