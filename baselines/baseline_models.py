# 导入所需库
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense
from tensorflow.keras.utils import to_categorical

from utils.tokenizer_utils import load_tokenized_data

# 设置最大序列长度和词向量维度
MAX_LEN = 100
EMBED_DIM = 100

def prepare_data_for_dl(X_tokens, y):
    """
    将 token 化后的文本和标签处理为深度学习模型可用的格式：
    - 文本被编码为整数序列
    - 序列被 padding 到统一长度
    - 标签被 one-hot 编码
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tokens)                         # 建立词汇表
    X_seq = tokenizer.texts_to_sequences(X_tokens)           # 文本转为整数序列
    X_pad = pad_sequences(X_seq, maxlen=MAX_LEN)             # 序列补齐到最大长度
    y_cat = to_categorical(y)                                # 标签 one-hot 编码
    return X_pad, y_cat, tokenizer                           # 返回编码后的数据和 tokenizer

def build_cnn_model(vocab_size):
    """
    构建一个简单的 CNN 模型用于文本分类：
    - 包含嵌入层、卷积层、全局最大池化层和输出层
    """
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),  # 嵌入层
        Conv1D(128, 5, activation='relu'),                        # 一维卷积层
        GlobalMaxPooling1D(),                                     # 全局最大池化
        Dense(2, activation='softmax')                            # 输出层：2分类 softmax
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型
    return model

def build_rnn_model(vocab_size):
    """
    构建一个简单的 RNN 模型（使用 LSTM）用于文本分类：
    - 包含嵌入层、LSTM 层和输出层
    """
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),  # 嵌入层
        LSTM(64),                                                 # LSTM 层
        Dense(2, activation='softmax')                            # 输出层：2分类 softmax
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型
    return model

def load_split_test_data(ham_path, spam_path):
    """
    加载并合并两个测试集（正常邮件和垃圾邮件），返回合并后的文本和标签列表。
    """
    X_ham, y_ham = load_tokenized_data(ham_path)     # 加载正常邮件数据
    X_spam, y_spam = load_tokenized_data(spam_path)  # 加载垃圾邮件数据
    return X_ham + X_spam, y_ham + y_spam            # 合并数据

def train_and_evaluate_models(train_data, test_data):
    """
    训练 SVM、CNN、RNN 三个模型并对测试集进行预测，保存预测结果，返回所有预测值和标签。

    参数：
    - train_path: 训练集路径
    - ham_path: 正常邮件测试集路径
    - spam_path: 垃圾邮件测试集路径

    返回：
    - y_test: 测试集真实标签
    - svm_preds, cnn_preds, rnn_preds: 三种模型的预测类别
    - svm_proba, cnn_proba, rnn_proba: 三种模型预测为 spam 的概率
    """
    # 1. 加载数据
    X_train, y_train = train_data
    X_test, y_test = test_data

    # 2. SVM + TF-IDF + 概率输出
    svm_pipeline = make_pipeline(
        TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False),  # 使用 TF-IDF 向量化（跳过预处理）
        SGDClassifier(loss='log_loss', max_iter=1000)  # 使用逻辑回归损失支持 predict_proba
    )
    svm_pipeline.fit(X_train, y_train)                     # 训练 SVM 模型
    svm_preds = svm_pipeline.predict(X_test)               # 分类预测
    svm_proba = svm_pipeline.predict_proba(X_test)[:, 1]   # “spam” 类别的预测概率

    # 3. CNN + RNN 深度学习模型
    X_train_dl, y_train_cat, tokenizer = prepare_data_for_dl(X_train, y_train)   # 准备深度学习输入数据
    X_test_dl = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)  # 测试集 padding
    vocab_size = len(tokenizer.word_index) + 1            # 词汇表大小

    cnn = build_cnn_model(vocab_size)                     # 构建 CNN 模型
    rnn = build_rnn_model(vocab_size)                     # 构建 RNN 模型

    cnn.fit(X_train_dl, y_train_cat, epochs=3, batch_size=32, verbose=0)  # 训练 CNN
    rnn.fit(X_train_dl, y_train_cat, epochs=3, batch_size=32, verbose=0)  # 训练 RNN

    cnn_proba_all = cnn.predict(X_test_dl)                # CNN 输出所有类别概率
    cnn_preds = np.argmax(cnn_proba_all, axis=1)          # CNN 分类结果
    cnn_proba = cnn_proba_all[:, 1]                       # CNN “spam” 概率

    rnn_proba_all = rnn.predict(X_test_dl)                # RNN 输出所有类别概率
    rnn_preds = np.argmax(rnn_proba_all, axis=1)          # RNN 分类结果
    rnn_proba = rnn_proba_all[:, 1]                       # RNN “spam” 概率

    # 返回所有模型的预测信息和真实标签
    return (
        y_test,
        svm_preds, svm_proba,
        cnn_preds, cnn_proba,
        rnn_preds, rnn_proba,
    )
