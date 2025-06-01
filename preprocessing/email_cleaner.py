import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# 确保已经下载了punkt和stopwords
nltk.download('punkt')
nltk.download('stopwords')


class EnronSpamProcessor:
    def __init__(self, data_path, output_dir):
        """初始化并加载数据集"""
        self.data = pd.read_csv(data_path)
        self.stop_words = set(stopwords.words('english'))
        self.output_dir = output_dir
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, 'cleaned'), exist_ok=True)

    def remove_stopwords(self, text):
        """移除停用词"""
        words = nltk.word_tokenize(text)
        return ' '.join([word for word in words if word not in self.stop_words])

    def clean_text(self, text):
        """文本清洗，增加对空值的处理"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # 移除HTML标签
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # 移除URL
        text = re.sub(r'\S+@\S+', '', text)  # 移除邮箱
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # 保留字母和空格
        text = ' '.join(text.split())  # 移除多余空格
        return text

    def preprocess_data(self):
        """数据预处理"""
        print("开始数据预处理...")
        self.data['Message'] = self.data['Message'].fillna('')
        self.data['clean_text'] = self.data['Message'].apply(self.clean_text)
        self.data['clean_text'] = self.data['clean_text'].apply(self.remove_stopwords)
        self.data = self.data[self.data['clean_text'] != '']
        # 转换标签为0/1格式
        self.data['label'] = self.data['Spam/Ham'].map({'ham': 0, 'spam': 1})
        print(f"预处理完成 - 总数据量: {len(self.data)}")
        print(f"正常邮件(ham/0): {len(self.data[self.data['label'] == 0])}")
        print(f"垃圾邮件(spam/1): {len(self.data[self.data['label'] == 1])}")

    def save_data(self, data, filename):
        """保存数据到文件（保持原始格式）"""
        file_path = os.path.join(self.output_dir, 'cleaned', filename)
        data.to_csv(file_path, sep='\t', index=False, header=False)
        print(f"{filename} 已保存，条目数量: {len(data)}")

    def split_and_save_data(self):
        """分割数据并保存到文件（保持原始格式）"""
        # 分离ham和spam数据
        ham_data = self.data[self.data['label'] == 0].copy()
        spam_data = self.data[self.data['label'] == 1].copy()

        # 划分训练集和测试集 (仅使用ham作为训练集)
        train_ham = ham_data.sample(frac=0.8, random_state=42)
        test_ham = ham_data.drop(train_ham.index)

        # 准备保存的数据格式（使用clean_text和label列）
        train_data = train_ham[['clean_text', 'label']]
        test_ham_data = test_ham[['clean_text', 'label']]
        test_spam_data = spam_data[['clean_text', 'label']]

        # 保存训练集 (全部是ham/0)
        self.save_data(train_data, 'train.txt')

        # 保存测试集 (分为ham/0和spam/1两个文件)
        self.save_data(test_ham_data, 'test_ham.txt')
        self.save_data(test_spam_data, 'test_spam.txt')

    def process_pipeline(self):
        """完整处理流程"""
        self.preprocess_data()
        self.split_and_save_data()


# 使用示例
if __name__ == "__main__":
    output_dir = "D:/python/LoRASpamDetection/data"
    processor = EnronSpamProcessor("D:/python/LoRASpamDetection/data/enron_spam_data.csv", output_dir)
    processor.process_pipeline()