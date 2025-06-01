# tokenizer_utils.py

import os

def load_tokenized_data(file_path):
    """
    从给定文件中加载数据，每行包含一条记录：<空格分隔的文本>\t<label>
    返回: tokenized_texts: List[List[str]]，labels: List[int]
    """
    tokenized_texts = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                # 拆分文本与标签
                text_str, label_str = line.rsplit('\t', 1)
                tokens = text_str.strip().split()
                label = int(label_str.strip())
                
                tokenized_texts.append(tokens)
                labels.append(label)
            except ValueError:
                print(f"[Warning] Line {line_num} 格式错误，跳过: {line}")
    
    return tokenized_texts, labels

def ensure_dir_exists(file_path):
    """
    如果目标文件路径中的目录不存在，则创建它。
    """
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)