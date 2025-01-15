import re
import nltk
from nltk.tokenize import word_tokenize
import os

def download_nltk_resources():
    """下载NLTK资源"""
    try:
        # 检查punkt是否已下载
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

class EnhancedTokenizer:
    """增强的分词器类"""
    def __init__(self):
        self.punct_regex = re.compile(r'[^\w\s]')
        download_nltk_resources()  # 在初始化时检查并下载
    
    def preprocess(self, text):
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        # 移除标点符号
        text = self.punct_regex.sub(' ', text)
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text):
        """分词"""
        # 预处理
        text = self.preprocess(text)
        # 使用nltk分词
        return word_tokenize(text)