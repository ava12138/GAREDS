import nltk
import os
from pathlib import Path

def download_nltk_data():
    """下载所需的 NLTK 数据"""
    required_data = ['wordnet', 'punkt', 'omw-1.4']
    
    print("开始下载 NLTK 数据...")
    for data in required_data:
        try:
            print(f"下载 {data}...")
            nltk.download(data, quiet=True)
            print(f"✓ {data} 下载完成")
        except Exception as e:
            print(f"✗ {data} 下载失败: {e}")
    
    print("\n数据下载完成!")
    print(f"数据存储位置: {os.path.expanduser('~/nltk_data')}")

if __name__ == "__main__":
    download_nltk_data()