import json
import os
# 使用 read_parquet 加载parquet文件
import pandas as pd
from pandas import read_parquet


# 构建文件路径
file_path_sciq = os.path.expanduser('/data/lzx/sciq/train.json')
file_path_dream = os.path.expanduser('/data/lzx/dream/train.json')
data = read_parquet("/data/lzx/race/all/train-00000-of-00001.parquet")

# 读取JSON文件
with open(file_path_sciq, 'r') as file:
    data1 = json.load(file)

with open(file_path_dream, 'r') as file:
    data2 = json.load(file)

# 输出第一个项
if data1:
    first_item = data1[0]
    print(first_item)
    print("问题:", first_item.get('question'))
    print("干扰项1:", first_item.get('distractor1'))
    print("干扰项2:", first_item.get('distractor2'))
    print("干扰项3:", first_item.get('distractor3'))
    print("正确答案:", first_item.get('correct_answer'))
else:
    print("文件为空或没有数据")

# 输出第二个项
if data2:
    first_item = data2[0]
    print(first_item)
else:
    print("文件为空或没有数据")

# 获取第一条数据
first_item = data.iloc[0]["article"]

print(first_item)
