import os
# 使用 read_parquet 加载parquet文件
import pandas as pd
import csv
from pandas import read_parquet
from utils import str_to_dict_eedi_df


def read_json_to_df(filepath: str) -> pd.DataFrame:
    df = pd.read_json(filepath)
    df = str_to_dict_eedi_df(df)
    return df


# 构建文件路径
file_path = os.path.expanduser('/data/lzx/sciq/train.json')

df = read_json_to_df(file_path)
# 将 DataFrame 保存到 CSV 文件
output_file = os.path.expanduser('./evaluation/train_output.csv')
df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

print(f"DataFrame 已保存到 {output_file}")

# 构建文件路径
file = os.path.expanduser('./evaluation/train_output.csv')

# 读取 CSV 文件，确保使用 UTF-8 编码
df1 = pd.read_csv(output_file, encoding='utf-8')

# 将 DataFrame 转换为字符串并打印
print(df1.head())
