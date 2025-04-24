import os
import json
import re
# 使用 read_parquet 加载parquet文件
import pandas as pd
import csv
from pandas import read_parquet
import pro.test as test
from pro.utils.utils import str_to_dict_eedi_df, format_distractor_output, format_question_output, read_test_data

text = """
Question: What process in plants is primarily responsible for the uptake of carbon dioxide and release of oxygen, converting light energy into chemical energy?
Answer: photosynthesis
Distractor1: **Respiration**
Feedback: Respiration is not the correct answer because, in plants, it is primarily a process that consumes oxygen and releases carbon dioxide to generate energy, which is the opposite of what you are looking for. The process you are thinking of, which involves taking in carbon dioxide and releasing oxygen, is photosynthesis.

Distractor2: **Fermentation**
Feedback: Fermentation is not the correct answer because it is an anaerobic process that occurs in the absence of oxygen and does not involve the uptake of carbon dioxide or the release of oxygen. The process you need is photosynthesis, which uses light energy to convert carbon dioxide and water into glucose and oxygen.

Distractor3: **Transpiration**
Feedback: Transpiration is not the correct answer because it is the process by which water is carried through plants from roots to the leaves, where it changes to vapor and is released to the atmosphere. The process responsible for the uptake of carbon dioxide and release of oxygen, and the conversion of light energy into chemical energy, is photosynthesis.

 **Distractor1:**  
*Feedback:* This is not correct because it refers to the process by which water moves through the environment, but it does not describe the conversion of water vapor into liquid water.  
*Distractor:* **Evaporation**

**Distractor2:**  
*Feedback:* Although this process involves the transport of water through plants, it does not describe the conversion of water vapor into liquid water.  
*Distractor:* **Transpiration**

**Distractor3:**  
*Feedback:* This process involves the movement of water underground but does not relate to the conversion of water vapor into liquid water.  
*Distractor:* **Infiltration**
"""

# 提取干扰项
question_data = format_question_output(text)
extracted_distractors = format_distractor_output(text)
print(question_data)
print(extracted_distractors)
# 合并题目信息和干扰项
combined_data = {**question_data, **extracted_distractors}
 
# 保存为JSON文件
output_filepath = './evaluation/output_dg.json'
with open(output_filepath, 'w') as json_file:
    json.dump(combined_data, json_file, indent=4)

print(f"所有干扰项已保存到 {output_filepath}")
test_file = "./evaluation/test.json"
data = read_test_data(test_file)
print(data)

def read_json_to_df(filepath: str) -> pd.DataFrame:
    df = pd.read_json(filepath)
    df = str_to_dict_eedi_df(df)
    return df


# # 构建文件路径
# file_path = os.path.expanduser('/data/lzx/sciq/train.json')

# df = read_json_to_df(file_path)
# # 将 DataFrame 保存到 CSV 文件
# output_file = os.path.expanduser('./evaluation/train_output.csv')
# df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

# print(f"DataFrame 已保存到 {output_file}")

# # 构建文件路径
# file = os.path.expanduser('./evaluation/train_output.csv')

# # 读取 CSV 文件，确保使用 UTF-8 编码
# df1 = pd.read_csv(output_file, encoding='utf-8')

# # 将 DataFrame 转换为字符串并打印
# print(df1.head())
