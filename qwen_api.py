from calendar import c
import os
import json
import yaml
from tqdm import tqdm
from openai import OpenAI
import threading

with open('./config/api.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['api_key']
client = OpenAI(
        api_key=api_key, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

file_path_sciq = os.path.expanduser('/data/lzx/sciq/train.json')
with open(file_path_sciq, 'r') as file:
    data1 = json.load(file)

first_item = data1[0]
# 提取各个字段
question = first_item.get('question')
distractor1 = first_item.get('distractor1')
distractor2 = first_item.get('distractor2')
distractor3 = first_item.get('distractor3')
correct_answer = first_item.get('correct_answer')
support = first_item.get('support')

# 构建格式化的字符串
formatted_item = f"""
question: {question}
choices:
A. {distractor1}
B. {distractor2}
C. {distractor3}
D. {correct_answer}(correct)
support: {support}
"""
prompt = "You are a helpful education assistant. You need to geneate multiple choice question based on the following example: \n" + formatted_item


messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": "PLease generate a multiple choice question based on history."},
]
temperature = 0.75
top_p = 1
presence_penalty = 0.0

# 定义一个函数来获取响应
def get_response():
    global response
    response = client.chat.completions.create(
                model='qwen-max',
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
            )

# 创建一个线程来运行 get_response 函数
response_thread = threading.Thread(target=get_response)
response_thread.start()

# 显示进度条
for _ in tqdm(range(100), desc="Waiting for API response"):
    if not response_thread.is_alive():
        break
    response_thread.join(timeout=0.1)

# 确保线程已经完成
response_thread.join()

print(response.choices[0].message.content)
