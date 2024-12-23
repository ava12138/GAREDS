import os
import json
from pandas import qcut
import yaml
from openai import OpenAI
import re
from PromptFramwork import PromptFramework as pf
from utils import format_question_output, format_rationale_output

with open('./config/api.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['api_key']
client = OpenAI(
        api_key=api_key, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

file_path_sciq = os.path.expanduser('/data/lzx/sciq/train.json')
with open(file_path_sciq, 'r') as file:
    data = json.load(file)

question_examples = [data[0], data[1]]

distractor_principle = [
    'confusing concepts',
    'irrelevant answers',
    'vague memories'
]

qg_prompt = pf.producePrompt("qg", examples=question_examples)

# 设置参数
temperature = 1
top_p = 1
presence_penalty = 0.0

# 定义函数来获取响应
def get_response(prompt):
    response = client.chat.completions.create(
        model='qwen-max',
         messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
    )
    return response.choices[0].message.content


q = get_response(qg_prompt)
print("出的题", q)
questiondata = format_question_output(q)

rg_prompt = pf.producePrompt("rg", questiondata, distractor_principle)
# print(dg_prompt)
r = get_response(rg_prompt)
print("错误推理", r)

example = format_rationale_output(r)
dg_prompt = pf.producePrompt("dg", questiondata, example)
d = get_response(dg_prompt)
print("干扰项", d)
