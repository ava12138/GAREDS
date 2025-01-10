import os
import json
from pandas import qcut
import yaml
from openai import OpenAI
import re
from PromptFramwork import PromptFramework as pf
from utils import format_question_output, format_rationale_output, format_distractor_output

with open('./config/api.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config['api_key']
api_model = config['model']
client = OpenAI(
        api_key=api_key, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

file_path_sciq = os.path.expanduser('/data/lzx/sciq/train.json')
with open(file_path_sciq, 'r') as file:
    data = json.load(file)

question_examples = [data[0], data[1]]

distractor_principle = [
    'Confusing similar concepts: This principle leverages the use of seemingly related but actually incorrect information to test the examinee\'s ability to carefully analyze details and distinguish between similar concepts, ensuring their memory and comprehension are accurate.',
    'Answering irrelevant questions: This principle introduces distractors containing information that is irrelevant to the question stem but appears plausible on the surface, aiming to test the examinee\'s ability to stay focused on the real problem and avoid being misled.',
    'Vague memories: This principle leverages the examinee\'s incomplete or imprecise recollection of learned concepts, facts, or processes to design distractors. The goal is to test the examinee\'s ability to discern between accurate information and distorted or partially correct recollections.',
    'Concept substitution: This principle involves replacing the core concepts in the question stem with similar but incorrect concepts to test the examinee\'s deep understanding of the topic and their ability to recognize subtle differences.',
    'Reversing the primary and secondary relationships: This principle modifies the logical structure of the question by reversing primary and secondary relationships, such as cause and effect, to test the examinee\'s judgment and understanding of the connections between concepts.',
    'Over-detailing or generalization: This principle provides distractors with excessive detail or oversimplified generalizations, aiming to challenge the examinee\'s ability to identify the core idea and avoid being overwhelmed by irrelevant information.'
]




# 设置参数
temperature = 1
top_p = 1
presence_penalty = 0.0

# 定义函数来获取响应
def get_response(prompt):
    response = client.chat.completions.create(
        model=api_model,
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

# qg_prompt = pf.producePrompt("qg", examples=question_examples)    
# q = get_response(qg_prompt)
# print("出的题:\n", q)
# questiondata = format_question_output(q)
# print("题目信息:", questiondata)
# rg_prompt = pf.producePrompt("rg", questiondata, distractor_principle)
# # print(dg_prompt)
# r = get_response(rg_prompt)
# print("错误推理:\n", r)

# example = format_rationale_output(r)
# dg_prompt = pf.producePrompt("dg", questiondata, example)
# d = get_response(dg_prompt)
# print("干扰项:\n", d)
# import json
# import os


# 定义函数来逐条读取测试集
def read_test_data_iter(test_file):
    with open(test_file, 'r') as f:
        test_data = json.load(f)
        for item in test_data:
            yield {
                "question": item["question"],
                "correct_answer": item["correct_answer"],
                "support": item["support"]
            }

# 定义函数来追加写入结果文件
def append_to_output_file(output_file, data):
    if os.path.exists(output_file):
        with open(output_file, 'r+') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
            existing_data.append(data)
            f.seek(0)
            json.dump(existing_data, f, indent=4)
    else:
        with open(output_file, 'w') as f:
            json.dump([data], f, indent=4)
# 示例用法
test_filename = "./evaluation/test.json"
output_filename = "./evaluation/output_dg.json"

# 逐条读取和处理测试集
for question_data in read_test_data_iter(test_filename):
    # qg_prompt = pf.producePrompt("qg", examples=question_examples)
    # q = get_response(qg_prompt)
    # print("出的题:\n", q)
    # questiondata = format_question_output(q)
    # print("题目信息:", questiondata)
    
    rg_prompt = pf.producePrompt("rg", question_data, distractor_principle)
    print("观察rg的prompt:\n", rg_prompt)
    r = get_response(rg_prompt)
    print("错误推理:\n", r)
    
    example = format_rationale_output(r)
    print("观察dg的example:\n", example)
    dg_prompt = pf.producePrompt("dg", question_data, example)
    print("观察dg的prompt:\n", dg_prompt)
    d = get_response(dg_prompt)
    print("干扰项:\n", d)
    extracted_distractors = format_distractor_output(d)
    print("提取的干扰项:", extracted_distractors)
    # 将结果打包为一个JSON对象
    result = {
        "question": question_data['question'],
        "correct_answer": question_data['correct_answer'],
        "distractor1": extracted_distractors['distractor1'],
        "distractor2": extracted_distractors['distractor2'],
        "distractor3": extracted_distractors['distractor3']
    }
    
    # 追加写入结果文件
    append_to_output_file(output_filename, result)

print(f"所有干扰项已保存到 {output_filename}")
