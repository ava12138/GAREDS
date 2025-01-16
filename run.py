import os
import json
from pandas import qcut
import yaml
import argparse
import logging
import time
from tqdm import tqdm
from openai import OpenAI
from PromptFramwork import PromptFramework as pf
from utils.utils import format_question_output, format_rationale_output, format_distractor_output


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

# 加载配置
def load_config():
    with open('./config/api.yaml', 'r') as file:
        api_config = yaml.safe_load(file)
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    with open('./config/principle.json', 'r') as file:
        principles_config = json.load(file)
    return api_config, config, principles_config


# 初始化API客户端
def initialize_api_client(api_config, model_key):
    api_key = api_config['api_key']
    api_model = api_config['model'][model_key]
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    return client, api_model

def get_response(client, api_model, prompt, temperature, top_p, presence_penalty):
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
    # 获取token使用情况
    total_tokens = response.usage.total_tokens
    return response.choices[0].message.content, total_tokens

def process_test_data(client, api_model, test_file, output_file, prompt_config, distractor_principle, temperature, top_p, presence_penalty):
    total_items = sum(1 for _ in read_test_data_iter(test_file))
    api_calls = 0
    start_time = time.time()
    # 初始化token计数
    total_tokens = 0
    cost = 0.002  # 每1K tokens的价格

    with tqdm(total=total_items, desc="Generating distractors") as pbar:
        for question_data in read_test_data_iter(test_file):
            try:
                rg_prompt = pf.producePrompt(prompt_config['rg'], question_data, distractor_principle)
                r, tokens_rg = get_response(client, api_model, rg_prompt, temperature, top_p, presence_penalty)
                api_calls += 1
                total_tokens += tokens_rg
                # print("错误推理:\n", r)

                inference = format_rationale_output(r)
                dg_prompt = pf.producePrompt(prompt_config['dg'], question_data, inference)
                # print("观察dg的prompt:\n", dg_prompt)
                d, tokens_dg = get_response(client, api_model, dg_prompt, temperature, top_p, presence_penalty)
                api_calls += 1
                total_tokens += tokens_dg

                extracted_distractors = format_distractor_output(d)
                print("提取的干扰项:", extracted_distractors)
                result = {
                    "question": question_data['question'],
                    "correct_answer": question_data['correct_answer'],
                    "distractor1": extracted_distractors['distractor1'],
                    "distractor2": extracted_distractors['distractor2'],
                    "distractor3": extracted_distractors['distractor3']
                }
                append_to_output_file(output_file, result)
            finally:
                pbar.update(1)
                elapsed = time.time() - start_time
                rate = api_calls / elapsed
                token_rate = total_tokens / elapsed
                estimated_cost = (total_tokens / 1000) * cost
                
                pbar.set_postfix({
                    'API calls': api_calls,
                    'Calls/s': f'{rate:.2f}',
                    'Tokens': total_tokens,
                    'Tokens/s': f'{token_rate:.1f}',
                    'Cost(元)': f'{estimated_cost:.3f}'
                })

    print(f"\nGeneration completed:")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Average tokens/call: {total_tokens/api_calls:.1f}")
    print(f"Token rate: {total_tokens/(time.time() - start_time):.1f} tokens/s")
    print(f"Estimated cost: ${(total_tokens/1000) * cost:.3f}")
    print(f"Results saved to {output_file}")


def main():

    # 解析参数
    parser = argparse.ArgumentParser(description="Generate distractors")
    parser.add_argument('-d', '--dataset', choices=['lan', 'nat', 'soc', 'sciqa-text', 'sciq'], required=True, help="Type of test file to process")
    parser.add_argument('-m', '--model', choices=['plus', 'qwen7b'], required=True, help="Type of model to use")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot'], required=True, help="Prompt type")
    args = parser.parse_args()

    # 加载配置
    api_config, config, principles_config = load_config()
    distractor_principle = principles_config['distractor_principle']
    prompt_config = config['prompt_types'][args.prompt]
    file_config = config['files'][args.dataset]

    # 初始化API客户端
    client, api_model = initialize_api_client(api_config, args.model)

    # 配置文件路径
    test_file = file_config['test_file']
    output_file = f"{file_config['output_file']}-{args.model}-{args.prompt}.json"

    # 参数配置
    temperature = config['temperature']
    top_p = config['top_p']
    presence_penalty = config['presence_penalty']

    # 处理测试数据
    process_test_data(client, api_model, test_file, output_file, prompt_config, distractor_principle, temperature, top_p, presence_penalty)

if __name__ == "__main__":
    main()





# # 示例用法
# test_filename = "./data_divided/sciqa-test-lan.json"
# output_filename = "./output/output_dg-sciqa-lan.json"

# # 逐条读取和处理测试集
# for question_data in read_test_data_iter(test_filename):
#     # qg_prompt = pf.producePrompt("qg", examples=question_examples)
#     # q = get_response(qg_prompt)
#     # print("出的题:\n", q)
#     # questiondata = format_question_output(q)
#     # print("题目信息:", questiondata)
    
#     rg_prompt = pf.producePrompt("rg", question_data, distractor_principle)
#     r = get_response(rg_prompt)
#     print("错误推理:\n", r)

#     example = format_rationale_output(r)
#     dg_prompt = pf.producePrompt("dg", question_data, example)
#     print("观察dg的prompt:\n", dg_prompt)
#     d = get_response(dg_prompt)

#     extracted_distractors = format_distractor_output(d)
#     print("提取的干扰项:", extracted_distractors)
#     # 将结果打包为一个JSON对象
#     result = {
#         "question": question_data['question'],
#         "correct_answer": question_data['correct_answer'],
#         "distractor1": extracted_distractors['distractor1'],
#         "distractor2": extracted_distractors['distractor2'],
#         "distractor3": extracted_distractors['distractor3']
#     }
    
#     # 追加写入结果文件
#     append_to_output_file(output_filename, result)

# print(f"所有干扰项已保存到 {output_filename}")


# with open('./config/api.yaml', 'r') as file:
#     api_config = yaml.safe_load(file)
#     api_key = api_config['api_key']
#     api_model = api_config['model']

# with open('./config/config.yaml', 'r') as file:
#     config = yaml.safe_load(file)
#     temperature = config['temperature']
#     top_p = config['top_p']
#     presence_penalty = config['presence_penalty']

# with open('./config/principle.json', 'r') as file:
#     principles_config = json.load(file)
#     distractor_principle = principles_config['distractor_principle']

# client = OpenAI(
#         api_key=api_key, 
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     )

# file_path_sciq = os.path.expanduser('/data/lzx/sciq/train.json')
# with open(file_path_sciq, 'r') as file:
#     data = json.load(file)

# question_examples = [data[0], data[1]]

# 定义函数来获取响应
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