import os
import json
import yaml
import argparse
import time
from tqdm import tqdm
from openai import OpenAI
from PromptFramwork import PromptFramework as pf
from RetrieveFramework import RetrieverFramework as rf
from datasets import load_dataset
from utils.utils import initialize_seeds
from utils.utils import (
    format_question_output, format_rationale_output, format_distractor_output,
        get_processed_count, log_error, create_error_result, update_progress_stats, print_final_stats
)


# 定义函数来逐条读取测试集
def read_test_data_iter(dataset, start_index=0,  split='test'):
    """逐条读取 Hugging Face 数据集数据
    Returns:
        tuple: (数据集长度, 数据迭代器)
    """
    hf_dataset = load_dataset(dataset, split=split)
    print(f"当前加载的数据集分割: {split}")
    print(f"数据集样本数量: {len(hf_dataset)}")

    def data_iterator():
        for index, sample in enumerate(hf_dataset): #  使用 enumerate 获取索引
            if index < start_index: #  跳过已处理的数据项
                 continue
            
            transformed_sample = {}
            
            # 添加原始索引
            transformed_sample['original_index'] = index
            # 保留基本信息
            transformed_sample['question'] = sample['question']
            # transformed_sample['image'] = sample['image']
            transformed_sample['subject'] = sample['subject']
            
            # 合并 lecture 和 solution 作为支持文本
            transformed_sample['support'] = f"{sample['lecture']}. {sample['solution']}. {sample['hint']}"
            
            # 处理正确答案和干扰项
            correct_answer_index = sample['answer']
            all_choices = sample['choices']
            transformed_sample['correct_answer'] = all_choices[correct_answer_index]

            # 提取所有干扰项
            distractors = []
            for index_c, choice in enumerate(all_choices): #  避免变量名冲突，修改为 index_c
                if index_c != correct_answer_index:
                    distractors.append(choice)
            
            # 添加所有干扰项
            for i, distractor in enumerate(distractors):
                transformed_sample[f'distractor{i+1}'] = distractor
            
            yield transformed_sample
    
    return len(hf_dataset), data_iterator()
    
# 定义函数来追加写入结果文件
def batch_append_to_output_file(output_file, data_list): #  函数保持不变
    if os.path.exists(output_file):
        with open(output_file, 'r+') as f:
            try:
                existing_data_list = json.load(f)
                if not isinstance(existing_data_list, list):
                    existing_data_list = [existing_data_list]
            except json.JSONDecodeError:
                existing_data_list = []
            existing_data_list.extend(data_list) #  批量追加数据
            f.seek(0)
            json.dump(existing_data_list, f, indent=4, ensure_ascii=False)
    else:
        with open(output_file, 'w') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)

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

def get_response(client, api_model, prompt, temperature, presence_penalty):
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        presence_penalty=presence_penalty,
    )
    # 获取token使用情况
    total_tokens = response.usage.total_tokens
    return response.choices[0].message.content, total_tokens

def generate_rationale(client, api_model, question_data, prompt_config, distractor_principle, temperature, presence_penalty, idx=None):
    """生成错误推理"""

    similar_examples = None
    if idx is not None:
        try:
            similar_examples = rf.get_similar_examples(idx, k=3)
        except Exception as e:
            print(f"\n\033[93m警告: 检索相似示例失败: {str(e)}\033[0m")

    rg_prompt = pf.producePrompt(prompt_config['rg'], question_data, distractor_principle, similar_examples)
    response, tokens = get_response(client, api_model, rg_prompt, temperature, presence_penalty)
    
    # 格式化并验证推理结果
    inference = format_rationale_output(response, prompt_config['format'])
    
    # 验证推理结果是否为空
    is_empty = False
    if isinstance(inference, dict) and 'incorrect_inferences' in inference:
        is_empty = not inference['incorrect_inferences'].strip()
    elif isinstance(inference, str):
        is_empty = not inference.strip()
        
    if is_empty:
        print("\n\033[91m警告: 错误推理为空！\033[0m")
        print("\033[93m原始推理内容:\033[0m")
        print(response)
        raise ValueError("错误推理格式化结果为空")
        
    return inference, tokens, response

def generate_distractors(client, api_model, question_data, inference, prompt_config, temperature, presence_penalty, idx=None):
    """生成干扰项"""

    similar_examples = None
    if idx is not None:
        try:
            similar_examples = rf.get_similar_examples(idx, k=3)
        except Exception as e:
            print(f"\n\033[93m警告: 检索相似示例失败: {str(e)}\033[0m")

    dg_prompt = pf.producePrompt(prompt_config['dg'], question_data, inference, similar_examples)
    response, tokens = get_response(client, api_model, dg_prompt, temperature, presence_penalty)
    
    # 获取当前问题需要的干扰项数量并提取
    distractor_count = pf.count_distractors(question_data)
    extracted_distractors = format_distractor_output(response, distractor_count)
    
    return extracted_distractors, tokens, distractor_count

def process_test_data(client, api_model, dataset, output_file, prompt_config, distractor_principle, temperature, presence_penalty):
    """处理测试数据的主函数"""
    api_calls = 0
    total_tokens = 0
    start_time = time.time()
    results_buffer = []
    batch_size = 10
    error_log_file = "./log/api_error_log.json"

    # 加载断点续传信息
    processed_count = get_processed_count(output_file)
    total_items, dataset_iter = read_test_data_iter(dataset, start_index=processed_count)
    current_index = processed_count
    # 确保错误日志目录存在
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)

    with tqdm(total=total_items, initial=processed_count, desc="Generating distractors") as pbar:
        for question_data in dataset_iter:
            
            idx = question_data['original_index']

            try:
                # 生成错误推理
                inference, tokens_rg, raw_inference = None, 0, None
                try:
                    inference, tokens_rg, raw_inference = generate_rationale(
                        client, api_model, question_data, prompt_config, 
                        distractor_principle, temperature, presence_penalty,
                        idx=idx
                    )
                    api_calls += 1
                    total_tokens += tokens_rg
                except Exception as e:
                    print(f"\n\033[93m推理生成错误 (索引 {current_index}): {str(e)}\033[0m")
                    if raw_inference:
                        print(f"原始推理输出: {raw_inference[:200]}...")
                    log_error(error_log_file, current_index, question_data, f"推理错误: {e}")
                    distractor_count = pf.count_distractors(question_data)
                    results_buffer.append(create_error_result(question_data, distractor_count, "INFERENCE_ERROR"))
                    current_index += 1
                    continue
                
                # 生成干扰项
                try:
                    extracted_distractors, tokens_dg, distractor_count = generate_distractors(
                        client, api_model, question_data, inference, prompt_config,
                        temperature, presence_penalty,
                        idx=idx
                    )
                    api_calls += 1
                    total_tokens += tokens_dg
                    
                    # 构建结果
                    result = {
                        "question": question_data['question'],
                        "correct_answer": question_data['correct_answer']
                    }
                    for i in range(1, distractor_count + 1):
                        result[f'distractor{i}'] = extracted_distractors.get(f'distractor{i}', '')
                    
                    results_buffer.append(result)
                except Exception as e:
                    print(f"\n\033[93m干扰项生成错误 (索引 {current_index}): {str(e)}\033[0m")
                    log_error(error_log_file, current_index, question_data, f"干扰项错误: {e}")
                    distractor_count = pf.count_distractors(question_data)
                    results_buffer.append(create_error_result(question_data, distractor_count, "DISTRACTOR_ERROR"))
            
            except Exception as e:
                # 捕获其他意外错误
                print(f"\n\033[91m未预期错误 (索引 {current_index}): {str(e)}\033[0m")
                log_error(error_log_file, current_index, question_data, f"未预期错误: {e}")
                distractor_count = pf.count_distractors(question_data)
                results_buffer.append(create_error_result(question_data, distractor_count, "UNEXPECTED_ERROR"))
            
            finally:
                # 批量写入检查
                if len(results_buffer) >= batch_size:
                    batch_append_to_output_file(output_file, results_buffer)
                    results_buffer = []
                
                # 更新进度
                current_index += 1
                pbar.update(1)
                update_progress_stats(pbar, api_calls, total_tokens, start_time)

    # 处理剩余结果
    if results_buffer:
        batch_append_to_output_file(output_file, results_buffer)
    
    print_final_stats(start_time, api_calls, total_tokens, output_file)

def main():

    # 解析参数
    parser = argparse.ArgumentParser(description="Generate distractors")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Hugging Face Dataset name (e.g., science_qa, squad)")
    parser.add_argument('-m', '--model', choices=['plus', 'qwen7b', 'dp-qwen7b'], required=True, help="Type of model to use")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], required=True, help="Prompt type")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--split', type=str, default='test', help="Dataset split to use (train, validation, test)") # 添加 split 参数
    args = parser.parse_args()

    rf.load_caches()
    # 加载配置
    api_config, config, principles_config = load_config()
    distractor_principle = principles_config['distractor_principle']
    prompt_config = config['prompt_types'][args.prompt]

    initialize_seeds(args.seed)
    # 初始化API客户端
    client, api_model = initialize_api_client(api_config, args.model)

    # 获取数据集名称和配置
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    file_config = config['files'].get(args.dataset)
    
    if file_config and 'test_file' in file_config:
        # 对于本地数据集（如 soc, lan 等），使用本地文件路径
        dataset_name = file_config['test_file']
    
    output_file = f"{file_config['output_file']}-{args.model}-{args.prompt}.json"

    # 参数配置
    temperature = config['temperature']
    presence_penalty = config['presence_penalty']

    # 处理测试数据
    process_test_data(client, api_model, dataset_name, output_file, prompt_config, distractor_principle, temperature,  presence_penalty)

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