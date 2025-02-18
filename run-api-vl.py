import os
import json
import yaml
import argparse
import time
from tqdm import tqdm
from openai import OpenAI
from PromptFramwork import PromptFramework as pf
from datasets import load_dataset
from utils.utils import initialize_seeds, convert_image_to_base64
from utils.utils import (
    format_question_output, format_rationale_output, format_distractor_output, count_distractors
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
            
            # 保留基本信息
            transformed_sample['question'] = sample['question']
            transformed_sample['image'] = sample['image']
            transformed_sample['subject'] = sample['subject']
            
            # 合并 lecture 和 solution 作为支持文本
            transformed_sample['support'] = f"{sample['lecture']}. {sample['solution']}"
            
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

def get_response(client, api_model, prompt, image=None, temperature=0.7, top_p=0.95, presence_penalty=0.0):
    """获取API响应，支持多模态输入"""
    try:
        # 构建消息内容
        content = [{"type": "text", "text": prompt}]
        
        # 如果有图片，添加图片内容
        if image is not None:
            image_base64 = convert_image_to_base64(image)
            if image_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_base64}
                })
        
        messages = [{"role": "user", "content": content}]
        
        # 调用API
        response = client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )
        
        return response.choices[0].message.content, response.usage.total_tokens
            
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        raise

def process_test_data(client, api_model, dataset, output_file, prompt_config, distractor_principle, temperature, top_p, presence_penalty):
    api_calls = 0
    start_time = time.time()
    # 初始化token计数
    total_tokens = 0

    # 加载已处理的数据项数量，用于断点续传
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    processed_count = len(existing_results)
                else:
                    processed_count = 1 if existing_results else 0 #  兼容非列表格式
            except json.JSONDecodeError:
                processed_count = 0

        # 获取数据集长度和迭代器
    total_items, dataset_iter = read_test_data_iter(dataset, start_index=processed_count)

    batch_size = 10 # 批量写入大小
    results_buffer = []

    with tqdm(total=total_items, initial=processed_count, desc="Generating distractors") as pbar:
        for question_data in dataset_iter:
            try:
                # 检查是否为多模态问题
                is_multimodal = 'image' in question_data and question_data['image'] is not None
                
                # 生成错误推理
                rg_prompt = pf.producePrompt(prompt_config['rg'], question_data, distractor_principle)
                r, tokens_rg = get_response(
                    client, api_model, rg_prompt,
                    image=question_data['image'] if is_multimodal else None,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty
                )
                
                api_calls += 1
                total_tokens += tokens_rg

                # 生成干扰项
                inference = format_rationale_output(r, prompt_config['format'])
                dg_prompt = pf.producePrompt(prompt_config['dg'], question_data, inference)
                d, tokens_dg = get_response(
                    client, api_model, dg_prompt,
                    image=question_data['image'] if is_multimodal else None,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty
                )
                
                api_calls += 1
                total_tokens += tokens_dg

                # 生成干扰项
                inference = format_rationale_output(r, prompt_config['format'])
                dg_prompt = pf.producePrompt(prompt_config['dg'], question_data, inference)
                
                if is_multimodal :  # 多模态模型处理带图片的问题
                    d, tokens_dg = get_response(
                        client, api_model, dg_prompt,
                        image=question_data['image'],
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=presence_penalty
                    )
                else:  # 文本模式处理
                    d, tokens_dg = get_response(
                        client, api_model, dg_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=presence_penalty
                    )
                api_calls += 1
                total_tokens += tokens_dg

               # 获取当前问题需要的干扰项数量
                distractor_count = pf.count_distractors(question_data)

                # 使用预期数量提取干扰项
                extracted_distractors = format_distractor_output(d, distractor_count)
                print("提取的干扰项:", extracted_distractors)
                
                # 构建动态结果字典
                result = {
                    "question": question_data['question'],
                    "correct_answer": question_data['correct_answer']
                }
                # 动态添加干扰项
                for i in range(1, distractor_count + 1):
                    result[f'distractor{i}'] = extracted_distractors.get(f'distractor{i}', '')
                    
                results_buffer.append(result)
                if len(results_buffer) >= batch_size: #  当 buffer 大小达到 batch_size 时，批量写入
                    batch_append_to_output_file(output_file, results_buffer)
                    results_buffer = []

            finally:
                pbar.update(1)
                elapsed = time.time() - start_time
                rate = api_calls / elapsed
                token_rate = total_tokens / elapsed
                
                pbar.set_postfix({
                    'API calls': api_calls,
                    'Calls/s': f'{rate:.2f}',
                    'Tokens': total_tokens,
                    'Tokens/s': f'{token_rate:.1f}'
                })

     #  处理最后一批不足 batch_size 的结果
    if results_buffer:
        batch_append_to_output_file(output_file, results_buffer)
        results_buffer = []

    print(f"\nGeneration completed:")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Average tokens/call: {total_tokens/api_calls:.1f}")
    print(f"Token rate: {total_tokens/(time.time() - start_time):.1f} tokens/s")
    print(f"Results saved to {output_file}")


def main():

    # 解析参数
    parser = argparse.ArgumentParser(description="Generate distractors")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Hugging Face Dataset name (e.g., science_qa, squad)")
    parser.add_argument('-m', '--model', choices=['plus', 'qwen7b','vl'], required=True, help="Type of model to use")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], required=True, help="Prompt type")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--split', type=str, default='test', help="Dataset split to use (train, validation, test)") # 添加 split 参数
    args = parser.parse_args()

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
    top_p = config['top_p']
    presence_penalty = config['presence_penalty']

    # 处理测试数据
    process_test_data(client, api_model, dataset_name, output_file, prompt_config, distractor_principle, temperature, top_p, presence_penalty)

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