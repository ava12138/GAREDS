from math import dist
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
from utils.utils import initialize_seeds, convert_image_to_base64
from utils.utils import (
     format_rationale_output, format_distractor_output,
        get_processed_count, log_error, create_error_result, update_progress_stats, print_final_stats
)


# 定义函数来逐条读取测试集
def read_test_data_iter(dataset, start_index=0,  split='test', use_multimodal=True):
    """
    逐条读取 Hugging Face 数据集数据，支持多模态。

    Args:
        dataset (str): 数据集名称或路径。
        start_index (int): 开始读取的索引。
        split (str): 数据集分割 ('train', 'validation', 'test')。
        use_multimodal (bool): 是否加载图像数据。

    Returns:
        tuple: (数据集长度, 数据迭代器)
    """
    hf_dataset = load_dataset(dataset, split=split)
    print(f"当前加载的数据集分割: {split}")
    print(f"数据集样本数量: {len(hf_dataset)}")
    print(f"多模态模式: {'启用' if use_multimodal else '禁用'}")

    def data_iterator():
        for index, sample in enumerate(hf_dataset): #  使用 enumerate 获取索引
            if index < start_index: #  跳过已处理的数据项
                 continue
            
            transformed_sample = {}
            
            # 添加原始索引
            transformed_sample['original_index'] = index
            # 保留基本信息
            transformed_sample['question'] = sample['question']
            if use_multimodal and 'image' in sample and sample['image'] is not None:
                transformed_sample['image'] = sample['image']
        
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
    """
    初始化 OpenAI API 客户端。

    Args:
        api_config (dict): API 配置。
        model_key (str): 要使用的模型键名 (例如 'plus', 'qwenvl')。

    Returns:
        tuple: (OpenAI 客户端实例, 模型名称)
    """
    api_key = api_config['api_key']
    if model_key not in api_config['model']:
        raise ValueError(f"模型键 '{model_key}' 在 api.yaml 的 'model' 配置中未找到。")
    api_model = api_config['model'][model_key]
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    return client, api_model

def get_response(client, api_model, prompt, image=None, temperature=0.7, presence_penalty=0.0):
    """
    获取API响应，支持多模态输入。

    Args:
        client: OpenAI API 客户端实例。
        api_model (str): 使用的模型名称。
        prompt (str): 发送给模型的文本提示。
        image (optional): PIL Image 对象或图像文件路径。
        temperature (float): 控制生成随机性的温度参数。
        presence_penalty (float): 控制生成内容中重复话题的惩罚参数。

    Returns:
        tuple: (响应内容字符串, 使用的总 token 数)
    """
    try:
        # 构建消息内容
        content = [{"type": "text", "text": prompt}]

        # 如果有图片，添加图片内容并进行验证
        if image is not None:
            image_base64 = convert_image_to_base64(image)
            if image_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_base64}
                })
                print("图片已成功添加到请求中")
            else:
                print("警告：图片转换失败，将仅使用文本提示。")

        messages = [{"role": "user", "content": content}]

        # 打印请求信息（调试用）
        print(f"请求模型：{api_model}")
        print(f"消息内容类型：{[item['type'] for item in content]}")

        # 调用API
        response = client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            presence_penalty=presence_penalty,
        )

        # 获取token使用情况
        total_tokens = response.usage.total_tokens
        return response.choices[0].message.content, total_tokens

    except Exception as e:
        print(f"\n\033[91mAPI调用错误: {str(e)}\033[0m")
        # 打印部分请求内容以帮助调试，避免打印过长的 base64 字符串
        debug_messages = []
        for msg in messages:
            debug_content = []
            for item in msg['content']:
                if item['type'] == 'text':
                    debug_content.append({"type": "text", "text": item['text'][:200] + "..."}) # 截断长文本
                elif item['type'] == 'image_url':
                     debug_content.append({"type": "image_url", "image_url": {"url": "base64_image_data..."}}) # 隐藏 base64
            debug_messages.append({"role": msg['role'], "content": debug_content})
        print(f"\033[93m请求内容 (部分): {debug_messages}\033[0m")
        raise

def generate_rationale(client, api_model, question_data, prompt_config, distractor_principle, temperature, presence_penalty, idx=None, split="test"):
    """
    生成错误推理，支持多模态输入。

    Args:
        client: OpenAI API 客户端实例。
        api_model (str): 使用的模型名称。
        question_data (dict): 当前问题的数据。
        prompt_config (dict): 推理生成 (rg) 的提示配置。
        distractor_principle (str): 干扰项生成原则。
        temperature (float): 温度参数。
        presence_penalty (float): 存在惩罚参数。
        idx (int, optional): 当前问题的原始索引，用于检索。
        split (str): 数据集分割。

    Returns:
        tuple: (格式化后的推理结果, 使用的 token 数, 原始 API 响应)
    """
    is_multimodal = 'image' in question_data and question_data['image'] is not None
    similar_examples = None
    if idx is not None:
        try:
            similar_examples = rf.get_similar_examples(idx, split=split, k=3)
        except Exception as e:
            print(f"\n\033[93m警告: 检索相似示例失败: {str(e)}\033[0m")

    rg_prompt = pf.producePrompt(prompt_config['rg'], question_data, distractor_principle, similar_examples)

    response, tokens = get_response(        
        client, api_model, rg_prompt,
        image=question_data.get('image') if is_multimodal else None, 
        temperature=temperature,
        presence_penalty=presence_penalty
    )
    
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
        raise ValueError("错误推理格式化结果为空")
        
    return inference, tokens, response

def generate_distractors(client, api_model, question_data, inference, prompt_config, temperature, presence_penalty, idx=None, split="test"):
    """
    生成干扰项，支持多模态输入。

    Args:
        client: OpenAI API 客户端实例。
        api_model (str): 使用的模型名称。
        question_data (dict): 当前问题的数据。
        inference (dict or str): 上一步生成的错误推理。
        prompt_config (dict): 干扰项生成 (dg) 的提示配置。
        temperature (float): 温度参数。
        presence_penalty (float): 存在惩罚参数。
        idx (int, optional): 当前问题的原始索引，用于检索。
        split (str): 数据集分割。

    Returns:
        tuple: (提取的干扰项字典, 使用的 token 数, 需要的干扰项数量, 原始 API 响应)
    """
    is_multimodal = 'image' in question_data and question_data['image'] is not None
    similar_examples = None
    if idx is not None:
        try:
            similar_examples = rf.get_similar_examples(idx, split=split, k=3)
        except Exception as e:
            print(f"\n\033[93m警告: 检索相似示例失败: {str(e)}\033[0m")

    dg_prompt, distractor_count = pf.producePrompt(prompt_config['dg'], question_data, inference, similar_examples)
    distractor_count = int(distractor_count)
    response, tokens = get_response(
        client, api_model, dg_prompt, 
        image=question_data.get('image') if is_multimodal else None,
        temperature=temperature,
        presence_penalty=presence_penalty        
    )

    extracted_distractors = format_distractor_output(response, distractor_count)
    print(f"Extracted distractors: {extracted_distractors}\n")

    is_empty = False
    if isinstance(extracted_distractors, dict):
        is_empty = all(not value.strip() for value in extracted_distractors.values())
    
    if is_empty:
        print("\n\033[91m警告: 提取的干扰项为空！\033[0m")
        raise ValueError("干扰项格式化结果为空")

    return extracted_distractors, tokens, distractor_count, response

def process_test_data(client, api_model, dataset, split, output_file, prompt_config, distractor_principle, 
    temperature, presence_penalty, use_multimodal=True):
    """
    处理测试数据的主函数，支持多模态。

    Args:
        client: OpenAI API 客户端实例。
        api_model (str): 使用的模型名称。
        dataset (str): 数据集名称或路径。
        split (str): 数据集分割。
        output_file (str): 输出结果的文件路径。
        prompt_config (dict): 提示配置。
        distractor_principle (str): 干扰项生成原则。
        temperature (float): 温度参数。
        presence_penalty (float): 存在惩罚参数。
        use_multimodal (bool): 是否以多模态模式处理数据。
    """
    api_calls = 0
    total_tokens = 0
    start_time = time.time()
    results_buffer = []
    batch_size = 10
    error_log_file = "./log/api_error_log.json"

    # 加载断点续传信息
    processed_count = get_processed_count(output_file)
    total_items, dataset_iter = read_test_data_iter(dataset, start_index=processed_count, split=split, use_multimodal=use_multimodal)
    current_index = processed_count
    # 确保错误日志目录存在
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)

    with tqdm(total=total_items, initial=processed_count, desc="Generating distractors") as pbar:
        for question_data in dataset_iter:
            
            idx = question_data['original_index']
            print(f"\n\033[96m处理索引 {idx}:\033[0m")
            raw_inference, raw_response = None, None
            try:
                # 生成错误推理
                inference, tokens_rg = None, 0
                try:
                    inference, tokens_rg, raw_inference = generate_rationale(
                        client, api_model, question_data, prompt_config, 
                        distractor_principle, temperature, presence_penalty,
                        idx=idx, split=split
                    )
                    api_calls += 1
                    total_tokens += tokens_rg
                except Exception as e:
                    print(f"\n\033[93m推理生成错误 (索引 {current_index}): {str(e)}\033[0m")
                    if raw_inference:
                        print(f"原始推理输出: {raw_inference[:200]}...")
                    log_error(error_log_file, current_index, question_data, f"推理错误: {e}", raw_inference)
                    distractor_count = pf.count_distractors(question_data)
                    results_buffer.append(create_error_result(question_data, distractor_count, "INFERENCE_ERROR"))
                    current_index += 1
                    continue
                
                # 生成干扰项
                try:
                    extracted_distractors, tokens_dg, distractor_count, raw_response = generate_distractors(
                        client, api_model, question_data, inference, prompt_config,
                        temperature, presence_penalty,
                        idx=idx, split=split
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
                    log_error(error_log_file, current_index, question_data, f"干扰项错误: {e}", raw_response)
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
    parser.add_argument('-m', '--model', choices=['plus', 'qwenvl', 'dp'], required=True, help="Type of model to use")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], required=True, help="Prompt type")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--split', type=str, default='test', help="Dataset split to use (train, validation, test)") 
    parser.add_argument('-v', '--multimodal', type=int, choices=[0, 1], default=1,
                       help="是否使用多模态信息 (0: 否/text-only, 1: 是/multimodal)")
    args = parser.parse_args()
    use_multimodal_flag = bool(args.multimodal)
    print(f"多模态模式: {'启用' if use_multimodal_flag else '禁用'}")
    # 加载缓存的相似度索引
    rf.load_caches(args.split)
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

    model_mode_suffix = "-text" if not use_multimodal_flag else "" 
    output_file = f"{file_config.get('output_file', f'./output/output_dg-{args.dataset}')}-{args.model}{model_mode_suffix}-{args.prompt}-{args.split}.json"

    # 参数配置
    temperature = config['temperature']
    presence_penalty = config['presence_penalty']

    # 处理测试数据
    process_test_data(client, api_model, dataset_name, args.split, output_file, prompt_config, distractor_principle,
         temperature,  presence_penalty, use_multimodal=use_multimodal_flag)

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