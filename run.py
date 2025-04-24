import os
import json
import yaml
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
from ModelInference import ModelInference as model_inference
from PromptFramwork import PromptFramework as pf
from RetrieveFramework import RetrieverFramework as rf
from utils.utils import (
    initialize_seeds, format_rationale_output,
    format_distractor_output, get_processed_count,  log_error, create_error_result
)

def read_test_data_iter(dataset, start_index=0, split='test', use_multimodal=True):
    """逐条读取数据集数据，支持 Hugging Face 数据集和本地 JSON 文件
    Args:
        dataset: 数据集名称或本地文件路径
        start_index: 起始索引
        split: 数据集分割名称（仅用于 Hugging Face 数据集）
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

def batch_append_to_output_file(output_file, data_list):
    """批量追加写入结果文件"""
    if os.path.exists(output_file):
        with open(output_file, 'r+') as f:
            try:
                existing_data_list = json.load(f)
                if not isinstance(existing_data_list, list):
                    existing_data_list = [existing_data_list]
            except json.JSONDecodeError:
                existing_data_list = []
            existing_data_list.extend(data_list)
            f.seek(0)
            f.truncate()
            json.dump(existing_data_list, f, indent=4, ensure_ascii=False)
    else:
        with open(output_file, 'w') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)

def load_config():
    """加载配置文件"""
    with open('./config/api.yaml', 'r') as file:
        api_config = yaml.safe_load(file)
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    with open('./config/principle.json', 'r') as file:
        principles_config = json.load(file)
    return api_config, config, principles_config


def generate_rationale(model, question_data, prompt_config, distractor_principle, temperature, presence_penalty, max_token_config, idx=None, split="test"):
    """生成错误推理，支持多模态输入"""
    # 检查是否为多模态问题
    is_multimodal = 'image' in question_data and question_data['image'] is not None
    similar_examples = None
    if idx is not None:
        try:
            similar_examples = rf.get_similar_examples(idx, split=split, k=3)
        except Exception as e:
            print(f"\n\033[93m警告: 检索相似示例失败: {str(e)}\033[0m")
    # 生成推理提示
    rg_prompt = pf.producePrompt(prompt_config['rg'], question_data, distractor_principle, similar_examples)
    
    # 调用模型获取响应
    response = model.generate_response(
        prompt=rg_prompt,
        image=question_data['image'] if is_multimodal else None,
        temperature=temperature,
        presence_penalty=presence_penalty,
        max_tokens=max_token_config['rg']
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
        print("\033[93m原始推理内容:\033[0m")
        print(response)
        raise ValueError("错误推理格式化结果为空")
        
    return inference, response

def generate_distractors(model, question_data, inference, prompt_config, temperature, presence_penalty, max_token_config, idx=None, split="test"):
    """生成干扰项，支持多模态输入"""
    # 检查是否为多模态问题
    is_multimodal = 'image' in question_data and question_data['image'] is not None
    similar_examples = None
    if idx is not None:
        try:
            similar_examples = rf.get_similar_examples(idx, split=split, k=3)
        except Exception as e:
            print(f"\n\033[93m警告: 检索相似示例失败: {str(e)}\033[0m")
    # 生成干扰项提示
    dg_prompt, distractor_count = pf.producePrompt(prompt_config['dg'], question_data, inference)
    distractor_count = int(distractor_count)
    # 调用模型获取响应
    response = model.generate_response(
        prompt=dg_prompt,
        image=question_data['image'] if is_multimodal else None,
        temperature=temperature,
        presence_penalty=presence_penalty,
        max_tokens=max_token_config['dg']
    )
    
    extracted_distractors = format_distractor_output(response, distractor_count)
    print("\n=== 干扰项提取结果 ===", extracted_distractors)

    is_empty = False
    if isinstance(extracted_distractors, dict):
        is_empty = all(not value.strip() for value in extracted_distractors.values())
    
    if is_empty:
        print("\n\033[91m警告: 提取的干扰项为空！\033[0m")
        raise ValueError("干扰项格式化结果为空")   

    return extracted_distractors, distractor_count, response

def process_test_data(model, dataset, split, output_file, prompt_config, distractor_principle, 
        temperature, presence_penalty, max_token_config, use_multimodal=True):
    """处理测试数据"""
    inference_calls = 0
    start_time = time.time()
    results_buffer = []
    batch_size = 10
    error_log_file = "./log/local_error_log.json"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)

    # 加载已处理的数据项数量，用于断点续传
    processed_count = get_processed_count(output_file)
    total_items, dataset_iter = read_test_data_iter(dataset, start_index=processed_count, split=split, use_multimodal=use_multimodal)
    current_index = processed_count

    with tqdm(total=total_items, initial=processed_count, desc="Generating distractors") as pbar:
        for question_data in dataset_iter:

            idx = question_data['original_index']
            print(f"\n\033[96m处理索引 {idx}:\033[0m")
            try:
                # 生成错误推理
                try:
                    inference, raw_inference = generate_rationale(
                        model, question_data, prompt_config, distractor_principle,
                        temperature, presence_penalty, max_token_config, idx, split
                    )
                    inference_calls += 1
                    
                        
                except Exception as e:
                    print(f"\n\033[93m推理生成错误 (索引 {current_index}): {str(e)}\033[0m")
                    if raw_inference:
                        print(f"原始推理输出: {raw_inference[:200]}...")
                    log_error(error_log_file, current_index, question_data, f"推理错误: {e}", raw_inference)
                    distractor_count = pf.count_distractors(question_data)
                    results_buffer.append(create_error_result(question_data, distractor_count, "INFERENCE_ERROR"))
                    current_index += 1
                    pbar.update(1)
                    continue
                
                # 生成干扰项
                try:
                    extracted_distractors, distractor_count, raw_distractor = generate_distractors(
                        model, question_data, inference, prompt_config,
                        temperature, presence_penalty, max_token_config, idx, split
                    )
                    inference_calls += 1
            
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
                    log_error(error_log_file, current_index, question_data, f"干扰项错误: {e}", raw_distractor)
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
                elapsed = time.time() - start_time
                rate = inference_calls / elapsed if elapsed > 0 else 0
                
                pbar.set_postfix({
                    'Calls': inference_calls,
                    'Calls/s': f'{rate:.2f}'
                })

    # 保存剩余结果
    if results_buffer:
        batch_append_to_output_file(output_file, results_buffer)

    print(f"\nGeneration completed:")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Total model calls: {inference_calls}")
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate distractors")
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                       help="Hugging Face Dataset name (e.g., science_qa, squad)")
    parser.add_argument('-m', '--model', choices=['qwenvl','dp'],
                       required=True, help="模型名称")
    parser.add_argument('-i', '--inference', choices=['pt', 'vllm', 'hf'],
                    default='pt', help="推理后端类型")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], 
                       required=True, help="Prompt type")
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                    help="指定使用的 GPU ID")
    parser.add_argument('-s', '--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--split', type=str, default='test',
                       help="Dataset split to use")
    parser.add_argument('-v', '--multimodal', type=int, choices=[0, 1], default=1,
                       help="是否使用多模态信息 (0: 否, 1: 是)")
    args = parser.parse_args()

    # 将整数参数转换为布尔值
    use_multimodal_flag = bool(args.multimodal)
    print(f"多模态模式: {'启用' if use_multimodal_flag else '禁用'}")

    # 加载缓存的相似度索引
    rf.load_caches(args.split)
    # 加载配置
    api_config, config, principles_config = load_config()
    distractor_principle = principles_config['distractor_principle']
    model_path = config['model_path'][args.model]
    prompt_config = config['prompt_types'][args.prompt]
    token_config = config['max_tokens']

    initialize_seeds(args.seed)
    
    # 初始化推理引擎
    model = model_inference(
        model_path=model_path,
        inference_type=args.inference,
        device_id=args.gpu_id,
    )

    # 获取数据集名称和配置
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    file_config = config['files'].get(args.dataset)
    
    if file_config and 'test_file' in file_config:
        dataset_name = file_config['test_file']
    
    model_mode_suffix = "-text" if not use_multimodal_flag else ""
    output_file = f"{file_config['output_file']}-{args.model}{model_mode_suffix}-{args.prompt}-{args.split}-local.json"

    # 参数配置
    temperature = config['temperature']
    presence_penalty = config['presence_penalty']

    # 处理测试数据
    process_test_data(model, dataset_name, args.split,  output_file, prompt_config, distractor_principle,
                         temperature, presence_penalty, token_config, use_multimodal=use_multimodal_flag)

if __name__ == "__main__":
    main()