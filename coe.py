import os
import json
from urllib import response
import yaml
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from PromptFramwork import PromptFramework as pf
from RetrieveFramework import RetrieverFramework as rf
from utils.utils import (
    initialize_seeds, format_rationale_output, format_distractor_output,
    get_processed_count, log_error, create_error_result
)

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class ModelInference:
    def __init__(self, model_name, device_id=0):
        """初始化 BLIP 模型"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        self.model_name = model_name
        print(f"正在加载模型: {self.model_name} 到 GPU: {device_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map={'': f'cuda:{device_id}'},
            trust_remote_code=True
        ).eval()
        print(f"模型 {model_name} 加载完成")

    def generate_response(self, prompt, image=None, temperature=0.7, presence_penalty=0.0, max_tokens=2048):
        """生成响应"""
        try:
            # 处理图片（如果有的话）
            if image is not None:
                from PIL import Image
                import base64
                from io import BytesIO
                max_image_size = (448, 448)
                img = None                
                # 将图像转换为 base64 字符串
                try:
                    # 检查输入是文件路径还是 PIL Image 对象
                    if isinstance(image, str):
                        if os.path.exists(image):
                            img = Image.open(image).convert('RGB') # 确保是 RGB 格式
                            print(f"从路径加载图片: {image}")
                    elif hasattr(image, 'thumbnail') and callable(image.thumbnail): # 检查是否像 PIL Image 对象
                        img = image.convert('RGB') # 确保是 RGB 格式
                        print("直接使用 PIL 图片对象")
                    else:
                        print(f"警告: 未知的图片类型: {type(image)}")

                    if img:
                        print(f"原始图片尺寸: {img.size}, 模式: {img.mode}")
                        # 缩放图片，保持纵横比
                        img.thumbnail(max_image_size, Image.Resampling.LANCZOS)
                        print(f"缩放后图片尺寸: {img.size}")

                        buffered = BytesIO()
                        # 保存为 JPEG 通常更小，除非需要透明度（PNG）
                        save_format = "JPEG"
                        img.save(buffered, format=save_format, quality=90) # 可以调整 quality
                        img_str = base64.b64encode(buffered.getvalue()).decode()

                        # 使用 <img> 标签将 Base64 图像嵌入提示
                        processed_image_prompt = f"<img>{img_str}</img>\n{prompt}"
                        print("图片已成功加载、缩放并编码为 Base64")

                except Exception as img_e:
                    print(f"处理图片时出错: {img_e}")
                    # 如果图片处理失败，可以选择继续（不带图片）或抛出错误
                    # 这里选择继续，但打印警告
                    print("警告: 图片处理失败，将仅使用文本提示。")

            response, _ = self.model.chat(self.tokenizer, query=prompt, history=None)
            return response
            
        except Exception as e:
            print(f"生成响应时出错: {str(e)}")
            raise

def read_test_data_iter(dataset, start_index=0, split='test', img="img"):
    """逐条读取数据集数据，支持 Hugging Face 数据集和本地 JSON 文件"""
    # 检查是否为本地 JSON 文件
    if dataset.endswith('.json'):
        try:
            with open(dataset, 'r', encoding='utf-8') as f:
                local_dataset = json.load(f)
            print(f"已加载本地数据集: {dataset}")
            print(f"数据集样本数量: {len(local_dataset)}")

            def local_data_iterator():
                for index, sample in enumerate(local_dataset):
                    if index < start_index:
                        continue
                    
                    transformed_sample = {
                        'question': sample['question'],
                        'correct_answer': sample['correct_answer'],
                        'support': sample.get('support', ''),
                        'subject': sample.get('subject', ''),
                        'category': sample.get('category', '')
                    }
                    
                    # 处理干扰项
                    distractor_keys = [k for k in sample.keys() if k.startswith('distractor')]
                    for key in distractor_keys:
                        transformed_sample[key] = sample[key]
                    
                    yield transformed_sample
            
            return len(local_dataset), local_data_iterator()
            
        except Exception as e:
            print(f"读取本地数据集时出错: {e}")
            raise
    else:
        hf_dataset = load_dataset(dataset, split=split)
        print(f"当前加载的数据集分割: {split}")
        print(f"数据集样本数量: {len(hf_dataset)}")

        def data_iterator():
            for index, sample in enumerate(hf_dataset):
                if index < start_index:
                    continue
                
                transformed_sample = {}
                # 添加原始索引
                transformed_sample['original_index'] = index
                # 保留基本信息
                transformed_sample['question'] = sample['question']
                if img == "img":
                    transformed_sample['image'] = sample.get('image')
                transformed_sample['subject'] = sample['subject']
                
                # 合并 lecture 和 solution 作为支持文本
                transformed_sample['support'] = f"{sample['lecture']}. {sample['solution']}. {sample['hint']}"
                
                # 处理正确答案和干扰项
                correct_answer_index = sample['answer']
                all_choices = sample['choices']
                transformed_sample['correct_answer'] = all_choices[correct_answer_index]

                # 提取所有干扰项
                distractors = []
                for index_c, choice in enumerate(all_choices):
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
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def generate_distractors(model, question_data, idx=None, split="test"):
    """生成干扰项"""
    # 生成干扰项提示
    is_multimodal = 'image' in question_data and question_data['image'] is not None
    similar_examples = None
    if idx is not None:
        try:
            similar_examples = rf.get_similar_examples(idx, split=split, k=1)
        except Exception as e:
            print(f"\n\033[93mWarin: Failed to retrieve: {str(e)}\033[0m")
    dg_prompt = (
        f"Context:{question_data['question'].strip()}. "
        f"Answer: {question_data['correct_answer'].strip()}.\n "
        f"Reasoing:{question_data['support'].strip()}\n"
        f"Refer to the example and based on the above context includes image and answer and reasoning, generate at least 1 plausible yet incorrect answers and separate them with numbers like (1) (2) (3).\n "
        f"Exemplar:{similar_examples}\n"
        )

    # 调用模型获取响应
    response = model.generate_response(prompt=dg_prompt, image=question_data['image'] if is_multimodal else None)
    print("\n=== Raw Response ===", response)   

    return  response

def process_test_data(model, dataset, split, output_file, multimodal):
    """处理测试数据"""
    inference_calls = 0
    start_time = time.time()
    results_buffer = []
    batch_size = 10
    error_log_file = "./log/blip_error_log.json"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)

    # 加载已处理的数据项数量，用于断点续传
    processed_count = get_processed_count(output_file)
    total_items, dataset_iter = read_test_data_iter(dataset, start_index=processed_count, split=split, img=multimodal)
    current_index = processed_count

    with tqdm(total=total_items, initial=processed_count, desc="Generating distractors") as pbar:
        for question_data in dataset_iter:
            idx = question_data.get('original_index')
            print(f"\n\033[96mProcessing Index {idx}:\033[0m")
            
            try:
                
                # 生成干扰项
                try:
                    response = generate_distractors(
                        model, question_data, idx, split
                    )
                    inference_calls += 1
            
                    # 构建结果
                    result = {
                        "question": question_data['question'],
                        "correct_answer": question_data['correct_answer'],
                        "output": response
                    }
                    results_buffer.append(result)
                    
                except Exception as e:
                    print(f"\n\033[93mDG error: Index {current_index}): {str(e)}\033[0m")
                    log_error(error_log_file, current_index, question_data, f"DG error: {e}", response)
                    results_buffer.append(create_error_result(question_data, distractor_count, "DISTRACTOR_ERROR"))
            
            except Exception as e:
                # 捕获其他意外错误
                print(f"\n\033[91m Error: (Index: {current_index}): {str(e)}\033[0m")
                log_error(error_log_file, current_index, question_data, f"Error: {e}")
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
    parser = argparse.ArgumentParser(description="Generate distractors using BLIP model")
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        help="Hugging Face Dataset name (e.g., science_qa, squad)")
    parser.add_argument('-m', '--model', type=str,
                        default="Lhh123/coe_multitask_blip2xl_angle_2ep", 
                        help=" model name")
    parser.add_argument('-v', '--vl', type=str, choices=['img', 'text'],
                        default='img', help="Multimodal type")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], 
                        required=True, help="Prompt type")
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help="指定使用的 GPU ID")
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--split', type=str, default='test',
                        help="Dataset split to use")
    args = parser.parse_args()

    rf.load_caches(args.split)
    # 加载配置
    config = load_config()

    initialize_seeds(args.seed)
    
    # 初始化模型
    model = ModelInference(
        model_name=args.model,
        device_id=args.gpu_id
    )

    # 获取数据集名称和配置
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    file_config = config['files'].get(args.dataset)
    modal_suffix = '-text' if args.vl == 'text' else ''
    if file_config and 'test_file' in file_config:
        dataset_name = file_config['test_file']
    
    output_file = f"{file_config['output_file']}-coe{modal_suffix}-{args.prompt}-{args.split}.json"

    # 处理测试数据
    process_test_data(model, dataset_name, args.split, output_file, args.vl)

if __name__ == "__main__":
    main()