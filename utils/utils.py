import re
import pandas as pd
import ast
import json
import time
import os 
import numpy as np
import torch
import random
import base64
from io import BytesIO
from PIL import Image,UnidentifiedImageError

def convert_image_to_base64(image, format="JPEG"):
    """将PIL图像转换为base64字符串，并允许指定输出格式，提供更详细的错误信息"""
    if not isinstance(image, Image.Image):
        return None

    try:
        buffered = BytesIO()
        image_format = format.upper()
        # 如果是 JPEG 格式，确保图片以 RGB 模式保存
        if image_format == "JPEG" and image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        image.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    except FileNotFoundError as e:
        error_message = f"文件未找到错误: {e.filename}. 请检查图片文件路径是否正确。"
        print(f"图片转换错误: {error_message}")
        return None
    except UnidentifiedImageError as e:
        error_message = f"无法识别的图像格式: {e}. 请确保图片文件是有效的图像文件。"
        print(f"图片转换错误: {error_message}")
        return None
    except Exception as e:
        error_message = f"图像转换过程中发生未知错误: {type(e).__name__} - {e}."
        print(f"图片转换错误: {error_message}")
        return None
    
def format_question_output(response):
    question_match = re.search(r'Question:\s*(.*)', response)
    answer_match = re.search(r'Answer:\s*(.*)', response)
    question = question_match.group(1).strip() if question_match else ''
    answer = answer_match.group(1).strip() if answer_match else ''
    return {'question': question, 'correct_answer': answer}

def format_rationale_output(response, format_type="rule_format"):
    """
    格式化推理输出 - 增强版
    Args:
        response: 原始响应文本
        format_type: 格式类型，可选值为 "rule_format" 或 "cot_format"
    Returns:
        dict 或 str: 根据format_type返回不同格式的结果
    """
    # 清理响应文本，去除所有的 **
    cleaned_response = response.replace('**', '')
    
    if format_type == "cot_format":
        # 只提取 Explanation 后的内容
        explanation_match = re.search(r'Explanation:?\s*(.*?)(?=\n|$)', cleaned_response, re.DOTALL)
        return explanation_match.group(1).strip() if explanation_match else ''
    else:
        # 提取 Explanation
        explanation_match = re.search(r'Explanation:?\s*(.*?)(?=\nIncorrect Inference|$)', cleaned_response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ''
        
        # 尝试多种格式的正则表达式模式
        inference_patterns = [
            # 模式1: Incorrect Inference 1: Principle 1 (Confusing...)
            r'Incorrect Inference (\d+):\s*Principle \d+ \((.*?)\):\s*(.*?)(?=\nIncorrect Inference|$)',
            
            # 模式2: Incorrect Inference1: The bilberry...
            r'Incorrect Inference(\d+):\s*(.*?)(?=\nIncorrect Inference\d+:|$)',
            
            # 模式3: Incorrect Inference 1 (Principle...): Text
            r'Incorrect Inference (\d+) \((.*?)\):\s*(.*?)(?=\nIncorrect Inference|$)',
            
            # 模式4: 如有其他变体格式可继续添加
        ]
        
        incorrect_inferences = []
        
        # 尝试每一种模式
        for pattern in inference_patterns:
            matches = re.findall(pattern, cleaned_response, re.DOTALL)
            if matches:
                # 根据正则表达式的捕获组数量决定如何处理
                if len(matches[0]) == 3:  # 三元组: 编号, 原则, 内容
                    for num, principle, inference in matches:
                        formatted_inference = f"Incorrect Inference {num} ({principle.strip()}): {inference.strip()}"
                        incorrect_inferences.append(formatted_inference)
                elif len(matches[0]) == 2:  # 二元组: 编号, 内容
                    for num, inference in matches:
                        formatted_inference = f"Incorrect Inference {num}: {inference.strip()}"
                        incorrect_inferences.append(formatted_inference)
                
                # 如果找到匹配，则跳出循环
                break
        
        # 如果没有找到匹配，尝试更宽松的模式
        if not incorrect_inferences:
            fallback_matches = re.findall(r'Incorrect.*?(\d+).*?:(.*?)(?=\nIncorrect|$)', cleaned_response, re.DOTALL)
            if fallback_matches:
                for num, inference in fallback_matches:
                    formatted_inference = f"Incorrect Inference {num}: {inference.strip()}"
                    incorrect_inferences.append(formatted_inference)
        
        return {
            'explanation': explanation,
            'incorrect_inferences': '\n'.join(incorrect_inferences) if incorrect_inferences else ''
        }

def format_distractor_output(text: str, expected_count: int = None) -> dict:
    """
    格式化干扰项输出
    Args:
        text: 原始响应文本
        expected_count: 预期的干扰项数量，如果为None则提取所有找到的干扰项
    Returns:
        dict: 包含格式化后干扰项的字典
    """
    output = {}
    # 去除所有的 **
    cleaned_text = text.replace('**', '')
    
    # 匹配干扰项
    distractor_pattern = re.compile(r'Distractor\d:\s*(.*?)\s*(?=\n|$)')
    matches = distractor_pattern.findall(cleaned_text)
    
    # 如果指定了预期数量，只取指定数量的干扰项
    count = expected_count if expected_count is not None else len(matches)
    
    for i, match in enumerate(matches[:count], 1):
        output[f'distractor{i}'] = match.strip()
    
    return output


def read_test_data(test_file):
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    question_data_list = []
    for item in test_data:
        question_data = {
            "Question": item["question"],
            "Answer": item["correct_answer"],
            "Support": item["support"]
        }
        question_data_list.append(question_data)
    
    return question_data_list

def str_to_dict_eedi_df(df: pd.DataFrame):
    cols = ["correct_option", "gt_distractors", "generated_distractors", "distractors", "construct_info"]
    cols = [col for col in cols if col in df.columns]
    for i, row in df.iterrows():
        for col in cols:
            try:
                df.at[i, col] = ast.literal_eval(row[col])
            except Exception:
                df.at[i, col] = None
    return df

def initialize_seeds(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

def get_processed_count(output_file):
    """获取已处理的数据项数量，用于断点续传"""
    if not os.path.exists(output_file):
        return 0
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            if isinstance(existing_results, list):
                return len(existing_results)
            else:
                return 1 if existing_results else 0
    except json.JSONDecodeError:
        return 0

def log_error(error_log_file, index, question_data, error_msg, response=None):
    """记录错误到日志文件"""
    error_record = {
        "index": index,
        "question": question_data['question'],
        "correct_answer": question_data['correct_answer'],
        "error": str(error_msg),
        "original_response": response if response else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
    with open(error_log_file, 'a', encoding='utf-8') as f:
        json.dump(error_record, f, ensure_ascii=False)
        f.write('\n')
    return error_record

def create_error_result(question_data, distractor_count, error_type="API_ERROR"):
    """为出错的问题创建结果字典"""
    result = {
        "question": question_data['question'],
        "correct_answer": question_data['correct_answer']
    }
    for i in range(1, distractor_count + 1):
        result[f'distractor{i}'] = error_type
    return result

def update_progress_stats(pbar, api_calls, total_tokens, start_time):
    """更新进度条统计信息"""
    elapsed = time.time() - start_time
    rate = api_calls / elapsed if elapsed > 0 else 0
    token_rate = total_tokens / elapsed if elapsed > 0 else 0
    
    pbar.set_postfix({
        'API calls': api_calls,
        'Calls/s': f'{rate:.2f}',
        'Tokens': total_tokens,
        'Tokens/s': f'{token_rate:.1f}'
    })

def print_final_stats(start_time, api_calls, total_tokens, output_file):
    """打印最终统计数据"""
    elapsed = time.time() - start_time
    print(f"\nGeneration completed:")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Average tokens/call: {total_tokens/api_calls if api_calls else 0:.1f}")
    print(f"Token rate: {total_tokens/elapsed if elapsed else 0:.1f} tokens/s")
    print(f"Results saved to {output_file}")

