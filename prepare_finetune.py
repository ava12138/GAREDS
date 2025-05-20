import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import random
import sys
from PIL import Image
import io


# 加载ScienceQA数据集
def load_scienceqa():
    dataset = load_dataset("derek-thomas/ScienceQA")
    print(f"原始数据集大小: train={len(dataset['train'])}, validation={len(dataset['validation'])}, test={len(dataset['test'])}")
    return dataset

# 使用与run.py相似的方法处理数据
def process_sciqa_item(item):
    """处理ScienceQA数据集的单个条目，类似于read_test_data_iter函数"""
    question_data = {}
    
    # 基本信息
    question_data['question'] = item['question']
    
    # 处理正确答案和干扰项
    correct_answer_index = item['answer']
    all_choices = item['choices']
    question_data['correct_answer'] = all_choices[correct_answer_index]
    
    # 支持信息
    question_data['support'] = f"{item.get('lecture', '')}. {item.get('solution', '')}. {item.get('hint', '')}"
    
    # 提取所有干扰项
    distractors = []
    for index_c, choice in enumerate(all_choices):
        if index_c != correct_answer_index:
            distractors.append(choice)
    
    # 添加干扰项
    for i, distractor in enumerate(distractors):
        question_data[f'distractor{i+1}'] = distractor
    
    return question_data, distractors

# 转换为干扰项生成任务的Swift格式
def convert_to_distractor_generation_format(dataset, output_dir="dg_finetune"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集、验证集和测试集
    for split in ["train", "validation"]:
        data_list = []
        split_data = dataset[split]
        print(f"开始处理{split}集，原始大小: {len(split_data)}")
        
        for item in tqdm(split_data, desc=f"处理{split}集"):
            # 跳过没有至少两个选项的问题
            if len(item["choices"]) < 2:
                continue
            
            # 使用与run.py类似的方法处理数据
            question_data, distractors = process_sciqa_item(item)
            
            # 获取干扰项数量(就是distractors的长度)
            distractor_count = len(distractors)
            
            if distractor_count == 0:
                continue  # 跳过没有干扰项的问题
            
            # 动态生成模板字符串
            template_parts = [f"Distractor{i}: **XXX**" for i in range(1, distractor_count + 1)]
            template = "\n".join(template_parts)
            
            # 使用与rule_based_dg_prompt一致的格式，包含instructions
            instructions = (
                "You are given the following question along with the correct answer as context for helping you generate distractors. "
                "The question may include an image. The image contains information that helps you understand the question and can help you generate subsequent distractors. "
                f"Please use the template to generate **{distractor_count}** alternative incorrect but plausible answers to be used as multiple-choice options in a multiple-choice exam.\n"
                "You don't need to provide the explanation of distractors, just provide the distractors.\n"
            )
            
            prompt = (
                f"{instructions}\n\n"
                f"=== Context ===\n"
                f"Question: {question_data['question'].strip()}\n"
                f"Answer: {question_data['correct_answer'].strip()}\n"
                f"Support: {question_data['support'].strip()}\n"
                f"=== Template ===\n"
                f"{template}\n"
            )
            
            # 创建干扰项输出格式的响应
            response = ""
            for i, distractor in enumerate(distractors, 1):
                response += f"Distractor{i}: **{distractor}**\n"
            
            # 使用标准messages格式
            user_message = {"role": "user", "content": prompt}
            assistant_message = {"role": "assistant", "content": response.strip()}
            
            # 创建标准格式的数据
            example = {
                "messages": [
                    {"role": "system", "content": "You are an expert in the field of education, and you are good at generating high-quality distractors that seem reasonable but are wrong based on the questions."},
                    user_message,
                    assistant_message
                ]
            }
            
            # 如果有图像，添加图像路径
            if item["image"] is not None:
                try:
                    # 保存图像路径，Swift会自动处理图像加载
                    image_dir = os.path.join(output_dir, "images", split, str(item.get("id", len(data_list))))
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, "image.jpg")
                    
                    # 根据图像类型处理
                    if hasattr(item["image"], "save"):  # PIL Image对象
                        if not os.path.exists(image_path):
                            item["image"].save(image_path)
                    elif isinstance(item["image"], dict) and "bytes" in item["image"]:  # 包含bytes的字典
                        if not os.path.exists(image_path):
                            with open(image_path, "wb") as f:
                                f.write(item["image"]["bytes"])
                    else:
                        print(f"未知图像格式: {type(item['image'])}")
                        continue
                    
                    # 将图像添加到用户消息中
                    example["messages"][1]["images"] = [image_path]
                except Exception as e:
                    print(f"处理图像时出错: {e}, 类型: {type(item['image'])}")
                    # 即使图像处理失败，仍然保留文本数据
            
            data_list.append(example)
        
        # 写入jsonl文件
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for example in data_list:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        # 计算实际写入的行数
        with open(output_file, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        
        print(f"完成{split}集干扰项生成数据，保存至 {output_file}，共 {len(data_list)} 条数据")
        print(f"文件实际行数检查: {line_count} 行")

# 创建dataset_info.json文件
def create_dataset_info(output_dir="dg_finetune"):
    """
    创建与官方格式兼容的dataset_info.json文件
    """
    # 使用正确的格式，与Swift的DatasetMeta类兼容
    dataset_info = [
        {
            "dataset_path": f"./{output_dir}"
        }
    ]
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"数据集信息保存至 {os.path.join(output_dir, 'dataset_info.json')}")

if __name__ == "__main__":
    output_dir = "dg_finetune"
    print("开始处理ScienceQA数据集用于干扰项生成任务...")
    dataset = load_scienceqa()
    convert_to_distractor_generation_format(dataset, output_dir)
    create_dataset_info(output_dir)
    print("干扰项生成数据集处理完成！")