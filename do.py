import json
import os
from tqdm import tqdm

def process_predictions(input_files, output_file):
    """处理并合并多个预测文件"""
    all_results = []
    
    for file_name in tqdm(input_files, desc="处理文件"):
        with open(file_name, 'r') as f:
            predictions = json.load(f)
            
        for item in predictions:
            # 提取ID中的索引
            idx = int(item['id'].split('_')[1])
            
            # 解析干扰项响应
            response = item['response']
            
            # 匹配 (1) xxx (2) xxx (3) xxx 或 1. xxx 2. xxx 3. xxx 格式
            import re
            pattern1 = r'\((\d+)\)\s*(.*?)(?=\(\d+\)|$)'  # 添加\s*使其可以匹配零个或多个空白字符
            pattern2 = r'(\d+)[\.)]([\s\S]*?)(?=\d+[\.)]|$)'
            
            matches = list(re.finditer(pattern1, response)) or list(re.finditer(pattern2, response))
            
            result = {
                "question": f"placeholder_question_{idx}",  # 占位符
                "correct_answer": "placeholder_answer",     # 占位符
            }
            
            # 提取干扰项
            for match in matches:
                index = match.group(1)
                text = match.group(2).strip()
                if text:  # 避免空字符串
                    result[f'distractor{index}'] = text
            
            all_results.append(result)
            result = {
                "question": f"placeholder_question_{idx}",
                "correct_answer": "placeholder_answer",
            }
            
            # 按换行分割多个选项
            parts = response.split('\n')
            
            # 处理每个选项
            for i, part in enumerate(parts, 1):
                # 移除(1)、(2)等标记并清理文本
                text = part.strip()
                if text:  # 确保非空
                    if text.startswith('('):
                        # 找到)的位置
                        pos = text.find(')')
                        if pos != -1:
                            text = text[pos + 1:].strip()
                    # 使用原始序号(不使用enumerate的i)
                    num = part.split(')')[0].strip('(')
                    if text:  # 再次确认清理后的文本非空
                        result[f'distractor{num}'] = text
            
            all_results.append(result)            
    
    # 保存合并后的结果
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"处理完成! 共合并 {len(all_results)} 个结果到 {output_file}")
    return all_results

if __name__ == "__main__":
    # 指定输入文件和输出文件
    input_files = [
        "/home/lzx/lib/CoE/infer/pred_val_dg/0.json",
        "/home/lzx/lib/CoE/infer/pred_val_dg/1.json",
        "/home/lzx/lib/CoE/infer/pred_val_dg/2.json",
        "/home/lzx/lib/CoE/infer/pred_val_dg/3.json"
    ]
    
    output_file = "./output/output_dg-scienceqa-coevl-cot-validation.json"
    
    # 处理文件
    process_predictions(input_files, output_file)