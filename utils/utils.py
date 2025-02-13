import re
import csv
import re
import pandas as pd
import ast
import json
import numpy as np
import torch
import random

def format_question_output(response):
    question_match = re.search(r'Question:\s*(.*)', response)
    answer_match = re.search(r'Answer:\s*(.*)', response)
    question = question_match.group(1).strip() if question_match else ''
    answer = answer_match.group(1).strip() if answer_match else ''
    return {'question': question, 'correct_answer': answer}

def format_rationale_output(response, format_type="rule_format"):
    """
    格式化推理输出
    Args:
        response: 原始响应文本
        format_type: 格式类型，可选值为 "rule_format" 或 "cot_format"
    Returns:
        dict 或 str: 根据format_type返回不同格式的结果
    """
    # 清理响应文本，去除所有的 **
    cleaned_response = response.replace('**', '')
    
    if format_type == "cot_format":
        # 只提取 Inference: 后的内容
        inference_match = re.search(r'Inference:\s*(.*?)(?=\n|$)', cleaned_response, re.DOTALL)
        return inference_match.group(1).strip() if inference_match else ''
    else:
        # 原有的rule_format处理逻辑
        explanation_match = re.search(r'Explanation:\s*(.*?)\n', cleaned_response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ''
        
        incorrect_inferences = re.findall(r'Incorrect Inference \d+ \((.*?)\):\s*(.*?)(?=\nIncorrect Inference \d+ \(|$|\n\n)', cleaned_response, re.DOTALL)
        incorrect_inferences_combined = ' '.join([f'Incorrect Inference {i+1} ({principle.strip()}): {inference.strip()}' for i, (principle, inference) in enumerate(incorrect_inferences)])
        
        return {'explanation': explanation, 'incorrect_inferences': incorrect_inferences_combined}

def format_distractor_output(text: str) -> dict:
    output = {}
    # 去除所有的 **
    cleaned_text = text.replace('**', '')
    
    # 匹配干扰项
    distractor_pattern = re.compile(r'Distractor\d:\s*(.*?)\s*(?=\n|$)')
    matches = distractor_pattern.findall(cleaned_text)
    
    for i, match in enumerate(matches, 1):
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

def clean_string(string):
    string = string.lower()

    # Standardize symbols
    string = string.replace("\\%", "%")
    string = string.replace("...", "\\ldots")
    string = string.replace('÷', '\\div')
    string = string.replace('≥', '\\geq')
    string = string.replace('≤', '\\leq')
    string = string.replace('≠', '\\neq')
    string = string.replace('≈', '\\approx')
    string = string.replace('δ', '\\delta')
    string = string.replace('|', '\\vert')

    # Remove math environment indicators
    string = string.replace("$", "")
    string = string.replace("\\[", "")
    string = string.replace("\\]", "")
    string = string.replace("\\(", "")
    string = string.replace("\\)", "")

    # convert / and \div fractions to \frac
    string = re.sub(r"([\d\.]+)\s*(/|\\div)\s*([\d\.]+)", r"\\frac{\g<1>}{\g<3>}", string) 
    # convert x to \times
    string = re.sub(r'\s*×\s*', r' \\times ', string)
    # convert √ to \\sqrt{}
    string = re.sub(r'√', r'\\sqrt', string) 
    # convert 2 cm to 2 \mathrm{~cm}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*cm',  r'\1 \\mathrm{~cm}', string)
    # convert 2 m to 2 \mathrm{~m}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*m',  r'\1 \\mathrm{~m}', string)
    # convert 2 km to 2 mathrm{~km}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*km',  r'\1 \\mathrm{~km}', string)

    # convert p^2 to p^{2}
    string = re.sub(r'([a-zA-Z])\^(\d+)', r'\1^{\2}', string)

    # remove hyphen between words
    string = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1\2', string)

    string = string.replace('\\mathrm{~m}athrm{~cm}', '\\mathrm{~cm}')
    string = string.replace('\\mathrm{~m}ore', 'more')
    string = string.replace(' ', '')
    string = string.strip()

    return string

