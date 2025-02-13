import json
import yaml
import string
import re
import time
import argparse
import numpy as np
from tqdm import tqdm
import os
import evaluate

# 设置 HuggingFace 缓存目录
os.environ['HF_HOME'] = '/home/lzx/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/home/lzx/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/home/lzx/.cache/huggingface'

def normalize(text):
    """标准化文本:小写、去除标点符号、冠词和多余空格"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def evaluate_distractors(test_data, generated_data, metrics):
    """评估生成的干扰项"""
    start_time = time.time()
    bleu4s, rouges = [], []
    
    with tqdm(total=len(test_data), desc="Evaluating") as pbar:
        for test_item, gen_item in zip(test_data, generated_data):
            # 获取参考干扰项和生成的干扰项
            refs = [test_item[f'distractor{i}'] for i in range(1, 4)]
            hyps = [gen_item[f'distractor{i}'] for i in range(1, 4)]
            
            # 规范化文本
            refs = [normalize(ref) for ref in refs]
            hyps = [normalize(hyp) for hyp in hyps]
            
            # # 将三个干扰项合并为一个字符串进行整体评估
            # ref_text = " [SEP] ".join(refs)
            # hyp_text = " [SEP] ".join(hyps)
            # 为每个文本添加序号并拼接成一个没有空格的字符串
            ref_text = "".join([f"({i+1}){ref}" for i, ref in enumerate(refs)])
            hyp_text = "".join([f"({i+1}){hyp}" for i, hyp in enumerate(hyps)])
            # 计算整体指标
            bleu = metrics['bleu'].compute(predictions=[hyp_text], references=[[ref_text]])['score']
            rouge = metrics['rouge'].compute(predictions=[hyp_text], references=[[ref_text]])['rougeL']
            
            bleu4s.append(bleu)
            rouges.append(rouge)
            
            pbar.update(1)

    # 计算平均分数
    scores = {
        'bleu_4': np.mean(bleu4s),
        'rouge': np.mean(rouges) * 100  # 转换为百分比
    }
    
    evaluation_time = time.time() - start_time
    scores['evaluation_time'] = evaluation_time
    
    return scores


def load_metrics():
    """加载评估指标并强制使用本地缓存"""
    import os
    # 设置环境变量强制使用本地缓存
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_EVALUATE_OFFLINE'] = '1'
    
    try:
        print("正在从本地加载评估指标...")
        metrics = {}
        metrics['bleu'] = evaluate.load('sacrebleu', module_type='metric')
        print("✓ BLEU 指标已加载")
        metrics['rouge'] = evaluate.load('rouge', module_type='metric')
        print("✓ ROUGE 指标已加载")
        return metrics
    except Exception as e:
        print(f"加载评估指标时出错: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate distractors")
    parser.add_argument('-d', '--dataset', choices=['lan', 'nat', 'soc','sciqa-text','sciq'], 
                       required=True, help="Dataset type")
    parser.add_argument('-m', '--model', required=True, help="Model name")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], 
                       required=True, help="Prompt type")
    args = parser.parse_args()
    
    # 加载配置
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # 获取文件路径
    file_config = config['files'][args.dataset]
    test_file = file_config['test_file']
    output_file = f"{file_config['output_file']}-{args.model}-{args.prompt}.json"
    results_file = f"{file_config['results_file']}-{args.model}-{args.prompt}-new.json"
    
    # 加载数据
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    with open(output_file, 'r') as f:
        generated_data = json.load(f)
    
    # 加载评估指标
    metrics = load_metrics()
    
    # 评估生成的干扰项
    scores = evaluate_distractors(test_data, generated_data, metrics)
    
    print(f"\n=== Evaluation Results for {args.dataset} ===")
    print(f"BLEU-4: {scores['bleu_4']:.2f}")
    print(f"ROUGE-L: {scores['rouge']:.2f}")
    print(f"Evaluation Time: {scores['evaluation_time']:.2f}s")
    
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=4)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()