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
    """评估生成的干扰项
    Args:
        test_data: 有序的测试数据列表
        generated_data: 有序的生成数据列表
        metrics: 评估指标字典
    """
    start_time = time.time()
    bleu4s, rouges = [], []
    skipped_count = 0
    processed_count = 0
    
    # 确定评估数量
    generated_count = len(generated_data)
    is_hf_dataset = hasattr(test_data, '__len__') and hasattr(test_data, 'features')
    
    print(f"\n数据集统计:")
    print(f"数据集类型: {'HuggingFace Dataset' if is_hf_dataset else 'Local JSON'}")
    print(f"测试集大小: {len(test_data)}")
    print(f"生成集大小: {generated_count}")
    print(f"将评估前 {generated_count} 个样本")
    
    with tqdm(total=generated_count, desc="Evaluating") as pbar:
        # 对于HF数据集和本地JSON使用统一的迭代方式
        test_items = test_data if not is_hf_dataset else test_data
        for test_item, gen_item in zip(test_items, generated_data):
            if processed_count >= generated_count:
                break
            try:
                processed_count += 1
                
                # 对于HF数据集，需要特殊处理干扰项的提取
                if is_hf_dataset:
                    correct_answer_index = test_item['answer']
                    all_choices = test_item['choices']
                    refs = [choice for i, choice in enumerate(all_choices) if i != correct_answer_index]
                else:
                    # 原有的本地JSON处理逻辑
                    distractor_count = len([k for k in test_item.keys() if k.startswith('distractor')])
                    refs = [test_item[f'distractor{i}'] for i in range(1, distractor_count + 1)]
                
                if not refs:
                    print(f"警告：问题 '{test_item['question'][:50]}...' 没有干扰项")
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                # 获取生成的干扰项
                hyps = [gen_item.get(f'distractor{i}', '') for i in range(1, len(refs) + 1)]
                
                # 规范化和评估逻辑保持不变
                refs = [normalize(ref) for ref in refs]
                hyps = [normalize(hyp) for hyp in hyps]
                
                ref_text = " [SEP] ".join(refs)
                hyp_text = " [SEP] ".join(hyps)
                
                bleu = metrics['bleu'].compute(predictions=[hyp_text], references=[[ref_text]])['score']
                rouge = metrics['rouge'].compute(predictions=[hyp_text], references=[[ref_text]])['rougeL']
                
                bleu4s.append(bleu)
                rouges.append(rouge)
            
            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
                print(f"问题: {test_item['question'][:100]}...")
                skipped_count += 1
            
            pbar.update(1)

    # 计算评估结果
    scores = {
        'bleu_4': np.mean(bleu4s) if bleu4s else 0.0,
        'rouge': np.mean(rouges) * 100 if rouges else 0.0,
        'total_samples': len(test_data),
        'generated_samples': generated_count,
        'processed_samples': processed_count,
        'evaluated_samples': len(bleu4s),
        'skipped_samples': skipped_count,
        'evaluation_time': time.time() - start_time
    }
    
    print(f"\n评估统计信息:")
    print(f"总样本数: {scores['total_samples']}")
    print(f"生成样本数: {scores['generated_samples']}")
    print(f"处理样本数: {scores['processed_samples']}")
    print(f"成功评估数: {scores['evaluated_samples']}")
    print(f"跳过样本数: {scores['skipped_samples']}")
    
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
    parser.add_argument('-d', '--dataset', required=True, help="Dataset name or path")
    parser.add_argument('-m', '--model', required=True, help="Model name")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], 
                       required=True, help="Prompt type")
    parser.add_argument('--split', type=str, default='test', help="Dataset split for HF datasets")
    args = parser.parse_args()
    
    # 加载配置
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
     # 获取数据集配置
    file_config = config['files'].get(args.dataset, {})
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    
    # 确定输出文件路径
    output_file = f"{file_config.get('output_file', f'./output/output_dg-{args.dataset}')}-{args.model}-{args.prompt}.json"
    results_file = f"{file_config.get('results_file', f'./evaluation/{args.dataset}')}-{args.model}-{args.prompt}-new.json"
    
    # 加载测试数据
    if 'test_file' in file_config:
        # 本地JSON文件
        with open(file_config['test_file'], 'r') as f:
            test_data = json.load(f)
    else:
        # HuggingFace数据集
        from datasets import load_dataset
        test_data = load_dataset(dataset_name, split=args.split)

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