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
from datasets import load_dataset

def normalize(text):
    """标准化文本:小写、去除标点符号、冠词和多余空格"""
    if not text:
        return ""
        
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

def clean_output_text(output_text):
    """清理输出文本以提取干扰项内容"""
    if not output_text:
        return ""
        
    # 移除Example部分
    output_text = re.sub(r'Example \d+:.*?(?=\(\d+\)|$)', '', output_text, flags=re.DOTALL)
    
    # 提取数字编号的干扰项
    items = re.findall(r'\((\d+)\)\s*(.+?)(?=\(\d+\)|$)', output_text, re.DOTALL)
    
    # 如果没有找到编号干扰项，尝试其他格式
    if not items:
        items = re.findall(r'^\d+\.\s*(.+?)$', output_text, re.MULTILINE)
        return ' '.join(items)
    
    # 返回干扰项文本
    return ' '.join(item[1].strip() for item in items)

def get_reference_distractors(item, dataset_type="hf"):
    """获取参考干扰项"""
    distractors = []
    
    if dataset_type == "hf":
        # HuggingFace数据集格式
        if 'choices' in item and 'answer' in item:
            correct_index = item['answer'] 
            distractors = [
                choice for i, choice in enumerate(item['choices']) 
                if i != correct_index
            ]
    else:
        # 本地JSON格式
        if isinstance(item, dict):
            for key in item:
                if key.startswith('distractor') and item[key]:
                    distractors.append(item[key])
    
    return ' '.join(d for d in distractors if d)

def evaluate_distractors(test_data, generated_data, metrics, dataset_type="hf"):
    """评估生成的干扰项
    Args:
        test_data: 有序的测试数据列表
        generated_data: 有序的生成数据列表 (使用output字段)
        metrics: 评估指标字典
        dataset_type: 'hf' 表示HuggingFace数据集, 'local' 表示本地JSON
    """
    start_time = time.time()
    bleu4s, rouges = [], []
    skipped_count = 0
    processed_count = 0
    
    # 确定评估数量
    generated_count = len(generated_data)
    
    print(f"\n数据集统计:")
    print(f"数据集类型: {dataset_type}")
    print(f"测试集大小: {len(test_data)}")
    print(f"生成集大小: {generated_count}")
    print(f"将评估前 {min(generated_count, len(test_data))} 个样本")
    
    with tqdm(total=min(generated_count, len(test_data)), desc="Evaluating") as pbar:
        for i, gen_item in enumerate(generated_data):
            if i >= len(test_data) or i >= generated_count:
                break
                
            test_item = test_data[i]
            try:
                processed_count += 1
                
                # 获取参考干扰项
                reference = get_reference_distractors(test_item, dataset_type)
                
                # 获取生成的干扰项输出
                if 'output' in gen_item:
                    hyp = gen_item['output']
                else:
                    hyp = get_reference_distractors(gen_item, "local")
                
                # 调试信息 - 打印前3个样本的比较内容
                if i < 3:
                    print(f"\n===== Sample {i} =====")
                    print(f"问题: {test_item.get('question', '')[:100]}...")
                    print(f"参考干扰项: {reference[:100]}...")
                    print(f"生成干扰项: {hyp[:100]}...")
                    print(f"规范化后参考: {normalize(reference)[:100]}...")
                    print(f"规范化后生成: {normalize(hyp)[:100]}...")
                
                # 跳过没有参考干扰项的样本
                if not reference.strip():
                    print(f"警告: 样本 {i} 没有参考干扰项")
                    skipped_count += 1
                    pbar.update(1)
                    continue
                    
                # 规范化文本
                ref_normalized = normalize(reference)
                hyp_normalized = normalize(hyp)
                
                if not hyp_normalized:
                    print(f"警告: 样本 {i} 没有生成干扰项")
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                # 计算评估指标
                bleu = metrics['bleu'].compute(predictions=[hyp_normalized], references=[[ref_normalized]])['score']
                rouge = metrics['rouge'].compute(predictions=[hyp_normalized], references=[[ref_normalized]])['rougeL']
                
                # 保存分数
                bleu4s.append(bleu)
                rouges.append(rouge)
            
            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
                print(f"问题索引: {i}")
                if 'question' in test_item:
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
    try:
        print("正在加载评估指标...")
        metrics = {}
        metrics['bleu'] = evaluate.load('sacrebleu')
        print("✓ BLEU 指标已加载")
        metrics['rouge'] = evaluate.load('rouge')
        print("✓ ROUGE 指标已加载")
        return metrics
    except Exception as e:
        print(f"加载评估指标时出错: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate distractors")
    parser.add_argument('-d', '--dataset', required=True, help="Dataset name or path")
    parser.add_argument('-i', '--input', required=True, help="Generated output JSON file path")
    parser.add_argument('-o', '--output', default=None, help="Output evaluation results JSON path")
    parser.add_argument('--split', type=str, default='test', help="Dataset split for HF datasets")
    args = parser.parse_args()
    
    # 加载配置
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # 获取数据集配置
    file_config = config['files'].get(args.dataset, {})
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    
    # 确定输出文件路径
    results_file = args.output
    if not results_file:
        results_file = f"./evaluation/{args.dataset}-{os.path.basename(args.input).split('.')[0]}-eval.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # 加载测试数据
    dataset_type = "local"
    if 'test_file' in file_config and os.path.exists(file_config['test_file']):
        # 本地JSON文件
        with open(file_config['test_file'], 'r') as f:
            test_data = json.load(f)
            print(f"已从本地文件加载测试数据: {file_config['test_file']}")
    else:
        # HuggingFace数据集
        dataset_type = "hf"
        test_data = load_dataset(dataset_name, split=args.split)
        print(f"已从HuggingFace加载测试数据: {dataset_name}({args.split})")

    # 加载生成的数据
    with open(args.input, 'r') as f:
        generated_data = json.load(f)
        print(f"已加载生成数据: {args.input}，共 {len(generated_data)} 条")
    
    # 加载评估指标
    metrics = load_metrics()
    
    # 评估生成的干扰项
    scores = evaluate_distractors(test_data, generated_data, metrics, dataset_type)
    
    print(f"\n=== Evaluation Results for {args.dataset} ===")
    print(f"BLEU-4: {scores['bleu_4']:.4f}")
    print(f"ROUGE-L: {scores['rouge']:.4f}")
    print(f"Evaluation Time: {scores['evaluation_time']:.2f}s")
    
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=4)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()