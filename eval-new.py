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
import collections 
from utils.utils import clean_output_text
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

def map_subject(subject_str):
    """将原始 subject 字符串映射到分类标签"""
    if not isinstance(subject_str, str):
        return 'UNKNOWN'
    subject_str = subject_str.lower().strip()
    if 'natural science' in subject_str:
        return 'NAT'
    elif 'social science' in subject_str:
        return 'SOC'
    elif 'language science' in subject_str: # 假设语言科学是 'language science'
        return 'LAN'
    else:
        return 'UNKNOWN'

def map_grade(grade_str):
    """将原始 grade 字符串 (如 'grade5') 映射到分类标签"""
    if not isinstance(grade_str, str):
        return 'UNKNOWN'
    match = re.search(r'\d+', grade_str)
    if match:
        try:
            grade_num = int(match.group(0))
            if 1 <= grade_num <= 6:
                return 'G1-6'
            elif 7 <= grade_num <= 12:
                return 'G7-12'
            else:
                return 'UNKNOWN' # 处理范围之外的年级
        except ValueError:
            return 'UNKNOWN' # 处理无法转换为整数的情况
    else:
        return 'UNKNOWN'


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

    # 使用 defaultdict 简化分类分数的累加
    categorized_bleu_sum = collections.defaultdict(lambda: collections.defaultdict(float))
    categorized_rouge_sum = collections.defaultdict(lambda: collections.defaultdict(float))
    categorized_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    # 确定评估数量
    generated_count = len(generated_data)
    total_test_items = len(test_data)

    print(f"\n数据集统计:")
    print(f"数据集类型: {dataset_type}")
    print(f"测试集大小: {len(test_data)}")
    print(f"生成集大小: {generated_count}")
    print(f"将评估前 {min(generated_count, len(test_data))} 个样本")
    
    with tqdm(total=min(generated_count, len(test_data)), desc="Evaluating") as pbar:
        test_items = test_data 
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
                    hyp = clean_output_text(hyp)
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

                # --- 进行分类并累加分类分数 ---
                subject_cat = map_subject(test_item.get('subject'))
                grade_cat = map_grade(test_item.get('grade'))

                if subject_cat != 'UNKNOWN' and grade_cat != 'UNKNOWN':
                    categorized_bleu_sum[subject_cat][grade_cat] += bleu
                    categorized_rouge_sum[subject_cat][grade_cat] += rouge
                    categorized_counts[subject_cat][grade_cat] += 1

            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
                print(f"问题索引: {i}")
                if 'question' in test_item:
                    print(f"问题: {test_item['question'][:100]}...")
                skipped_count += 1
            
            pbar.update(1)
            
    overall_scores = {
        'bleu_4': np.mean(bleu4s) if bleu4s else 0.0,
        'rouge': np.mean(rouges) * 100 if rouges else 0.0, # ROUGE 通常乘以 100
    }

    # --- 计算分类平均分 ---
    categorized_scores = collections.defaultdict(lambda: collections.defaultdict(dict))
    for subj, grades in categorized_counts.items():
        for grade, count in grades.items():
            if count > 0:
                avg_bleu = categorized_bleu_sum[subj][grade] / count
                avg_rouge = (categorized_rouge_sum[subj][grade] / count) * 100 # ROUGE 乘以 100
                categorized_scores[subj][grade] = {
                    'bleu_4': avg_bleu,
                    'rouge': avg_rouge,
                    'count': count
                }

    # --- 组合最终结果 ---
    final_scores = {
        'overall': overall_scores,
        'categorized': dict(categorized_scores), # 转换为普通 dict 以便 JSON 序列化
        'stats': {
            'total_test_samples': total_test_items,
            'generated_samples': generated_count,
            'processed_samples': processed_count,
            'evaluated_samples': len(bleu4s), # 成功计算分数的样本数
            'skipped_samples': skipped_count,
            'evaluation_time': time.time() - start_time
        }
    }

    print(f"\n评估统计信息:")
    print(f"总测试样本数: {final_scores['stats']['total_test_samples']}")
    print(f"生成样本数: {final_scores['stats']['generated_samples']}")
    print(f"处理样本数: {final_scores['stats']['processed_samples']}")
    print(f"成功评估数: {final_scores['stats']['evaluated_samples']}")
    print(f"跳过样本数: {final_scores['stats']['skipped_samples']}")

    return final_scores

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
    print("\n--- Overall Scores ---")
    print(f"BLEU-4: {scores['overall']['bleu_4']:.4f}") # 保持4位小数
    print(f"ROUGE-L: {scores['overall']['rouge']:.4f}") # 保持4位小数

    print("\n--- Categorized Scores ---")
    if scores['categorized']:
        for subj, grades in sorted(scores['categorized'].items()):
            print(f"\nSubject: {subj}")
            for grade, cat_scores in sorted(grades.items()):
                print(f"  Grade: {grade} (Count: {cat_scores['count']})")
                print(f"    BLEU-4: {cat_scores['bleu_4']:.4f}") # 保持4位小数
                print(f"    ROUGE-L: {cat_scores['rouge']:.4f}") # 保持4位小数
    else:
        print("No categorized scores available (check data or categories).")

    print(f"\nEvaluation Time: {scores['stats']['evaluation_time']:.2f}s")
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=4, ensure_ascii=False) 
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()