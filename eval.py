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
import collections

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

def evaluate_distractors(test_data, generated_data, metrics, include_context=False):
    """
    评估生成的干扰项, 并按 subject 和 grade 分类计算分数。

    Args:
        test_data: 有序的测试数据列表 (包含 subject 和 grade)。
        generated_data: 有序的生成数据列表。
        metrics: 评估指标字典。
        include_context (bool): 是否将问题和答案纳入评估上下文。

    Returns:
        dict: 包含总体分数和按类别划分的分数。
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
    is_hf_dataset = hasattr(test_data, '__len__') and hasattr(test_data, 'features')
    
    print(f"\n数据集统计:")
    print(f"数据集类型: {'HuggingFace Dataset' if is_hf_dataset else 'Local JSON'}")
    total_test_items = len(test_data)
    print(f"测试集大小: {total_test_items}")
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

                # 检查生成数据中的distractor1是否为空
                if not gen_item.get('distractor1'):
                    print(f"警告：生成的distractor1为空，跳过该样本评估")
                    skipped_count += 1
                    pbar.update(1)
                    continue          

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

                # 如果需要包含上下文
                if include_context:
                    question_norm = normalize(test_item.get('question', ''))
                    answer_norm = normalize(test_item.get('correct_answer', '')) # 假设正确答案字段是 'correct_answer'
                    context_prefix = f"{question_norm} [ANS] {answer_norm} [SEP] "
                    ref_text = context_prefix + ref_text

                bleu = metrics['bleu'].compute(predictions=[hyp_text], references=[[ref_text]])['score']
                rouge = metrics['rouge'].compute(predictions=[hyp_text], references=[[ref_text]])['rougeL']
                
                bleu4s.append(bleu)
                rouges.append(rouge)

                # --- 进行分类并累加分类分数 ---
                subject_cat = map_subject(test_item.get('subject'))
                grade_cat = map_grade(test_item.get('grade'))

                # 只有当分类有效时才记录
                if subject_cat != 'UNKNOWN' and grade_cat != 'UNKNOWN':
                    categorized_bleu_sum[subject_cat][grade_cat] += bleu
                    categorized_rouge_sum[subject_cat][grade_cat] += rouge
                    categorized_counts[subject_cat][grade_cat] += 1
                               
            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
                print(f"问题: {test_item['question'][:100]}...")
                skipped_count += 1
            
            pbar.update(1)

    # --- 计算总体平均分 ---
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

    # --- 打印评估统计信息 ---
    print(f"\n评估统计信息:")
    print(f"总测试样本数: {final_scores['stats']['total_test_samples']}")
    print(f"生成样本数: {final_scores['stats']['generated_samples']}")
    print(f"处理样本数: {final_scores['stats']['processed_samples']}")
    print(f"成功评估数: {final_scores['stats']['evaluated_samples']}")
    print(f"跳过样本数: {final_scores['stats']['skipped_samples']}")

    return final_scores


def load_metrics():
    
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
    parser.add_argument('-v', '--multimodal', type=int, choices=[0, 1], default=1,
                       help="是否评估多模态模式生成的文件 (0: 否/text-only, 1: 是/multimodal)")
    parser.add_argument('-w', '--way', choices=['api', 'local'], 
                       default='api', help="Running way: api or local")
    parser.add_argument('-c', '--include_context', type=int, choices=[0, 1], default=0,
                       help="是否在评估时包含问题和答案上下文 (0: 否, 1: 是)")
    args = parser.parse_args()
    
    # 加载配置
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
     # 获取数据集配置
    file_config = config['files'].get(args.dataset, {})
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    use_multimodal_flag = bool(args.multimodal)
    include_context_flag = bool(args.include_context) # 获取上下文标志
    print(f"评估模式: {'多模态' if use_multimodal_flag else '纯文本'}")
    print(f"包含上下文进行评估: {'是' if include_context_flag else '否'}")
    
    # 确定输出文件路径
    model_mode_suffix = "-text" if not use_multimodal_flag else ""
    way_suffix = "-local" if args.way == 'local' else ""
    context_suffix = "-ctx" if include_context_flag else "" 
    output_file = f"{file_config.get('output_file', f'./output/output_dg-{args.dataset}')}-{args.model}{model_mode_suffix}-{args.prompt}-{args.split}{way_suffix}.json"
    results_file = f"{file_config.get('results_file', f'./evaluation/{args.dataset}')}-{args.model}{model_mode_suffix}-{args.prompt}-{args.split}{way_suffix}{context_suffix}.json"
    
    # 加载测试数据
    if 'test_file' in file_config:
        # 本地JSON文件
        with open(file_config['test_file'], 'r') as f:
            test_data = json.load(f)
    else:
        # HuggingFace数据集
        from datasets import load_dataset
        test_data = load_dataset(dataset_name, split=args.split)

    try:
        with open(output_file, 'r') as f:
            generated_data = json.load(f)
            print(f"已加载生成数据: {output_file}")
    except FileNotFoundError:
        print(f"\n\033[91m错误: 找不到生成的输出文件。\033[0m")
        print("请确保 run.py 已成功运行，并且参数与 eval.py 匹配 (特别是 --model 和 --multimodal)。")
        return # 找不到文件则退出
    except json.JSONDecodeError:
        print(f"\n\033[91m错误: 文件 '{output_file}' 不是有效的 JSON 文件。\033[0m")
    
    # 加载评估指标
    metrics = load_metrics()
    
    # 评估生成的干扰项
    scores = evaluate_distractors(test_data, generated_data, metrics, include_context=include_context_flag)
    
    # --- 打印结果 (包括分类结果) ---
    print(f"\n=== Evaluation Results for {args.dataset} ({'with context' if include_context_flag else 'without context'}) ===")
    print("\n--- Overall Scores ---")
    print(f"BLEU-4: {scores['overall']['bleu_4']:.2f}")
    print(f"ROUGE-L: {scores['overall']['rouge']:.2f}")

    print("\n--- Categorized Scores ---")
    if scores['categorized']:
        for subj, grades in sorted(scores['categorized'].items()):
            print(f"\nSubject: {subj}")
            for grade, cat_scores in sorted(grades.items()):
                print(f"  Grade: {grade} (Count: {cat_scores['count']})")
                print(f"    BLEU-4: {cat_scores['bleu_4']:.2f}")
                print(f"    ROUGE-L: {cat_scores['rouge']:.2f}")
    else:
        print("No categorized scores available (check data or categories).")

    print(f"\nEvaluation Time: {scores['stats']['evaluation_time']:.2f}s")
    
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()