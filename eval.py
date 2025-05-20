import json
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm
import os
import evaluate
import collections
from utils.auto_metrics import calc_novelty, calc_diversity
from utils.stat_utils import map_subject, map_grade, calculate_category_scores
from utils.utils import preprocess_generation_data, normalize

def evaluate_distractors(test_data, generated_data, metrics, include_context=False, train_set=False):
    """
    评估生成的干扰项, 并按 subject、grade 分类计算分数。
    支持BLEU/ROUGE、自动指标（新颖性/多样性）。

    Args:
        test_data: 测试数据集
        generated_data: 生成的干扰项数据
        metrics: 评估指标字典
        include_context: 是否在评估时包含问题和答案上下文
        train_set: 是否加载训练集用于新颖性评估 
    """
    start_time = time.time()
    bleu4s, rouges = [], []
    novelty_scores, diversity_scores = [], []
    skipped_count = 0
    processed_count = 0
    
    # 按科目直接统计
    subject_bleu_sum = collections.defaultdict(float)
    subject_rouge_sum = collections.defaultdict(float)
    subject_novelty_sum = collections.defaultdict(float)
    subject_diversity_sum = collections.defaultdict(float)
    subject_counts = collections.defaultdict(int)
    
    # 按年级直接统计
    grade_bleu_sum = collections.defaultdict(float)
    grade_rouge_sum = collections.defaultdict(float)
    grade_novelty_sum = collections.defaultdict(float)
    grade_diversity_sum = collections.defaultdict(float)
    grade_counts = collections.defaultdict(int)
    
    generated_count = len(generated_data)
    is_hf_dataset = hasattr(test_data, '__len__') and hasattr(test_data, 'features')
    
    # 精简数据集统计打印
    print(f"\n数据集类型: {'HuggingFace Dataset' if is_hf_dataset else 'Local JSON'}")
    total_test_items = len(test_data)
    print(f"测试集: {total_test_items} 样本, 生成集: {generated_count} 样本")

    # 自动读取训练集
    train_set_data = None
    if train_set:
        # 读取config.yaml中的train配置
        config_path = './config/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            # 获取训练集配置，与测试集配置读取方式保持一致
            train_dataset = config['dataset_names'].get('scienceqa', 'scienceqa')
            train_file_config = config['files'].get(train_dataset, {})
            
            # 使用与测试集相同的方式读取训练集
            if 'train_file' in train_file_config:
                # 从本地JSON文件加载
                train_file = train_file_config['train_file']
                if os.path.exists(train_file):
                    with open(train_file, 'r') as f:
                        train_set_data = json.load(f)
                    print(f"已加载训练集文件: {train_file} ({len(train_set_data)} 样本)")
                else:
                    print(f"未找到训练集文件: {train_file}")
            else:
                # 从HuggingFace datasets加载
                try:
                    train_dataset_name = config['dataset_names'].get(train_dataset, train_dataset)
                    from datasets import load_dataset
                    train_set_data = load_dataset(train_dataset_name, split='train')
                    print(f"已加载HuggingFace训练集: {train_dataset_name} ({len(train_set_data)} 样本)")
                except Exception as e:
                    print(f"无法加载HuggingFace训练集: {e}")
        else:
            print("未找到config.yaml配置文件")

    # =====================
    # 1. 主评估循环（BLEU/ROUGE/分组统计等）
    # =====================
    with tqdm(total=generated_count, desc="Evaluating") as pbar:
        test_items = test_data if not is_hf_dataset else test_data
        for test_item, gen_item in zip(test_items, generated_data):
            if processed_count >= generated_count:
                break
            try:
                processed_count += 1
                # 1.1 检查生成数据有效性
                if not gen_item.get('distractor1'):
                    skipped_count += 1
                    pbar.update(1)
                    continue          
                # 1.2 获取参考干扰项
                if is_hf_dataset:
                    correct_answer_index = test_item['answer']
                    all_choices = test_item['choices']
                    refs = [choice for i, choice in enumerate(all_choices) if i != correct_answer_index]
                else:
                    distractor_count = len([k for k in test_item.keys() if k.startswith('distractor')])
                    refs = [test_item[f'distractor{i}'] for i in range(1, distractor_count + 1)]
                if not refs:
                    skipped_count += 1
                    pbar.update(1)
                    continue
                # 1.3 获取生成的干扰项
                hyps = [gen_item.get(f'distractor{i}', '') for i in range(1, len(refs) + 1)]
                # 1.4 规范化
                refs = [normalize(ref) for ref in refs]
                hyps = [normalize(hyp) for hyp in hyps]
                ref_text = " [SEP] ".join(refs)
                hyp_text = " [SEP] ".join(hyps)
                # 1.5 上下文拼接
                if include_context:
                    question_norm = normalize(test_item.get('question', ''))
                    answer_norm = normalize(test_item.get('correct_answer', ''))
                    context_prefix = f"{question_norm} [ANS] {answer_norm} [SEP] "
                    ref_text = context_prefix + ref_text
                # 1.6 BLEU/ROUGE计算
                bleu = metrics['bleu'].compute(predictions=[hyp_text], references=[[ref_text]])['score']
                rouge = metrics['rouge'].compute(predictions=[hyp_text], references=[[ref_text]])['rougeL']
                bleu4s.append(bleu)
                rouges.append(rouge)
                # 1.7 自动指标（新颖性/多样性）
                item_novelty = 0.0
                item_diversity = 0.0
                if train_set and train_set_data:
                    # 计算每个干扰项的新颖性并取平均
                    hyp_novelty_scores = []
                    for hyp in hyps:
                        if hasattr(train_set_data, 'features'):  # HuggingFace数据集
                            novelty_score = calc_novelty(hyp, train_set_data, is_hf_dataset=True)
                        else:  # 普通JSON数据
                            novelty_score = calc_novelty(hyp, train_set_data)
                        hyp_novelty_scores.append(novelty_score)
                    
                    # 整个问题的新颖性是所有干扰项新颖性的平均
                    item_novelty = sum(hyp_novelty_scores) / len(hyp_novelty_scores) if hyp_novelty_scores else 0.0
                    novelty_scores.append(item_novelty)
                    
                    # 计算多样性
                    item_diversity = calc_diversity(hyps)
                    diversity_scores.append(item_diversity)
                
                # 1.8 按科目和年级分别统计 - 不再使用嵌套结构
                subject_cat = map_subject(test_item.get('subject'))
                grade_cat = map_grade(test_item.get('grade'))
                
                # 科目统计
                if subject_cat != 'UNKNOWN':
                    subject_bleu_sum[subject_cat] += bleu
                    subject_rouge_sum[subject_cat] += rouge
                    subject_counts[subject_cat] += 1
                    if train_set and train_set_data:
                        subject_novelty_sum[subject_cat] += item_novelty
                        subject_diversity_sum[subject_cat] += item_diversity
                
                # 年级统计
                if grade_cat != 'UNKNOWN':
                    grade_bleu_sum[grade_cat] += bleu
                    grade_rouge_sum[grade_cat] += rouge
                    grade_counts[grade_cat] += 1
                    if train_set and train_set_data:
                        grade_novelty_sum[grade_cat] += item_novelty
                        grade_diversity_sum[grade_cat] += item_diversity
                
            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
                print(f"问题: {test_item['question'][:100]}...")
                skipped_count += 1
            pbar.update(1)

    # =====================
    # 2. 统计与输出
    # =====================
    overall_scores = {
        'bleu_4': np.mean(bleu4s) if bleu4s else 0.0,
        'rouge': np.mean(rouges) * 100 if rouges else 0.0
    }
    
    # 2.1 自动指标总体统计
    auto_metrics = {}
    if train_set and novelty_scores:
        auto_metrics.update({
            'novelty': float(np.mean(novelty_scores)),
            'diversity': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
            'novelty_reference': '与train_set对比'
        })
    else:
        auto_metrics.update({
            'novelty': 0.0,
            'diversity': 0.0,
            'novelty_reference': '未指定参考集'
        })
    
    # 2.2 直接计算科目统计 - 使用封装的函数
    subject_metric_sums = {
        'bleu_4': subject_bleu_sum,
        'rouge': subject_rouge_sum
    }
    # 如果有新颖性/多样性指标，添加到指标字典
    if train_set and novelty_scores:
        subject_metric_sums.update({
            'novelty': subject_novelty_sum,
            'diversity': subject_diversity_sum
        })
    
    # 使用封装的函数计算科目分类统计
    subject_scores = calculate_category_scores(subject_metric_sums, subject_counts)
    
    # 2.3 直接计算年级统计 - 使用封装的函数
    grade_metric_sums = {
        'bleu_4': grade_bleu_sum,
        'rouge': grade_rouge_sum
    }
    # 如果有新颖性/多样性指标，添加到指标字典
    if train_set and novelty_scores:
        grade_metric_sums.update({
            'novelty': grade_novelty_sum,
            'diversity': grade_diversity_sum
        })
    
    # 使用封装的函数计算年级分类统计
    grade_scores = calculate_category_scores(grade_metric_sums, grade_counts)
    
    # 2.4 组合最终结果
    final_scores = {
        'overall': overall_scores,
        'subject': subject_scores,
        'grade': grade_scores,
        'auto_metrics': auto_metrics,
        'stats': {
            'total_test_samples': total_test_items,
            'generated_samples': generated_count,
            'processed_samples': processed_count,
            'evaluated_samples': len(bleu4s),
            'skipped_samples': skipped_count,
            'evaluation_time': time.time() - start_time,
        }
    }
    
    # 2.5 打印评估统计信息（精简版）
    print(f"\n评估统计: 处理{final_scores['stats']['processed_samples']}样本, 成功{final_scores['stats']['evaluated_samples']}样本, 跳过{final_scores['stats']['skipped_samples']}样本")
    if train_set and novelty_scores:
        print(f"自动指标: 新颖性 {auto_metrics['novelty']:.4f}, 多样性 {auto_metrics['diversity']:.4f}")

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
    parser.add_argument('--train_set', action='store_true', help="加载训练集用于新颖性计算")
    args = parser.parse_args()
    
    # 加载配置
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
     # 获取数据集配置
    file_config = config['files'].get(args.dataset, {})
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    use_multimodal_flag = bool(args.multimodal)
    include_context_flag = bool(args.include_context)
    print(f"评估配置: 模式={'多模态' if use_multimodal_flag else '纯文本'}, 上下文={'包含' if include_context_flag else '不包含'}")
    
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
            if any('output' in item for item in generated_data):
                print("检测到output格式的生成数据，进行预处理...")
                generated_data = preprocess_generation_data(generated_data)
                print(f"预处理完成，共处理 {len(generated_data)} 个样本")

    except FileNotFoundError:
        print(f"\n\033[91m错误: 找不到生成的输出文件。\033[0m")
        print("请确保 run.py 已成功运行，并且参数与 eval.py 匹配 (特别是 --model 和 --multimodal)。")
        return # 找不到文件则退出
    except json.JSONDecodeError:
        print(f"\n\033[91m错误: 文件 '{output_file}' 不是有效的 JSON 文件。\033[0m")
        return
    
    # 加载评估指标
    metrics = load_metrics()
    
    # 评估生成的干扰项
    scores = evaluate_distractors(
        test_data, 
        generated_data, 
        metrics, 
        include_context=include_context_flag,
        train_set=args.train_set)
    
    # --- 打印结果 (精简版) ---
    print(f"\n=== {args.dataset} 评估结果 ({args.model}, {'上下文' if include_context_flag else '无上下文'}) ===")
    print(f"BLEU-4: {scores['overall']['bleu_4']:.2f}, ROUGE-L: {scores['overall']['rouge']:.2f}")

    # 按科目打印BLEU/ROUGE和自动指标
    if scores.get('subject'):
        print("\n--- 科目级别指标 ---")
        for subj, subj_scores in sorted(scores['subject'].items()):
            output = f"{subj} (样本数: {subj_scores['count']}): BLEU-4={subj_scores['bleu_4']:.2f}, ROUGE-L={subj_scores['rouge']:.2f}"
            if 'novelty' in subj_scores:
                output += f", 新颖性={subj_scores['novelty']:.4f}"
            if 'diversity' in subj_scores:
                output += f", 多样性={subj_scores['diversity']:.4f}"
            print(output)
    
    # 按年级打印BLEU/ROUGE和自动指标
    if scores.get('grade'):
        print("\n--- 年级级别指标 ---")
        for grade, grade_scores in sorted(scores['grade'].items()):
            output = f"{grade} (样本数: {grade_scores['count']}): BLEU-4={grade_scores['bleu_4']:.2f}, ROUGE-L={grade_scores['rouge']:.2f}"
            if 'novelty' in grade_scores:
                output += f", 新颖性={grade_scores['novelty']:.4f}"
            if 'diversity' in grade_scores:
                output += f", 多样性={grade_scores['diversity']:.4f}"
            print(output)
    
    # 打印总体统计
    print(f"\n总体统计:")
    print(f"评估用时: {scores['stats']['evaluation_time']:.2f}秒")
    print(f"结果已保存至: {results_file}")
    
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()