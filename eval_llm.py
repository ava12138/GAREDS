#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大模型干扰项评估脚本 - 专注于LLM对干扰项的多维度评估
包括：评委评测（大模型能否选出正确答案）和多维度打分（四个维度）
"""

import json
import yaml
import argparse
import os
import time
import collections
from openai import OpenAI
from utils.llm_eval_utils import batch_llm_mcq_judge, batch_llm_multi_dim_score
from utils.stat_utils import map_subject, map_grade, calculate_category_scores
from utils.utils import preprocess_generation_data

def load_llm_config():
    """加载 API 配置"""
    try:
        with open('./config/api.yaml', 'r') as file:
            api_config = yaml.safe_load(file)
        return api_config
    except FileNotFoundError as e:
        print(f"\n\033[91m错误: 找不到 LLM 配置文件: {e}\033[0m")
        print("请确保 './config/api.yaml' 文件存在。")
        return None
    except Exception as e:
        print(f"\n\033[91m加载 LLM 配置时出错: {e}\033[0m")
        return None


def initialize_llm_client(api_config, model_key='deepseek'):
    """初始化 OpenAI 兼容的 API 客户端"""
    if not api_config:
        return None, None
    try:
        api_key = api_config.get('api_key')
        if not api_key:
            print("\n\033[91m错误: api.yaml 中未找到 'api_key'。\033[0m")
            return None, None
        if model_key not in api_config.get('model', {}):
            print(f"\n\033[91m错误: 模型键 '{model_key}' 在 api.yaml 的 'model' 配置中未找到。\033[0m")
            return None, None

        api_model = api_config['model'][model_key]
        base_url = api_config.get('base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"LLM 评估客户端已初始化，使用模型: {api_model}, EndPoint: {base_url}")
        return client, api_model
    except Exception as e:
        print(f"\n\033[91m初始化 LLM 客户端时出错: {e}\033[0m")
        return None, None

def evaluate_with_llm(test_data, generated_data, llm_client, llm_model, max_samples=None, token_budget=None, 
                     token_log_file=None, save_dir=None, batch_size=10, run_judge=True, run_dim_score=True):
    """
    使用大模型对干扰项进行评估
    
    Args:
        test_data: 测试数据
        generated_data: 生成的干扰项数据
        llm_client: 大模型客户端
        llm_model: 大模型名称
        max_samples: 最大样本数限制
        token_budget: token预算限制
        token_log_file: token消耗日志文件
        save_dir: 保存断点和结果的目录
        batch_size: 批次保存大小
        run_judge: 是否运行评委测试
        run_dim_score: 是否运行多维度打分
        
    Returns:
        Dict: 评估结果
    """
    start_time = time.time()
    judge_correct_count, judge_total = 0, 0
    judge_results = []
    input_tokens_judge, output_tokens_judge = 0, 0
    
    llm_batch_results = []
    input_tokens_dim, output_tokens_dim = 0, 0
    
    total_input_tokens, total_output_tokens = 0, 0
    
    # 记录样本的科目和年级信息
    sample_subjects = []
    sample_grades = []
    
    # 保存路径设置
    judge_save_path = os.path.join(save_dir, "judge_checkpoint.json") if save_dir else None
    dim_save_path = os.path.join(save_dir, "dim_score_checkpoint.json") if save_dir else None
    
    # 1. 评委评测（大模型能否在干扰项中选出正确答案）
    if run_judge:
        print(f"\n=== 第一阶段: 评委评测 (判断大模型是否能选出正确答案) ===")
        
        # 准备评委评测数据
        qa_list = []
        
        for test_item, gen_item in zip(test_data, generated_data):
            question = gen_item.get('question', '')
            correct_answer = gen_item.get('correct_answer', '')
            distractors = [gen_item.get(f'distractor{i}', '') for i in range(1, 4) if gen_item.get(f'distractor{i}', '')]
            if not distractors or not correct_answer:
                continue
            
            qa_list.append((question, correct_answer, distractors))
            subject_cat = map_subject(test_item.get('subject'))
            grade_cat = map_grade(test_item.get('grade'))
            sample_subjects.append(subject_cat)
            sample_grades.append(grade_cat)
        
        print(f"共找到 {len(qa_list)} 个可用于评委评测的问题")
        remaining_token_budget = token_budget
        
        # 执行评委评测
        judge_correct_count, judge_total, judge_results, input_tokens_judge, output_tokens_judge = batch_llm_mcq_judge(
            llm_client, llm_model, qa_list, max_workers=8,
            max_samples=max_samples, token_budget=token_budget,
            save_path=judge_save_path, batch_size=batch_size
        )
        
        # 更新token统计
        total_input_tokens += input_tokens_judge
        total_output_tokens += output_tokens_judge
        
        # 更新token预算
        if token_budget:
            remaining_token_budget = token_budget - (total_input_tokens + total_output_tokens)
            print(f"已使用 {total_input_tokens + total_output_tokens}/{token_budget} tokens, 剩余预算: {remaining_token_budget}")
        
        # 直接按科目和年级分别统计评委结果
        subject_judge_correct = collections.defaultdict(int)
        subject_judge_total = collections.defaultdict(int)
        grade_judge_correct = collections.defaultdict(int)
        grade_judge_total = collections.defaultdict(int)
        
        for idx, (correct, subject, grade) in enumerate(zip(judge_results, sample_subjects[:len(judge_results)], sample_grades[:len(judge_results)])):
            if subject != 'UNKNOWN':
                subject_judge_total[subject] += 1
                if correct:
                    subject_judge_correct[subject] += 1
            
            if grade != 'UNKNOWN':
                grade_judge_total[grade] += 1
                if correct:
                    grade_judge_correct[grade] += 1
        
        # 计算科目级别评委评测结果
        judge_by_subject = {}
        for subject in subject_judge_total:
            total = subject_judge_total[subject]
            correct = subject_judge_correct[subject]
            if total > 0:
                judge_by_subject[subject] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': correct / total
                }
        
        # 计算年级级别评委评测结果
        judge_by_grade = {}
        for grade in grade_judge_total:
            total = grade_judge_total[grade]
            correct = grade_judge_correct[grade]
            if total > 0:
                judge_by_grade[grade] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': correct / total
                }
    else:
        # 如果不运行评委评测，设置空结果
        judge_correct_count = 0
        judge_total = 0
        judge_results = []
        judge_by_subject = {}
        judge_by_grade = {}
    
    # 2. 多维度打分评测
    dimensions = ['plausibility', 'incorrectness', 'distinctiveness', 'diagnostic_value']
    llm_eval_avg = {dim: 0.0 for dim in dimensions}
    
    # 直接按科目和年级分别统计多维度评分
    subject_dim_sums = {dim: collections.defaultdict(float) for dim in dimensions}
    subject_dim_counts = collections.defaultdict(int)
    grade_dim_sums = {dim: collections.defaultdict(float) for dim in dimensions}
    grade_dim_counts = collections.defaultdict(int)
    
    if run_dim_score:
        print(f"\n=== 第二阶段: 多维度打分评测 ===")
        
        # 只有当有token预算时继续进行多维度评分
        remaining_token_budget = token_budget - (total_input_tokens + total_output_tokens) if token_budget else None
        if remaining_token_budget is None or remaining_token_budget > 0:
            # 准备多维度打分数据
            dim_qa_list = []
            dim_subjects = []
            dim_grades = []
            
            for test_item, gen_item in zip(test_data, generated_data):
                question = gen_item.get('question', '')
                correct_answer = gen_item.get('correct_answer', '')
                distractors = [gen_item.get(f'distractor{i}', '') for i in range(1, 4) if gen_item.get(f'distractor{i}', '')]
                
                if distractors and question and correct_answer:  # 确保有干扰项且问题和答案非空
                    subject_cat = map_subject(test_item.get('subject'))
                    grade_cat = map_grade(test_item.get('grade'))
                    
                    # 将所有干扰项作为一个整体添加
                    dim_qa_list.append((question, correct_answer, distractors))
                    dim_subjects.append(subject_cat)
                    dim_grades.append(grade_cat)
            
            print(f"共找到 {len(dim_qa_list)} 个问题用于多维度打分")
            
            # 执行多维度打分
            llm_batch_results, input_tokens_dim, output_tokens_dim = batch_llm_multi_dim_score(
                llm_client, llm_model, dim_qa_list, max_workers=8,
                max_samples=max_samples, token_budget=remaining_token_budget,
                log_file=token_log_file, save_path=dim_save_path, batch_size=batch_size
            )
            
            # 更新token统计
            total_input_tokens += input_tokens_dim
            total_output_tokens += output_tokens_dim
            
            # 全局维度统计
            dim_counts = {dim: 0 for dim in dimensions}
            dim_sums = {dim: 0.0 for dim in dimensions}
            
            # 分别统计科目和年级维度得分
            for idx, (result, subject, grade) in enumerate(zip(llm_batch_results, dim_subjects[:len(llm_batch_results)], dim_grades[:len(llm_batch_results)])):
                if not result:
                    continue
                    
                for dim in dimensions:
                    if dim in result and 'score' in result[dim]:
                        score = result[dim]['score']
                        
                        # 全局统计
                        dim_sums[dim] += score
                        dim_counts[dim] += 1
                        
                        # 科目统计 
                        if subject != 'UNKNOWN':
                            subject_dim_sums[dim][subject] += score
                            subject_dim_counts[subject] += 1
                        
                        # 年级统计
                        if grade != 'UNKNOWN':
                            grade_dim_sums[dim][grade] += score
                            grade_dim_counts[grade] += 1
            
            # 计算全局维度平均分
            for dim in dimensions:
                if dim_counts[dim] > 0:
                    llm_eval_avg[dim] = dim_sums[dim] / dim_counts[dim]
            
            # 使用calculate_category_scores计算科目和年级级别统计
            subject_metrics = {}
            for dim in dimensions:
                subject_metrics[dim] = subject_dim_sums[dim]
            
            grade_metrics = {}
            for dim in dimensions:
                grade_metrics[dim] = grade_dim_sums[dim]
            
            llm_eval_by_subject = calculate_category_scores(subject_metrics, subject_dim_counts)
            llm_eval_by_grade = calculate_category_scores(grade_metrics, grade_dim_counts)
        else:
            # 如果没有足够token预算，设为空结果
            llm_eval_by_subject = {}
            llm_eval_by_grade = {}
    else:
        # 如果不运行多维度打分，设置空结果
        llm_eval_by_subject = {}
        llm_eval_by_grade = {}
    
    # 计算token费用
    input_cost = total_input_tokens / 1000 * 0.004  # 每千token输入0.004元人民币
    output_cost = total_output_tokens / 1000 * 0.016  # 每千token输出0.016元人民币
    total_cost = input_cost + output_cost
    
    # 组合最终结果
    final_result = {
        'llm_judge': {
            'correct_count': judge_correct_count,
            'total': judge_total,
            'accuracy': judge_correct_count / judge_total if judge_total else 0
        },
        'llm_judge_by_subject': judge_by_subject,
        'llm_judge_by_grade': judge_by_grade,
        'llm_eval': llm_eval_avg,
        'llm_eval_by_subject': llm_eval_by_subject,
        'llm_eval_by_grade': llm_eval_by_grade,
        'stats': {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'cost_rmb': total_cost,
            'evaluation_time': time.time() - start_time
        }
    }
    
    return final_result


def main():
    parser = argparse.ArgumentParser(description="大模型干扰项评估工具")
    parser.add_argument('-d', '--dataset', required=True, help="数据集名称或路径")
    parser.add_argument('-m', '--model', required=True, help="模型名称")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], 
                       required=True, help="提示类型")
    parser.add_argument('--split', type=str, default='test', help="HuggingFace数据集分割")
    parser.add_argument('-v', '--multimodal', type=int, choices=[0, 1], default=1,
                       help="是否评估多模态模式生成的文件 (0: 否/text-only, 1: 是/multimodal)")
    parser.add_argument('-w', '--way', choices=['api', 'local'], 
                       default='api', help="运行方式: api 或 local")
    parser.add_argument('--llm_model_key', type=str, default='deepseek', help="用于LLM评估的模型键名 (参考 api.yaml)")
    parser.add_argument('--max_samples', type=int, default=None, help="最大评估样本数量，用于调试或限制成本")
    parser.add_argument('--token_budget', type=int, default=1000000, help="Token预算上限，超过则停止评估")
    parser.add_argument('--token_log_file', type=str, default=None, help="Token消耗日志文件路径")
    parser.add_argument('--output_file', type=str, default=None, help="结果输出文件路径")
    parser.add_argument('--batch_size', type=int, default=10, help="批次保存大小，每处理多少个样本保存一次")
    parser.add_argument('--run_judge', type=int, choices=[0, 1], default=1, 
                       help="是否运行评委评测 (0: 否, 1: 是)")
    parser.add_argument('--run_dim_score', type=int, choices=[0, 1], default=1,
                      help="是否运行多维度打分 (0: 否, 1: 是)")
    args = parser.parse_args()
    
    # 加载配置
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # 获取数据集配置
    file_config = config['files'].get(args.dataset, {})
    dataset_name = config['dataset_names'].get(args.dataset, args.dataset)
    use_multimodal_flag = bool(args.multimodal)
    run_judge_flag = bool(args.run_judge)
    run_dim_score_flag = bool(args.run_dim_score)
    
    print(f"\n=== 大模型干扰项评估 ===")
    print(f"数据集: {args.dataset}, 模型: {args.model}")
    print(f"评估模式: {'多模态' if use_multimodal_flag else '纯文本'}")
    print(f"功能设置: {'运行' if run_judge_flag else '不运行'}评委评测, {'运行' if run_dim_score_flag else '不运行'}多维度打分")
    
    if args.max_samples:
        print(f"⚠️ 样本数量限制: 最多评估 {args.max_samples} 个样本")
    if args.token_budget:
        print(f"⚠️ Token预算限制: {args.token_budget} tokens")
    
    # 确定文件路径
    model_mode_suffix = "-text" if not use_multimodal_flag else ""
    way_suffix = "-local" if args.way == 'local' else ""
    test_file = file_config.get('test_file', None)
    output_file = f"{file_config.get('output_file', f'./output/output_dg-{args.dataset}')}-{args.model}{model_mode_suffix}-{args.prompt}-{args.split}{way_suffix}.json"
    
    if args.output_file:
        result_file = args.output_file
    else:
        result_file = f"./evaluation/llm_eval-{args.dataset}-{args.model}{model_mode_suffix}-{args.prompt}-{args.split}{way_suffix}.json"
    
    # 创建断点保存目录
    checkpoint_dir = f"./evaluation/checkpoints/{args.dataset}_{args.model}_{args.prompt}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Token日志文件
    token_log_file = args.token_log_file
    if token_log_file is None:
        # 默认的token日志文件
        token_log_dir = "./evaluation/token_logs"
        os.makedirs(token_log_dir, exist_ok=True)
        token_log_file = f"{token_log_dir}/{args.dataset}_{args.model}_{args.prompt}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # 初始化LLM
    print("\n--- 初始化 LLM 评估客户端 ---")
    llm_config = load_llm_config()
    if not llm_config:
        print("\033[91m无法加载 LLM 配置，无法进行评估。\033[0m")
        return
    
    llm_client, llm_api_model = initialize_llm_client(llm_config, model_key=args.llm_model_key)
    if not llm_client:
        print("\033[91mLLM 客户端初始化失败，无法进行评估。\033[0m")
        return
    
    # 加载测试数据
    try:
        print("\n--- 加载数据 ---")
        if test_file:
            print(f"从本地文件加载测试数据: {test_file}")
            with open(test_file, 'r') as f:
                test_data = json.load(f)
        else:
            print(f"从HuggingFace加载测试数据: {dataset_name} (split={args.split})")
            from datasets import load_dataset
            test_data = load_dataset(dataset_name, split=args.split)
        
        print(f"从文件加载生成数据: {output_file}")
        with open(output_file, 'r') as f:
            generated_data = json.load(f)
        
        if any('output' in item for item in generated_data):
            print("检测到output格式的生成数据，进行预处理...")
            generated_data = preprocess_generation_data(generated_data)
            print(f"预处理完成，共处理 {len(generated_data)} 个样本")
        
        print(f"测试集: {len(test_data)} 样本, 生成集: {len(generated_data)} 样本")
    except FileNotFoundError:
        print(f"\n\033[91m错误: 找不到数据文件。\033[0m")
        print("请确保生成的输出文件存在。")
        return
    except json.JSONDecodeError:
        print(f"\n\033[91m错误: 文件格式不是有效的 JSON。\033[0m")
        return
    
    # 评估
    print("\n--- 开始大模型评估 ---")
    result = evaluate_with_llm(
        test_data, 
        generated_data, 
        llm_client, 
        llm_api_model,
        max_samples=args.max_samples,
        token_budget=args.token_budget,
        token_log_file=token_log_file,
        save_dir=checkpoint_dir,
        batch_size=args.batch_size,
        run_judge=run_judge_flag,
        run_dim_score=run_dim_score_flag
    )
    
    # 打印结果摘要
    print("\n=== 评估结果摘要 ===")
    
    if run_judge_flag and 'llm_judge' in result:
        judge_accuracy = result['llm_judge']['accuracy']
        print(f"评委评测正确率: {result['llm_judge']['correct_count']}/{result['llm_judge']['total']} = {judge_accuracy:.4f}")
        print("(正确率越高，说明大模型越容易区分正确答案和干扰项，干扰项质量越差)")
        
        # 按科目打印评委评测结果
        if result.get('llm_judge_by_subject'):
            print("\n--- 评委评测科目级别结果 ---")
            for subj, stats in sorted(result['llm_judge_by_subject'].items()):
                print(f"{subj}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
        
        # 打印年级级别结果
        if result.get('llm_judge_by_grade'):
            print("\n--- 评委评测年级级别结果 ---")
            for grade, stats in sorted(result['llm_judge_by_grade'].items()):
                print(f"{grade}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
    
    # 打印维度评分结果
    if run_dim_score_flag and result.get('llm_eval'):
        print("\n--- 维度评分结果 (1-5分) ---")
        for dim, score in sorted(result['llm_eval'].items()):
            print(f"{dim}: {score:.2f}")
    
        # 按科目打印维度评分
        if result.get('llm_eval_by_subject'):
            print("\n--- 维度评分科目级别结果 ---")
            for subj, dims in sorted(result['llm_eval_by_subject'].items()):
                print(f"{subj}:", end="")
                for dim, score in sorted(dims.items()):
                    if dim != 'count':
                        print(f" {dim}={score:.2f}", end="")
                print(f" (样本数: {dims.get('count', 0)})")
        
        # 打印年级级别维度评分（如果有）
        if result.get('llm_eval_by_grade'):
            print("\n--- 维度评分年级级别结果 ---")
            for grade, dims in sorted(result['llm_eval_by_grade'].items()):
                print(f"{grade}:", end="")
                for dim, score in sorted(dims.items()):
                    if dim != 'count':
                        print(f" {dim}={score:.2f}", end="")
                print(f" (样本数: {dims.get('count', 0)})")
    
    # 打印token统计
    print(f"\n--- 资源消耗 ---")
    input_tokens = result['stats'].get('input_tokens', 0)
    output_tokens = result['stats'].get('output_tokens', 0)
    total_tokens = result['stats'].get('total_tokens', 0)
    cost_rmb = result['stats'].get('cost_rmb', 0)
    print(f"总Token消耗: 输入{input_tokens}, 输出{output_tokens}, 总计{total_tokens}")
    print(f"费用: 约人民币{cost_rmb:.2f}元 (输入0.004元/千tokens, 输出0.016元/千tokens)")
    print(f"评估用时: {result['stats']['evaluation_time']:.2f}秒")
    
    # 保存结果
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n结果已保存至: {result_file}")
    if token_log_file:
        print(f"Token消耗日志已保存至: {token_log_file}")

if __name__ == "__main__":
    main() 