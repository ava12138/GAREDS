import numpy as np
import torch
import json
import yaml
import argparse
from tqdm import tqdm
from utils.tokenizer import EnhancedTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

type_mapping = {
    'lan': 'Language Science',
    'nat': 'Natural Science',
    'soc': 'Social Science',
    'sciqa-text':'SciQa-Text',
    'sciq': 'SciQa'
}

def calculate_rouge_l(groundtruth, output, context):
    """改进的ROUGE-L计算，结合语义相似度和字面匹配
    Args:
        groundtruth: 真实干扰项
        output: 生成的干扰项
        context: 问题上下文（将被轻度使用）
    Returns:
        float: 综合评分（0-100）
    """
    # 使用增强的分词器
    tokenizer = EnhancedTokenizer()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True, tokenizer=tokenizer)
    
    # 1. 计算直接的ROUGE-L分数
    direct_score = scorer.score(output, groundtruth)['rougeL'].fmeasure
    
    # 2. 计算带轻量上下文的ROUGE-L分数
    context_score = scorer.score(
        f"{context} {output}",
        f"{context} {groundtruth}"
    )['rougeL'].fmeasure
    
    # 3. 综合计算最终分数
    # 直接匹配占70%权重，上下文匹配占30%权重
    final_score = context_score * 100
    
    return final_score

def calculate_context_bleu(groundtruth, output, context):
    """改进的BLEU计算，使用平滑函数和预处理"""
    # 使用增强的分词器预处理
    tokenizer = EnhancedTokenizer()
    
    # 预处理并分词
    groundtruth_tokens = tokenizer.tokenize(groundtruth)
    output_tokens = tokenizer.tokenize(output)
    
    # 使用平滑函数
    weights = (0.4, 0.3, 0.2, 0.1)  # 调整n-gram权重
    smoothing = SmoothingFunction().method1
    
    return sentence_bleu(
        [groundtruth_tokens], 
        output_tokens, 
        weights=weights,
        smoothing_function=smoothing
    ) * 100

def calculate_semantic_similarity(text1, text2, model):
    """计算两个文本的语义相似度"""
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()

# def calculate_novelty(distractor, training_distractors, model):
#     """计算干扰项的新颖性（与训练集的差异度）"""
#     similarities = []
#     for train_distractor in training_distractors:
#         sim = calculate_semantic_similarity(distractor, train_distractor, model)
#         similarities.append(sim)
#     return 1 - max(similarities) if similarities else 1

def calculate_diversity(distractors, model):
    """计算干扰项之间的多样性"""
    if len(distractors) < 2:
        return 0
    
    similarities = []
    for i in range(len(distractors)):
        for j in range(i + 1, len(distractors)):
            sim = calculate_semantic_similarity(distractors[i], distractors[j], model)
            similarities.append(sim)
    return 1 - np.mean(similarities)

def calculate_relevance(distractor, question, correct_answer, model):
    """计算干扰项与问题和答案的相关性"""
    question_sim = calculate_semantic_similarity(distractor, question, model)
    answer_sim = calculate_semantic_similarity(distractor, correct_answer, model)
    
    # 保持原有计算方式
    relevance = (question_sim * (1 - answer_sim)) ** 0.5
    
    # 处理复数结果
    if isinstance(relevance, complex):
        return abs(relevance)  # 返回复数的模
    return float(relevance)



def evaluate_distractors(test_file, output_file, training_file=None):
    # 加载预训练模型
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 加载数据
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    with open(output_file, 'r') as f:
        output_data = json.load(f)
    
    training_distractors = []
    if training_file:
        with open(training_file, 'r') as f:
            training_data = json.load(f)
            for item in training_data:
                training_distractors.extend([
                    item['distractor1'],
                    item['distractor2'],
                    item['distractor3']
                ])


    metrics = {
        'bleu_scores': [],
        'rouge_scores': [],
        # 'novelty_scores': [],
        'diversity_scores': [],
        'relevance_scores': []
    }
    
    for test_item in tqdm(test_data, desc="Evaluating"):
        for output_item in output_data:
            if output_item['question'] == test_item['question']:
                # 获取生成的干扰项
                generated_distractors = [
                    output_item['distractor1'],
                    output_item['distractor2'],
                    output_item['distractor3']
                ]

                # 1. BLEU得分
                bleu_score = np.mean([
                    calculate_context_bleu(d, gen_d, test_item['question'])
                    for d, gen_d in zip([test_item['distractor1'], test_item['distractor2'], test_item['distractor3']], 
                                      generated_distractors)
                ])
                metrics['bleu_scores'].append(bleu_score)
                
                # 2. ROUGE得分
                rouge_score = np.mean([
                    calculate_rouge_l(d, gen_d, test_item['question'])
                    for d, gen_d in zip([test_item['distractor1'], test_item['distractor2'], test_item['distractor3']], 
                                      generated_distractors)
                ])
                metrics['rouge_scores'].append(rouge_score)

                
                # # 3. 新颖性得分
                # novelty_score = np.mean([
                #     calculate_novelty(d, training_distractors, model)
                #     for d in generated_distractors
                # ])
                # metrics['novelty_scores'].append(novelty_score)
                
                # 4. 多样性得分
                diversity_score = calculate_diversity(generated_distractors, model)
                metrics['diversity_scores'].append(diversity_score)
                
                # 5. 相关性得分
                relevance_score = np.mean([
                    calculate_relevance(d, test_item['question'], test_item['correct_answer'], model)
                    for d in generated_distractors
                ])
                metrics['relevance_scores'].append(float(relevance_score))
                
                break
    
    # 计算平均分数
    final_scores = {
        metric: np.mean(scores) for metric, scores in metrics.items()
    }


    return final_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate distractors")
    parser.add_argument('-d', '--dataset', choices=['lan', 'nat', 'soc','sciqa-text','sciq'], required=True, help="Type of test file to evaluate")
    parser.add_argument('-m', '--model', required=True, help="Model name used for generation")
    parser.add_argument('-p', '--prompt', choices=['rule', 'cot', 'non'], required=True, help="Prompt type")
    args = parser.parse_args()
    type_description = type_mapping.get(args.dataset, 'Unknown Type')  # 如果未找到类型，默认显示 'Unknown Type'
    
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    file_config = config['files'][args.dataset]
    test_file = file_config['test_file']
    # 添加模型名称到输出文件路径
    output_file = f"{file_config['output_file']}-{args.model}-{args.prompt}.json"
    # 添加模型名称到结果文件路径
    results_file = f"{file_config['results_file']}-{args.model}-{args.prompt}.json"
    
    results = evaluate_distractors(test_file, output_file)
    
    print(f"\n=== Evaluation Results Of {type_description}===")
    print(f"BLEU Score: {results['bleu_scores']:.4f}")
    print(f"ROUGE-L Score: {results['rouge_scores']:.4f}")
    # print(f"Novelty Score: {results['novelty_scores']:.4f}")
    print(f"Diversity Score: {results['diversity_scores']:.4f}")
    print(f"Relevance Score: {results['relevance_scores']:.4f}")

    # 将结果保存为 JSON 文件
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()