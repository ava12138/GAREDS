from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import json


def calculate_rouge_l(reference, hypothesis, context):
    """改进的ROUGE-L计算，考虑上下文并返回百分比分数"""
    context_reference = f"{context} {reference}"
    context_hypothesis = f"{context} {hypothesis}"
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(context_reference, context_hypothesis)
    
    # 转换为百分比形式
    return score['rougeL'].fmeasure * 100

def calculate_context_bleu(reference, hypothesis, context):
    """改进的BLEU计算，考虑上下文并返回百分比分数"""
    weights = (0.25, 0.25, 0.25, 0.25)
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    context_tokens = context.split()
    reference_with_context = context_tokens + reference_tokens
    hypothesis_with_context = context_tokens + hypothesis_tokens
    
    # 转换为百分比形式
    return sentence_bleu([reference_with_context], hypothesis_with_context, weights=weights) * 100

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
    # 我们希望与问题相关但与答案有所区别
    return (question_sim * (1 - answer_sim)) ** 0.5



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
        'novelty_scores': [],
        'diversity_scores': [],
        'relevance_scores': []
    }
    
    for test_item in test_data:
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
                metrics['relevance_scores'].append(relevance_score)
                
                break
    
    # 计算平均分数
    final_scores = {
        metric: np.mean(scores) for metric, scores in metrics.items()
    }


    return final_scores

def main():
    test_file = './evaluation/test.json'
    output_file = './evaluation/output_dg.json'
    training_file = './evaluation/train.json'
    
    results = evaluate_distractors(test_file, output_file)
    
    print("\n=== Evaluation Results ===")
    print(f"BLEU Score: {results['bleu_scores']:.4f}")
    print(f"ROUGE-L Score: {results['rouge_scores']:.4f}")
    # print(f"Novelty Score: {results['novelty_scores']:.4f}")
    print(f"Diversity Score: {results['diversity_scores']:.4f}")
    print(f"Relevance Score: {results['relevance_scores']:.4f}")
    print(f"Relaxed Score: {results['relaxed_score']:.4f}")
    print(f"Hard Score: {results['hard_score']:.4f}")
    print(f"Proportional Score: {results['proportional_score']:.4f}")

if __name__ == "__main__":
    main()