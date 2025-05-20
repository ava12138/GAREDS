#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动评估指标工具模块 - 干扰项评估的自动计算指标
"""

import numpy as np
from typing import List, Dict, Any, Union
import re
import string
from collections import Counter
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# =====================
# 干扰项自动评估指标
# =====================

def normalize(text: str) -> str:
    """
    标准化文本:小写、去除标点符号、冠词和多余空格
    
    Args:
        text: 输入文本
        
    Returns:
        str: 标准化后的文本
    """
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

def calc_novelty(distractor: str, reference_data: List[Dict], threshold: float = 0.8, is_hf_dataset: bool = False) -> float:
    """
    计算干扰项的新颖性 - 与参考集(如训练集)的不同程度
    
    Args:
        distractor: 干扰项文本
        reference_data: 参考数据列表或HuggingFace数据集
        threshold: 判定为非新颖的相似度阈值
        is_hf_dataset: 是否为HuggingFace数据集格式
        
    Returns:
        float: 新颖性得分(0-1)，越高表示越新颖
    """
    # 标准化干扰项
    distractor_norm = normalize(distractor)
    distractor_tokens = set(distractor_norm.split())
    
    if not distractor_tokens:
        return 0.0
    
    # 与参考集每个干扰项计算相似度
    max_similarity = 0.0
    
    # 处理不同格式的参考数据
    if is_hf_dataset:
        # HuggingFace数据集格式
        for item in reference_data:
            # 从HuggingFace数据集提取干扰项
            if 'choices' in item and 'answer' in item:
                correct_idx = item['answer']
                choices = item['choices']
                ref_distractors = [normalize(choices[i]) for i in range(len(choices)) if i != correct_idx]
            else:
                # 尝试提取常规格式的干扰项
                ref_distractors = []
                for i in range(1, 4):
                    key = f'distractor{i}'
                    if key in item and item[key]:
                        ref_distractors.append(normalize(item[key]))
            
            # 计算与每个参考干扰项的相似度
            for ref in ref_distractors:
                ref_tokens = set(ref.split())
                if not ref_tokens:
                    continue
                
                # 使用Jaccard相似度
                intersection = len(distractor_tokens.intersection(ref_tokens))
                union = len(distractor_tokens.union(ref_tokens))
                
                similarity = intersection / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)
    else:
        # 普通JSON列表格式
        for item in reference_data:
            # 提取参考干扰项
            ref_distractors = []
            for i in range(1, 4):  # 假设每个问题最多3个干扰项
                key = f'distractor{i}'
                if key in item and item[key]:
                    ref_distractors.append(normalize(item[key]))
            
            # 计算与每个参考干扰项的相似度
            for ref in ref_distractors:
                ref_tokens = set(ref.split())
                if not ref_tokens:
                    continue
                
                # 使用Jaccard相似度
                intersection = len(distractor_tokens.intersection(ref_tokens))
                union = len(distractor_tokens.union(ref_tokens))
                
                similarity = intersection / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)
    
    # 新颖性得分 = 1 - 最大相似度
    novelty = 1.0 - max_similarity
    return novelty

def calc_diversity(distractors: List[str]) -> float:
    """
    计算一组干扰项之间的多样性
    
    Args:
        distractors: 干扰项列表
    
    Returns:
        float: 多样性得分(0-1)，越高表示越多样
    """
    if len(distractors) <= 1:
        return 0.0
    
    # 标准化干扰项
    normalized_distractors = [normalize(d) for d in distractors]
    tokenized_distractors = [set(d.split()) for d in normalized_distractors]
    
    # 计算两两之间的平均差异性
    total_diversity = 0.0
    pair_count = 0
    
    for i in range(len(tokenized_distractors)):
        for j in range(i+1, len(tokenized_distractors)):
            set1 = tokenized_distractors[i]
            set2 = tokenized_distractors[j]
            
            if not set1 or not set2:
                continue
            
            # 使用Jaccard距离
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            similarity = intersection / union if union > 0 else 0
            diversity = 1.0 - similarity
            
            total_diversity += diversity
            pair_count += 1
    
    # 平均多样性
    avg_diversity = total_diversity / pair_count if pair_count > 0 else 0.0
    return avg_diversity

def calc_bleu(hyp, refs):
    """
    计算BLEU分数（单句），可用于流畅性或与参考答案的相似度。
    """
    smoothie = SmoothingFunction().method4
    return sentence_bleu([r.split() for r in refs], hyp.split(), smoothing_function=smoothie)

def calc_ngram_diversity(distractors, n=2):
    """
    计算n-gram多样性（独特n-gram数量/总n-gram数量），衡量表达多样性。
    """
    all_ngrams = []
    for d in distractors:
        tokens = d.split()
        all_ngrams.extend(list(ngrams(tokens, n)))
    if not all_ngrams:
        return 0.0
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / (len(all_ngrams) + 1e-8)

def calc_lexical_complexity(text: str) -> float:
    """
    计算词汇复杂度 - 基于平均词长和词表大小
    
    Args:
        text: 输入文本
        
    Returns:
        float: 词汇复杂度得分(0-1)
    """
    # 标准化并分词
    normalized = normalize(text)
    tokens = normalized.split()
    
    if not tokens:
        return 0.0
    
    # 平均词长
    avg_word_length = sum(len(token) for token in tokens) / len(tokens)
    
    # 词表多样性 (不同词占总词数比例)
    vocab_diversity = len(set(tokens)) / len(tokens)
    
    # 综合得分
    score = (avg_word_length / 10 + vocab_diversity) / 2  # 假设平均词长最大约10
    return min(score, 1.0)  # 确保得分不超过1

def calc_plausibility(distractor: str, question: str, correct_answer: str) -> float:
    """
    计算干扰项的貌似合理性 - 基于与问题和正确答案的相关性
    
    Args:
        distractor: 干扰项文本
        question: 问题文本
        correct_answer: 正确答案文本
        
    Returns:
        float: 貌似合理性得分(0-1)
    """
    # 标准化文本
    distractor_norm = normalize(distractor)
    question_norm = normalize(question)
    answer_norm = normalize(correct_answer)
    
    # 分词
    distractor_tokens = set(distractor_norm.split())
    question_tokens = set(question_norm.split())
    answer_tokens = set(answer_norm.split())
    
    if not distractor_tokens:
        return 0.0
    
    # 与问题的相关性
    q_intersection = distractor_tokens.intersection(question_tokens)
    q_relation = len(q_intersection) / len(question_tokens) if question_tokens else 0
    
    # 与答案的差异性
    a_intersection = distractor_tokens.intersection(answer_tokens)
    a_difference = 1.0 - (len(a_intersection) / len(answer_tokens) if answer_tokens else 0)
    
    # 综合得分: 与问题相关但与答案不同
    score = (q_relation + a_difference) / 2
    return score 