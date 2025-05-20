#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计工具模块 - 用于干扰项评估的各种统计计算
"""

import collections
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import re

def map_subject(subject_str):
    """
    将原始 subject 字符串映射到分类标签
    """
    if not isinstance(subject_str, str):
        return 'UNKNOWN'
    subject_str = subject_str.lower().strip()
    if 'natural science' in subject_str:
        return 'NAT'
    elif 'social science' in subject_str:
        return 'SOC'
    elif 'language science' in subject_str:
        return 'LAN'
    else:
        return 'UNKNOWN'

def map_grade(grade_str):
    """
    将原始 grade 字符串 (如 'grade5') 映射到分类标签
    """
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

def calc_stats_by_category(sums: Dict[str, Dict[str, float]], counts: Dict[str, int], 
                          score_keys: List[str] = None, rouge_factor: float = 100.0) -> Dict[str, Dict[str, Any]]:
    """
    根据累积的求和和计数计算每个类别的统计分数
    
    Args:
        sums: 格式为{metric_name: {category: sum}}的累积求和字典
        counts: 格式为{category: count}的计数字典
        score_keys: 要计算的指标名称列表，默认为None表示计算所有指标
        rouge_factor: ROUGE指标的转换系数，默认为100.0转为百分比
    
    Returns:
        Dict: 按类别组织的统计结果 {category: {metric: score, count: n}}
    """
    # 确定需要计算的指标键
    all_metrics = set()
    for metric_dict in sums.values():
        all_metrics.update(metric_dict.keys())
    
    if score_keys is None:
        score_keys = list(sums.keys())
    
    # 计算每个类别的统计结果
    result = {}
    for category, count in counts.items():
        if count > 0:
            # 为每个类别创建包含计数的字典
            category_scores = {'count': count}
            
            # 计算每个指标的平均值
            for metric in score_keys:
                if metric in sums and category in sums[metric]:
                    # 对ROUGE特殊处理，应用转换系数
                    factor = rouge_factor if metric == 'rouge' else 1.0
                    category_scores[metric] = (sums[metric][category] / count) * factor
            
            # 将结果添加到输出字典
            result[category] = category_scores
    
    return result

def calculate_category_scores(scores_by_metric: Dict[str, Dict[str, float]], 
                             counts: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    """
    直接从单独的指标累积值计算每个类别的分数
    
    Args:
        scores_by_metric: 每个指标按类别累积的分数，格式为{metric: {category: sum}}
        counts: 每个类别的样本数，格式为{category: count}
    
    Returns:
        Dict: 按类别组织的统计结果 {category: {metric: score, count: n}}
    """
    result = {}
    
    # 获取所有指标名称
    metrics = list(scores_by_metric.keys())
    
    # 获取所有分类
    categories = set()
    for metric_dict in scores_by_metric.values():
        categories.update(metric_dict.keys())
    
    # 计算每个分类的平均分
    for category in categories:
        count = counts.get(category, 0)
        if count > 0:
            category_result = {'count': count}
            
            for metric in metrics:
                metric_scores = scores_by_metric.get(metric, {})
                if category in metric_scores:
                    # 对ROUGE特殊处理，转为百分比
                    value = metric_scores[category] / count
                    if metric == 'rouge':
                        value *= 100
                    category_result[metric] = value
            
            result[category] = category_result
    
    return result
