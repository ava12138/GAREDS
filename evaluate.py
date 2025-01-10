from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import re
import json
from utils import initialize_seeds, str_to_dict_eedi_df

def calculate_rouge_l(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, hypothesis)
    return score['rougeL'].fmeasure

def calculate_bleu(reference, hypothesis):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    score = sentence_bleu([reference_tokens], hypothesis_tokens)
    return score

def evaluate_distractors(test_file, output_file):
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    with open(output_file, 'r') as f:
        output_data = json.load(f)
    
    results = []
    
    for test_item in test_data:
        question = test_item['question']
        test_distractors = [test_item['distractor1'], test_item['distractor2'], test_item['distractor3']]
        
        for output_item in output_data:
            if output_item['question'] == question:
                output_distractors = [output_item['distractor1'], output_item['distractor2'], output_item['distractor3']]
                distractor_results = []
                
                for test_distractor, output_distractor in zip(test_distractors, output_distractors):
                    rouge_l_score = calculate_rouge_l(test_distractor, output_distractor)
                    bleu_score = calculate_bleu(test_distractor, output_distractor)
                    
                    distractor_results.append({
                        'distractor': output_distractor,
                        'rouge_l_score': rouge_l_score,
                        'bleu_score': bleu_score
                    })
                
                results.append({
                    'question': question,
                    'results': distractor_results
                })
    
    return results


# 示例用法
test_filename = "./evaluation/test.json"
output_filename = "./evaluation/output_dg.json"
results = evaluate_distractors(test_filename, output_filename)
for result in results:
    print(f"Question:\n {result['question']}:")
    for distractor_result in result['results']:
        print(f"  Distractor: \"{distractor_result['distractor']}\":")
        print(f"    ROUGE-L Score: {distractor_result['rouge_l_score']}")
        print(f"    BLEU Score: {distractor_result['bleu_score']}")