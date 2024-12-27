from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import re
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

def evaluate_distractors(filename, num_distractors=3):
    initialize_seeds(40)
    gt_distractors = []
    generated_distractors = []
    distractor_nl_pattern = re.compile(r"(?i)(distractor ?(?:\d+):\**)\n")
    distractor_pattern = re.compile(r"(?i)\**distractor ?(?:\d+):\** (.+)")

    data = pd.read_csv(filename)
    data = str_to_dict_eedi_df(data)
    for idx, row in data.iterrows():
        distractors_per_question = []
        response = str(row['raw_response'])
        response = distractor_nl_pattern.sub(r"\g<1> ", response)
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if distractor_pattern.match(line):
                distractor = distractor_pattern.match(line).group(1)
                distractors_per_question.append(distractor)
        generated_distractors.append(distractors_per_question)
        gt_distractors.append(row['gt_distractors'].split(','))  # 假设真实干扰项在这一列

    results = []
    for i, (gt, generated) in enumerate(zip(gt_distractors, generated_distractors)):
        question_results = []
        for j, (gt_distractor, gen_distractor) in enumerate(zip(gt, generated)):
            rouge_l_score = calculate_rouge_l(gt_distractor, gen_distractor)
            bleu_score = calculate_bleu(gt_distractor, gen_distractor)
            question_results.append({
                'distractor': j+1,
                'rouge_l_score': rouge_l_score,
                'bleu_score': bleu_score
            })
        results.append({
            'question': i+1,
            'results': question_results
        })
    return results

# 示例用法
filename = "path/to/your/csvfile.csv"
results = evaluate_distractors(filename)
for result in results:
    print(f"Question {result['question']}:")
    for distractor_result in result['results']:
        print(f"  Distractor {distractor_result['distractor']}:")
        print(f"    ROUGE-L Score: {distractor_result['rouge_l_score']}")
        print(f"    BLEU Score: {distractor_result['bleu_score']}")