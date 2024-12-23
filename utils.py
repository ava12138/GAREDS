import re
from PromptFramwork import PromptFramework as pf

def format_question_output(response):
    question_match = re.search(r'Question:\s*(.*)', response)
    answer_match = re.search(r'Answer:\s*(.*)', response)
    question = question_match.group(1).strip() if question_match else ''
    answer = answer_match.group(1).strip() if answer_match else ''
    return {'question': question, 'correct_answer': answer}

def format_rationale_output(response):
    explanation_match = re.search(r'Explanation:\s*(.*?)\n', response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else ''
    
    incorrect_inferences = re.findall(r'Incorrect Inference \d+ \((.*?)\): (.*?)(?=\nIncorrect Inference \d+|$)', response, re.DOTALL)
    incorrect_inferences_dict = {f'incorrect_inference_{i+1}': {'principle': principle, 'inference': inference.strip()} for i, (principle, inference) in enumerate(incorrect_inferences)}
    
    return {'explanation': explanation, **incorrect_inferences_dict}
