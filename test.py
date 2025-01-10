import re

def format_rationale_output(response):
    # 去除所有的 **
    cleaned_response = response.replace('**', '')
    
    # 匹配解释部分
    explanation_match = re.search(r'Explanation:\s*(.*?)\n', cleaned_response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else ''
    
    # 匹配所有不正确推断
    incorrect_inferences = re.findall(r'Incorrect Inference \d+ \((.*?)\):\s*(.*?)(?=\nIncorrect Inference \d+ \(|$|\n\n)', cleaned_response, re.DOTALL)
    incorrect_inferences_combined = ' '.join([f'Incorrect Inference {i+1} ({principle.strip()}): {inference.strip()}' for i, (principle, inference) in enumerate(incorrect_inferences)])
    
    return {'explanation': explanation, 'incorrect_inferences': incorrect_inferences_combined}

def format_distractor_output(text: str) -> dict:
    output = {}
    # 去除所有的 **
    cleaned_text = text.replace('**', '')
    
    # 匹配干扰项
    distractor_pattern = re.compile(r'Distractor\d:\s*(.*?)\s*(?=\n|$)')
    matches = distractor_pattern.findall(cleaned_text)
    
    for i, match in enumerate(matches, 1):
        output[f'distractor{i}'] = match.strip()
    
    return output

# 示例用法
if __name__ == "__main__":
    response = """
     **Template:**

Explanation: The term in biotechnology that means a genetically exact copy of an organism is a "clone." This is derived from the context that cloning involves creating a genetically identical organism by transferring the nucleus from a somatic cell into an enucleated egg cell, leading to an organism that is genetically identical to the donor.

**Incorrect Inference 1 (Confusing similar concepts):** 
The term in biotechnology that refers to a genetically exact copy of an organism is a "transgenic animal." This is incorrect because transgenic animals involve the introduction of a novel gene into the genome, whereas clones are exact genetic copies of an organism.

**Incorrect Inference 2 (Answering irrelevant questions):**
The term in biotechnology that indicates a genetically exact copy of an organism is "gene therapy." This is incorrect because gene therapy focuses on modifying an organism's existing genes rather than creating a genetically identical organism.

**Incorrect Inference 3 (Vague memories):**
The term in biotechnology that means a genetically exact copy of an organism is a "hybrid." This is incorrect because a hybrid is created by combining genetic material from two different species, rather than making an exact genetic copy of an individual organism.

**Incorrect Inference 4 (Concept substitution):**
The term in biotechnology that describes a genetically exact copy of an organism is "genetic modification." This is incorrect because genetic modification refers to altering the genetic makeup of an organism, not creating a clone with an exact copy of the donor's genome.

**Incorrect Inference 5 (Reversing the primary and secondary relationships):**
The term in biotechnology that means a genetically exact copy of an organism is an "embryo." This is incorrect because an embryo is an early stage of development of an organism, not a method or term for creating a genetically identical copy of an organism.

**Incorrect Inference 6 (Over-detailing or generalization):**
The term in biotechnology that means a genetically exact copy of an organism is the "process of transferring the nucleus of a somatic cell into an unfertilized egg cell whose nucleus has been removed or deactivated, followed by implantation into a surrogate mother." This is incorrect because it describes the cloning process in detail rather than the term for the resulting organism, which is "clone."
    """
    
    formatted_output = format_rationale_output(response)
    print(formatted_output)

    distractor_text = """
    **Distractor1**: **reductants**
    **Distractor2**: **respiration**
    **Distractor3**: **catalysts**
    """
    
    distractor_output = format_distractor_output(distractor_text)
    print(distractor_output)
# def rule_based_dg_prompt(cls, questionData, examples):
#     """
#     === EXAMPLE ===
#     <Instructions>
#     === PROMPT ===
#     Question: XXX\n
#     Answer: XXX\n
#     Explanation: XXX\n
#     Incorrect Inference 1 \n\
#     Incorrect Inference 2 \n\
#     Incorrect Inference 3 \n\
#     Incorrect Inference 4 \n\
#     Incorrect Inference 5 \n\
#     Incorrect Inference 6 \n
#     """
#     instructions = (
#         "You are given the following question along with the correct answer, explanation, and six faulty inferences. "
#         "Please use the following template to give **Three** alternative incorrect answers to be used as multiple-choice options "
#         "in a multiple-choice exam based on the given faulty inferences. \n"
#         "Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n"
#         "[Template]\n"
#         "Distractor1: **XXX**\n"
#         "Feedback1: XXX\n"
#         "Distractor2: **XXX**\n"
#         "Feedback2: XXX \n"
#         "Distractor3: **XXX**\n"
#         "Feedback3: XXX\n"
#     )
    
#     examples_text = f"Explanation: {examples['explanation'].strip()}\n"
#     incorrect_inferences = examples['incorrect_inferences'].split('\n\n')
#     for idx, inference in enumerate(incorrect_inferences, 1):
#         examples_text += f"{inference.strip()}\n"
    
#     prompt = (
#         f"{instructions}\n"
#         f"Question: {questionData['question'].strip()}\n"
#         f"Answer: {questionData['correct_answer'].strip()}\n"
#         f"{examples_text}"
#     )
    
#     return prompt.strip()

# # 示例用法
# questionData = {
#     "question": "What is the capital of France?",
#     "correct_answer": "Paris",
#     "support": "Paris is the capital and most populous city of France."
# }

# examples = {'explanation': ' Compounds that are capable of accepting electrons, such as O₂ or F₂, are called oxidants (or oxidizing agents) because they can oxidize other compounds. In the process of accepting electrons, an oxidant is reduced.', 'incorrect_inferences': 'Incorrect Inference 1:  Compounds that are capable of accepting electrons, such as O₂ or F₂, are called reductants because they can reduce other compounds.\n\n    Incorrect Inference 2 (Answering irrelevant questions): Compounds like O₂ or F₂ are important in respiration because they help break down glucose to release energy.\n\n    Incorrect Inference 3 (Vague memories): Compounds that are capable of accepting electrons, such as O₂ or F₂, are called catalysts because they speed up chemical reactions.\n\n    Incorrect Inference 4 (Concept substitution): Compounds that are capable of accepting electrons, such as O₂ or F₂, are called halogens, which are known for their tendency to bond with metals.\n\n    Incorrect Inference 5 (Reversing the primary and secondary relationships): Compounds that are capable of donating electrons, such as O₂ or F₂, are called oxidants because they can oxidize other compounds.\n\n    Incorrect Inference 6 (Over-detailing or generalization): Compounds that are capable of accepting electrons, such as O₂ or F₂, are called electron acceptors in redox reactions, which are characterized by their ability to undergo reduction by accepting electrons and donating oxygen atoms.'}

# prompt = rule_based_dg_prompt(None, questionData, examples)
# print(prompt)