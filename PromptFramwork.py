class PromptFramework():
    STOP_TOKEN = "[stop]"

    @classmethod
    def producePrompt(cls, promptType, questionData=None, examples=None):
        # Add to this ladder and create an internal method
        if promptType == "qg":
            return cls.qg_prompt(examples)
        elif promptType == "rg":
            return cls.rule_based_rg_prompt(questionData, examples)
        elif promptType == "dg":
            return cls.rule_based_dg_prompt(questionData, examples)
        else:
            raise ValueError(promptType + " is not an available prompt type")
    
    @classmethod
    def qg_prompt(cls, examples):
        """
        === EXAMPLE ===\n
        <Instructions>\n
        === PROMPT ===\n
        Example1: XXX\n
        Example2: XXX\n
        """
        instructions="Refer to the following example, generate a question and its correct answer.\n \
        [Template]\n \
        Quesiton: XXX\n \
        Answer: XXX\n "
        examples_text = ""
        for idx, example in enumerate(examples):
            examples_text += f"Example{idx+1}:\nQuestion: {example['question']}\nAnswer: {example['correct_answer']}\n"
        prompt = f"{instructions}\n{examples_text}"
        prompt = prompt[:-1]
        return prompt


    @classmethod
    def rule_based_rg_prompt(cls, questionData, examples):
            """
            === EXAMPLE ===
            <Instructions>
            === PROMPT ===
            Question: XXX\n
            Answer: XXX\n
            Guideline1: XXX\n
            Guideline2: XXX\n
            Guideline3: XXX\n
            """
            instructions="You are given the following question along with the correct answer, and three guidelines for Faulty Reasoning. Please use the following template to give one correct explanation and three incorrect inferences based on the given three faulty guidelines. These three faulty inferences are used to help generate distractors for multiple-choice questions. \n\
            [Template]\n \
            Explanation: XXX\n \
            Incorrect Infernece1 \n\
            Incorrect Infernece2 \n\
            Incorrect Infernece3 \n"
            examples_text = ""
            for idx, example in enumerate(examples):
                examples_text += f"Guideline{idx+1}: {example}\n"
            prompt = f"{instructions}\nQuestion: {questionData['question'].strip()}\nAnswer: {questionData['correct_answer'].strip()}\n{examples_text}"
            prompt = prompt[:-1]
            return prompt
    
    @classmethod
    def rule_based_dg_prompt(cls, questionData, examples):
            """
            === EXAMPLE ===
            <Instructions>
            === PROMPT ===
            Question: XXX\n
            Answer: XXX\n
            Explanation: XXX\n
            Incorrect Infernece1 \n
            Incorrect Infernece2 \n
            Incorrect Infernece3 \n
            """
            instructions="You are given the following question along with the correct answer, explanation, and three faulty inferences. Please use the following template to give three alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam based on the given three faulty inferences. \n Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n\
            [Template]\n \
            Distractor1: \
            Distractor2: \
            Distractor3:"
            examples_text = f"Explanation: {examples['explanation'].strip()}\n"
            for idx in range(1, 4):
                if f'incorrect_inference_{idx}' in examples:
                     examples_text += f"Incorrect Inference{idx} ({examples[f'incorrect_inference_{idx}']['principle']}): {examples[f'incorrect_inference_{idx}']['inference']}\n"
            prompt = f"{instructions}\nQuestion: {questionData['question'].strip()}\nAnswer: {questionData['correct_answer'].strip()}\n{examples_text}"
            prompt = prompt[:-1]
            return prompt
    
# Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n