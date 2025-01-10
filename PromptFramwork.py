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
        instructions="Refer to the following example, generate a question  and its correct answer.\n \
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
            Support: XXX\n
            Principle1: XXX\n
            Principle2: XXX\n
            Principle3: XXX\n
            Principle4: XXX\n
            Principle5: XXX\n
            Principle6: XXX\n
            """
            instructions="You are given the following question along with the correct answer, and six principles for Faulty Reasoning. Please use the following template to give one correct explanation and six incorrect inferences based on the given six principles. These six faulty inferences are used to help generate distractors for multiple-choice questions. \n\
            [Template]\n \
            Explanation: XXX\n \
            Incorrect Infernece1: XXX \n\
            Incorrect Infernece2: XXX \n\
            Incorrect Infernece3: XXX \n\
            Incorrect Infernece4: XXX \n\
            Incorrect Infernece5: XXX \n\
            Incorrect Infernece6: XXX \n"
            examples_text = ""
            for idx, example in enumerate(examples):
                examples_text += f"Principle{idx+1}: {example}\n"
            prompt = (
                f"{instructions}\n"
                f"Question: {questionData['question'].strip()}\n"
                f"Answer: {questionData['correct_answer'].strip()}\n"
                f"Support: {questionData['support'].strip()}\n"
                f"{examples_text}"
            )
            return prompt.strip()
    
    @classmethod
    def rule_based_dg_prompt(cls, questionData, examples):
            """
            === EXAMPLE ===
            <Instructions>
            === PROMPT ===
            Question: XXX\n
            Answer: XXX\n
            Explanation: XXX\n
            Incorrect Infernece1 \n\
            Incorrect Infernece2 \n\
            Incorrect Infernece3 \n\
            Incorrect Infernece4 \n\
            Incorrect Infernece5 \n\
            Incorrect Infernece6 \n
            """
            instructions="You are given the following question along with the correct answer, explanation, and six faulty inferences. Please use the following template to give **Three** alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam based on the given faulty inferences. \n Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n\
            [Template]\n \
            Distractor1: **XXX**\n\
            Feedback1: XXX\n\
            Distractor2: **XXX**\n\
            Feedback2: XXX \n\
            Distractor3: **XXX**\n\
            Feedback3: XXX\n"
            examples_text = f"Explanation: {examples['explanation'].strip()}\n"
            incorrect_inferences = examples['incorrect_inferences'].split('\n\n')
            for idx, inference in enumerate(incorrect_inferences, 1):
                examples_text += f"{inference.strip()}\n"
            prompt = (
                f"{instructions}\n"
                f"Question: {questionData['question'].strip()}\n"
                f"Answer: {questionData['correct_answer'].strip()}\n"
                f"{examples_text}"
            )
            return prompt.strip()
    
# Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n