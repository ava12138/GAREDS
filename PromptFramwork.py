from calendar import c
import dis


class PromptFramework():
    STOP_TOKEN = "[stop]"

    @staticmethod
    def count_distractors(question_data: dict) -> int:
        """
        统计问题数据中的干扰项数量
        Args:
            question_data: JSON格式的问题数据字典，包含 question, correct_answer 和 distractor1,2,3... 等字段
        Returns:
            int: 干扰项数量
        """
        # 方法 1: 直接计数具有 distractor 前缀的键
        return len([key for key in question_data.keys() if key.startswith('distractor')])

    @classmethod
    def producePrompt(cls, promptType, questionData=None, principles=None, examples=None):
        # Add to this ladder and create an internal method
        if promptType == "qg":
            return cls.qg_prompt(examples)
        elif promptType == "rule_rg":
            return cls.rule_based_rg_prompt(questionData, principles)
        elif promptType == "rule_dg":
            return cls.rule_based_dg_prompt(questionData, principles)
        elif promptType == "cot_rg":
            return cls.cot_rg_prompt(questionData)
        elif promptType == "cot_rg_shot":
            return cls.cot_rg_prompt_shot(questionData)
        elif promptType == "cot_dg":
            return cls.cot_dg_prompt(questionData, principles)
        elif promptType == "non_dg":
            return cls.dg_prompt(questionData)
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
            examples_text+= f"Example{idx+1}:\nQuestion: {example['question']}\nAnswer: {example['correct_answer']}\n"
        prompt = f"{instructions}\n{examples_text}"
        prompt = prompt[:-1]
        return prompt


    @classmethod
    def rule_based_rg_prompt(cls, questionData, principles):
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
        principles_text = ""
        for idx, principle in enumerate(principles):
            principles_text += f"Principle{idx+1}:s{principle}\n"
        prompt = (
            f"{instructions}\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"Support: {questionData['support'].strip()}\n"
            f"{principles_text}"
        )
        return prompt.strip()
    
    @classmethod
    def rule_based_dg_prompt(cls, questionData, principles):
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
            # 计算需要生成的干扰项数量
        distractor_count = cls.count_distractors(questionData)
        
        # 动态生成模板字符串
        template_parts = []
        for i in range(1, distractor_count + 1):
            template_parts.extend([
                f"Distractor{i}: **XXX**",
                f"Feedback{i}: XXX"
            ])
        template = "\n".join(template_parts)

        instructions=(
            "You are given the following question along with the correct answer, explanation, and six faulty inferences. "
            "Please use the following template to generate"
            f" **{distractor_count}** alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam based on the given faulty inferences. \n "
            "Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n"
            "[Template]\n"
            f"{template}\n"
        )
        
        principles_text = f"Explanation: {principles['explanation'].strip()}\n"
        incorrect_inferences = principles['incorrect_inferences'].split('\n\n')
        for idx, inference in enumerate(incorrect_inferences, 1):
            principles_text += f"{inference.strip()}\n"
        prompt = (
            f"{instructions}\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"{principles_text}"
        )
        return prompt.strip()
    
    @classmethod
    def cot_rg_prompt_shot(cls, questionData):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Answer: XXX\n
        Support: XXX\n
        """
        
        instructions="You are given the following example, which includes:\n- A question.\n- The correct answer to the question.\n- Supporting information that explains the answer.\n Carefully analyze the question and the supporting information.Break down the reasoning process into logical, step-by-step inferences that connect the supporting information to the correct answer.\n Ensure each step is clear, concise, and directly related to the question.\nThe intermediate reasoning should not include distractors, but should provide a strong foundation for generating distractors later.\
        [Template]\n \
        Inference: XXX\n"
        prompt = (
            f"{instructions}\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"Support: {questionData['support'].strip()}\n"
        )
        return prompt.strip()
    
    @classmethod
    def cot_rg_prompt(cls, questionData):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Answer: XXX\n
        """
        instructions="You are given the following question along with the correct answer. \n Carefully analyze the question and the answer.Break down the reasoning process into logical, step-by-step inferences that connect the supporting information to the correct answer.\n Ensure each step is clear, concise, and directly related to the question.\nThe intermediate reasoning should not include distractors, but should provide a strong foundation for generating distractors later.\
        [Template]\n \
        Inference: XXX\n"
        prompt = (
            f"{instructions}\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
        )
        return prompt.strip()
    
    @classmethod
    def cot_dg_prompt(cls, questionData, principles):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Answer: XXX\n
        Reasoning: XXX\n
        """
                # 计算需要生成的干扰项数量
        distractor_count = cls.count_distractors(questionData)
        
        # 动态生成模板字符串
        template_parts = []
        for i in range(1, distractor_count + 1):
            template_parts.extend([
                f"Distractor{i}: **XXX**",
                f"Feedback{i}: XXX"
            ])
        template = "\n".join(template_parts)

        instructions = (
            "You are provided with a question, the correct answer, and a step-by-step inference "
            "leading to that answer. Please use the following template to generate "
            f"**{distractor_count}** alternative incorrect answers to be used as multiple-choice "
            "options in a multiple-choice exam. \n"
            "[Template]\n"
            f"{template}\n"
        )
        reasoning_text = f"Inference: {principles.strip()}\n"
        prompt = (
            f"{instructions}\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"{reasoning_text}"
        )
        return prompt.strip()
    
    @classmethod
    def dg_prompt(cls, questionData):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Answer: XXX\n
        Support: XXX\n
        """
        distractor_count = cls.count_distractors(questionData)
        
        # 动态生成模板字符串
        template_parts = [f"Distractor{i}: **XXX**" for i in range(1, distractor_count + 1)]
        template = "\n".join(template_parts)
        
        instructions = (
            "You are provided with a question, the correct answer, and a support about the question. "
            f"Please use the following template to generate **{distractor_count}** alternative "
            "incorrect answers to be used as multiple-choice options in a multiple-choice exam.\n"
            "[Template]\n"
            f"{template}"
        )
        prompt = (
            f"{instructions}\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"Support: {questionData['support'].strip()}\n"
        )
        return prompt.strip()

# Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n