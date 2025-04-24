
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
            return cls.rule_based_rg_prompt(questionData, principles, examples)
        elif promptType == "rule_dg":
            return cls.rule_based_dg_prompt(questionData, principles, examples)
        elif promptType == "cot_rg":
            return cls.cot_rg_prompt(questionData, examples=examples)
        elif promptType == "cot_dg":
            return cls.cot_dg_prompt(questionData, principles, examples=examples)
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
    def rule_based_rg_prompt(cls, questionData, principles, examples=None):
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
        instructions = ("You are given the following question along with the correct answer, and six principles for Faulty Reasoning as context for helping you generate reasonings."
                   "The question may include an image. The image contains information that helps you understand the question and can help you generate subsequent reasonings."
                   "Please refer to the examples of question and distractors below and use the template to give one correct explanation and "
                   "six incorrect inferences based on the given six principles. These six faulty inferences are used to help "
                   "generate distractors for multiple-choice questions.\n")
        
        # 添加示例部分
        examples_text = ""
        if examples:
            examples_text = "=== Examples ===\n"
            examples_text += "\n\n".join(examples) + "\n\n"
            examples_text += "=== Template ===\n"
        
        template = ("Explanation: XXX\n"
                "Incorrect Inference1: XXX\n"
                "Incorrect Inference2: XXX\n"
                "Incorrect Inference3: XXX\n"
                "Incorrect Inference4: XXX\n"
                "Incorrect Inference5: XXX\n"
                "Incorrect Inference6: XXX\n")
        
        principles_text = ""
        for idx, principle in enumerate(principles):
            principles_text += f"Principle{idx+1}: {principle}\n"
            
        
        prompt = (
            f"{instructions}\n"
            f"=== Context ===\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"=== Principles ===\n"
            f"{principles_text}\n"
            f"{examples_text}"
            f"{template}"
        )
        return prompt.strip()
    
    @classmethod
    def rule_based_dg_prompt(cls, questionData, principles, examples=None):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Answer: XXX\n
        Explanation: XXX\n
        Incorrect Inference1 \n\
        Incorrect Inference2 \n\
        Incorrect Inference3 \n\
        Incorrect Inference4 \n\
        Incorrect Inference5 \n\
        Incorrect Inference6 \n
        """
            # 计算需要生成的干扰项数量
        distractor_count = cls.count_distractors(questionData)
        
        instructions = (
            "You are given the following question along with the correct answer, explanation, and six faulty inferences as context for helping you generate distractors. "
            "The question may include an image. The image contains information that helps you understand the question and can help you generate subsequent distractors."
            "Please refer to the examples below and use the template to generate "
            f"**{distractor_count}** alternative incorrect but plausible answers to be used as multiple-choice options in a multiple-choice exam.\n "
            "You don't need to provide the explanation of distractors, just provide the distractors.\n"
        )
        # 添加示例部分
        examples_text = ""
        if examples:
            examples_text = "=== Examples ===\n"
            examples_text += "\n\n".join(examples) + "\n\n"
            examples_text += "=== Template ===\n"

        # 动态生成模板字符串
        template_parts = []
        for i in range(1, distractor_count + 1):
            template_parts.extend([
                f"Distractor{i}: **XXX**"
            ])
        template = "\n".join(template_parts)

        principles_text = f"Explanation: {principles['explanation'].strip()}\n"
        incorrect_inferences = principles['incorrect_inferences'].split('\n\n')
        for idx, inference in enumerate(incorrect_inferences, 1):
            principles_text += f"{inference.strip()}\n"

        prompt = f"{instructions}\n"
        
        prompt = (
            f"{instructions}\n"
            f"=== Context ===\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"=== Faulty Inferences ===\n"
            f"{principles_text}\n"
            f"{examples_text}"
            f"{template}\n"
        )
        return prompt.strip(), distractor_count
      
    @classmethod
    def cot_rg_prompt(cls, questionData, examples=None):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX
        Answer: XXX
        Support: XXX
        """
        instructions = (
            "You are given the following question along with the correct answer and supporting information as context for helping you generate reasonings. "
            "The question may include an image. The image contains information that helps you understand the question "
            "and can help you generate subsequent reasonings. "
            "Please refer to the examples of question and distractors below and use the template to generate reasonings. "
            "Please break down the reasoning process into logical steps that connect the supporting information "
            "to the correct answer. Ensure each step is clear and directly related to the question.\n"
        )
        
        # 添加示例部分
        examples_text = ""
        if examples:
            examples_text = "=== Examples ===\n"
            examples_text += "\n\n".join(examples) + "\n\n"
            examples_text += "=== Template ===\n"
        
        template = (
            "Step 1: XXX\n"
            "Step 2: XXX\n"
            "Step 3: XXX\n"
            "Final Explanation: XXX\n"
        )
        
        prompt = (
            f"{instructions}\n"
            f"=== Context ===\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"Support: {questionData['support'].strip()}\n"
            f"{examples_text}"
            f"{template}"
        )
        return prompt.strip()
    
    @classmethod
    def cot_dg_prompt(cls, questionData, principles, examples=None):
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
            "You are given the following question along with the correct answer, a step-by-step reasoning as context for helping you generate distractors. "
            "The question may include an image. The image contains information that helps you understand the question and can help you generate subsequent distractors."
            "Please refer to the examples below and use the template to generate "
            f"**{distractor_count}** alternative incorrect but plausible answers to be used as multiple-choice options in a multiple-choice exam.\n "
            "You don't need to provide the explanation of distractors, just provide the distractors.\n"
        )
        examples_text = ""
        if examples:
            examples_text = "=== Examples ===\n"
            examples_text += "\n\n".join(examples) + "\n\n"
            examples_text += "=== Template ===\n"

        reasoning_text = f"Inference: {principles.strip()}\n"
        prompt = (
            f"{instructions}\n"
            f"=== Context ===\n"
            f"Question: {questionData['question'].strip()}\n"
            f"Answer: {questionData['correct_answer'].strip()}\n"
            f"Support: {questionData['support'].strip()}\n"
            f"=== Reasoning ===\n"
            f"{reasoning_text}\n"
            f"{examples_text}"
            f"{template}"
        )
        return prompt.strip(), distractor_count
    
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