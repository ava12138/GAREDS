import json
import pandas as pd
import torch

class RetrieverFactory:
    def __init__(self, retrieverCfg, json_file_path):
        self.retrieverCfg = retrieverCfg
        # 加载 JSON 数据并转换为 DataFrame
        with open(json_file_path, 'r') as file:
            self.data = pd.DataFrame(json.load(file))

    def sum_embeddings(self, token_embeddings, input_mask_expanded):
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # q: question only 
    # q_a: question and answer
    # q_a_f: question, answer and feedback
    def fetch_examples(self, query):
        examples = self.data
        # Parse the questions into the requested example string
        if self.retrieverCfg.encodingPattern == "q":
            parsed_examples = examples["question"].tolist()
            # NOTE: I am not sure why i get into error of x is not a string. Bypassing by using str(x)
            parsed_examples = ["question: " + str(x) for x in parsed_examples]
            parsed_query = "question: " + query["question"]
                        
        elif self.retrieverCfg.encodingPattern == "q+a":
            ex_questions = examples["question"].tolist()
            ex_coption = examples["correct_option"].tolist()
            ex_questions = [f"question: {x}" for x in ex_questions]
            ex_correct = [f"correct answer: {x['option']}" for x in ex_coption]
            # concatenate the two lists
            parsed_examples = [q + "\n" + c + "\n" for q, c in zip(ex_questions, ex_correct)]
            # do the same for the query
            q_question = query["question"]
            q_coption = query["correct_option"]
            q_correct = f"correct answer: {q_coption['option']}"
            parsed_query = q_question + "\n" + q_correct

        return parsed_examples, parsed_query

# 示例用法
retrieverCfg = ...  # 您的配置对象 
json_file_path = '/path/to/your/json/file.json'
retriever = RetrieverFactory(retrieverCfg, json_file_path)
query = {"question": "What is the capital of France?", "correct_option": {"option": "Paris"}}
examples, parsed_query = retriever.fetch_examples(query)
print(examples)
print(parsed_query)