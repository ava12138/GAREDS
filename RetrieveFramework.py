import json
from socket import PF_CAN
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
from PromptFramwork import PromptFramework as pf

class RetrieverFramework:
    cache_dir = "./vector_cache"
    train_vector_cache = os.path.join(cache_dir, 'train_vectors.pt')
    index_cache = os.path.join(cache_dir, 'similar_indices.json')
    dataset_cache = os.path.join(cache_dir, 'dataset.json')

    _similarity_indices = None  # 相似度索引缓存
    _dataset = None  # 数据集缓存
    
    def __init__(self, cache_dir="./vector_cache"):
        self.device = "cpu"
        self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        # 更新类属性
        RetrieverFramework.cache_dir = cache_dir
        RetrieverFramework.train_vector_cache = os.path.join(cache_dir, 'train_vectors.pt')
        RetrieverFramework.index_cache = os.path.join(cache_dir, 'similar_indices.json')
        RetrieverFramework.dataset_cache = os.path.join(cache_dir, 'dataset.json')
        os.makedirs(cache_dir, exist_ok=True)

    def build_vector_cache(self, dataset_path, split="test", encoding_pattern="q+c", batch_size=32):
        """第一阶段:构建向量缓存"""
        print(f"Building vector cache for {split} set...")
        
        # 1. 加载训练集向量
        print("Loading training vectors...")
        train_vectors = torch.load(self.train_vector_cache)
        print(f"Training vectors loaded, shape: {train_vectors.shape}")
        
        # 2. 加载测试集并生成向量
        print(f"Loading {split} dataset...")
        test_dataset = load_dataset(dataset_path, split=split)
        print(f"Test dataset loaded, total {len(test_dataset)} examples")
        
        # 生成测试集文本表示
        print("Generating text representations...")
        texts = []
        for item in tqdm(test_dataset, desc="Processing texts"):
            if encoding_pattern == "q":
                text = f"question: {item['question']}"
            elif encoding_pattern == "q+c":
                choices_text = " | ".join(item['choices'])
                text = f"question: {item['question']} choices: {choices_text}"
            elif encoding_pattern == "q+a":
                correct_answer = item['choices'][item['answer']]
                text = f"question: {item['question']} correct answer: {correct_answer}"
            elif encoding_pattern == "q+c+a":
                choices_text = " | ".join(item['choices'])
                correct_answer = item['choices'][item['answer']]
                text = f"question: {item['question']} choices: {choices_text} correct answer: {correct_answer}"
            texts.append(text)
        
        # 3. 分批生成测试集向量表示
        print("Generating test vectors...")
        test_vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating vectors"):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                test_vectors.append(batch_vectors)
            except RuntimeError as e:
                print(f"\nError processing batch {i}-{i+batch_size}: {e}")
                for text in batch_texts:
                    vector = self.model.encode(text, convert_to_tensor=True)
                    test_vectors.append(vector)
        
        test_vectors = torch.cat(test_vectors, dim=0)
        torch.save(test_vectors, os.path.join(self.cache_dir, f"{split}_vectors.pt"))
        # 4. 为每个测试集问题找到训练集中最相似的样例
        print("Computing similarity indices with training set...")
        similarity_indices = {}
        batch_size = min(batch_size, 100)
        
        for i in tqdm(range(0, len(test_vectors), batch_size), desc="Computing similarities"):
            batch_vectors = test_vectors[i:i+batch_size]
            
            for j in range(len(batch_vectors)):
                query_vector = batch_vectors[j].unsqueeze(0)
                # 计算与训练集所有向量的相似度
                similarities = util.cos_sim(query_vector, train_vectors)[0]
                # 获取最相似的k个训练集索引
                top_k = torch.topk(similarities, k=3).indices.tolist()
                test_idx = i + j
                similarity_indices[str(test_idx)] = top_k
        
        # 5. 保存相似度索引
        with open(self.index_cache, 'w') as f:
            json.dump(similarity_indices, f)
        print(f"Similarity indices saved to {self.index_cache}")

    @classmethod
    def load_caches(cls):
        """加载缓存文件到内存"""
        if cls._similarity_indices is None:
            with open(cls.index_cache, 'r') as f:
                cls._similarity_indices = json.load(f)
        
        if cls._dataset is None:
            with open(cls.dataset_cache, 'r') as f:
                cls._dataset = json.load(f)

    @classmethod
    def get_similar_examples(cls, query_idx, k=3):
        """
        获取k个相似示例并格式化输出
        Args:
            query_idx: 查询样例的索引
            k: 需要返回的示例数量，默认为3
        Returns:
            str: 格式化后的示例字符串
        """
        # 加载缓存的相似度索引
        cls.load_caches()
            
        # 使用内存中的缓存
        similar_indices = cls._similarity_indices[str(query_idx)][:k]
    
        # 格式化相似样例
        formatted_examples = []
        for i, idx in enumerate(similar_indices):
            example = cls._dataset[idx]
            # 获取正确答案
            correct_answer = example['choices'][example['answer']]
            # 获取干扰项（除正确答案外的其他选项）
            distractors = [choice for i, choice in enumerate(example['choices']) 
                        if i != example['answer']]
            
            # 直接以文本形式格式化单个示例
            example_str = (f"Example {i+1}:\n"
                        f"Question: {example['question']}\n"
                        f"Answer: {correct_answer}\n"
                        f"Distractors: {', '.join(distractors)}")
            formatted_examples.append(example_str)
        
        return formatted_examples
    
    
if __name__ == "__main__":
    # retriever = RetrieverFramework()

    # # 为测试集构建向量缓存和相似度索引
    # retriever.build_vector_cache(
    #     dataset_path="derek-thomas/ScienceQA",
    #     split="test",
    #     encoding_pattern="q+c",
    #     batch_size=32
    # )

    # 获取相似样例
    query_idx = 0
    similar_examples = RetrieverFramework.get_similar_examples(query_idx=0, k=3)  # 通过类调用
    
    for example in similar_examples:
        print()
        print(example)

    # questionData = {
    #     "question": "What is the capital of France?",
    #     "correct_answer": "Paris",
    #     "support": "Paris is the capital of France.",
    # }
    # principles = ["Principle 1: ...", "Principle 2: ..."]  # 示例原则
    # similar_examples = RetrieverFramework.get_similar_examples(query_idx=0, k=3)
    # prompt = pf.rule_based_rg_prompt(questionData, principles, examples=similar_examples)
    # print(prompt)
    # 初始化检索器
