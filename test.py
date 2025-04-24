import os
from RetrieveFramework import RetrieverFramework
from utils.utils import format_rationale_output
from PromptFramwork import PromptFramework as pf    
from datasets import load_dataset

def read_test_data_iter(dataset, start_index=0,  split='test'):
    """逐条读取 Hugging Face 数据集数据
    Returns:
        tuple: (数据集长度, 数据迭代器)
    """
    hf_dataset = load_dataset(dataset, split=split)
    print(f"当前加载的数据集分割: {split}")
    print(f"数据集样本数量: {len(hf_dataset)}")

    def data_iterator():
        for index, sample in enumerate(hf_dataset): #  使用 enumerate 获取索引
            if index < start_index: #  跳过已处理的数据项
                 continue
            
            transformed_sample = {}
            
            # 添加原始索引
            transformed_sample['original_index'] = index
            # 保留基本信息
            transformed_sample['question'] = sample['question']
            # transformed_sample['image'] = sample['image']
            transformed_sample['subject'] = sample['subject']
            
            # 合并 lecture 和 solution 作为支持文本
            transformed_sample['support'] = f"{sample['lecture']}. {sample['solution']}. {sample['hint']}"
            
            # 处理正确答案和干扰项
            correct_answer_index = sample['answer']
            all_choices = sample['choices']
            transformed_sample['correct_answer'] = all_choices[correct_answer_index]

            # 提取所有干扰项
            distractors = []
            for index_c, choice in enumerate(all_choices): #  避免变量名冲突，修改为 index_c
                if index_c != correct_answer_index:
                    distractors.append(choice)
            
            # 添加所有干扰项
            for i, distractor in enumerate(distractors):
                transformed_sample[f'distractor{i+1}'] = distractor
            
            yield transformed_sample
    
    return len(hf_dataset), data_iterator()

if __name__ == "__main__":    
    query_idx = 0
    similar_examples = RetrieverFramework.get_similar_examples(query_idx=0, split="validation", k=3)  # 通过类调用
    
    for example in similar_examples:
        print()
        print(example)

    total_items, dataset_iter = read_test_data_iter("science_qa", start_index=0, split='validation')
    questionData = next(dataset_iter)  # 只获取第一条数据
    print(questionData)

    principles = ["Principle 1: ...", "Principle 2: ..."]  # 示例原则
    similar_examples = RetrieverFramework.get_similar_examples(query_idx=0, k=3)
    prompt = pf.rule_based_rg_prompt(questionData, principles, examples=similar_examples)
    print(prompt)
    print("\n\n")
    response = """Incorrect Inference 1: Principle1 - Confusing similar concepts: Sodium bromide is an elementary substance because it is used in swimming pools to kill bacteria.
Incorrect Inference 2: Principle2 - Answering irrelevant questions: Sodium bromide is a compound because it is a type of salt.
Incorrect Inference 3: Principle3 - Vague memories: Sodium bromide is a compound because it contains the word bromide in its name, which sounds like a compound.
Incorrect Inference 4: Principle4 - Concept substitution: Sodium bromide is a compound because it is made from sodium and bromine, which are both metals.
Incorrect Inference 5: Principle5 - Reversing the primary and secondary relationships: Sodium bromide is a compound because it can be found in nature as a single element.
Incorrect Inference 6: Principle6 - Over-detailing or generalization: Sodium bromide is a compound because it is composed of sodium ions and bromide ions, which are bonded in a 1:1 ratio, making it a complex mixture rather than a simple combination of elements."""
    inference = format_rationale_output(response)
    prompt_dg = pf.rule_based_dg_prompt(questionData, inference, examples=similar_examples)
    print(prompt_dg)