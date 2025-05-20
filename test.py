import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
from RetrieveFramework import RetrieverFramework
from utils.utils import format_rationale_output
from PromptFramwork import PromptFramework as pf    
from datasets import load_dataset

# 设置参数
def parse_args():
    parser = argparse.ArgumentParser(description='测试微调后的多模态模型性能')
    parser.add_argument('--model_path', type=str, default='output/qwen25-vl-7b-scienceqa/checkpoint-best', help='模型路径')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-VL-7B', help='基础模型路径')
    parser.add_argument('--split', type=str, default='validation', help='测试数据集分割，可选 validation 或 test')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--max_samples', type=int, default=100, help='最大测试样本数')
    return parser.parse_args()

# 加载模型
def load_model(args):
    print(f"正在加载模型: {args.model_path}")
    
    # 判断是否是LoRA模型
    is_lora = os.path.exists(os.path.join(args.model_path, 'adapter_config.json'))
    
    if is_lora:
        # 加载基础模型和LoRA权重
        from peft import PeftModel, PeftConfig
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=args.device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        # 加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=args.device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path if not is_lora else args.base_model,
        trust_remote_code=True
    )
    
    return model, tokenizer

# 评估模型
def evaluate_model(args, model, tokenizer):
    print(f"正在评估模型性能...")
    
    # 加载ScienceQA数据集
    dataset = load_dataset("derek-thomas/ScienceQA")
    test_data = dataset[args.split]
    
    if args.max_samples > 0:
        # 随机采样以加速评估
        indices = np.random.choice(len(test_data), min(args.max_samples, len(test_data)), replace=False)
        test_data = test_data.select(indices)
    
    correct = 0
    total = 0
    
    results = []
    
    for item in tqdm(test_data):
        # 构建问题
        question = item["question"]
        choices = item["choices"]
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        query = f"{question}\n\n{choices_str}"
        
        # 处理图像（如果有）
        image = None
        if item["image"] is not None:
            try:
                image_bytes = item["image"].get("bytes")
                if image_bytes:
                    import io
                    image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                print(f"无法加载图像: {e}")
        
        # 构建提示
        if image is not None:
            response, _ = model.chat(tokenizer, query=query, image=image, history=None)
        else:
            response, _ = model.chat(tokenizer, query=query, history=None)
        
        # 提取模型预测答案
        answer_idx = item["answer"]
        correct_answer = f"{chr(65+answer_idx)}"
        
        # 判断是否正确（简单的字符串匹配）
        predicted_answer = None
        for i in range(len(choices)):
            if f"{chr(65+i)}" in response[:50]:  # 只检查答案的前50个字符
                predicted_answer = f"{chr(65+i)}"
                break
        
        if predicted_answer == correct_answer:
            correct += 1
        
        total += 1
        
        # 记录结果
        results.append({
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "full_response": response,
            "is_correct": predicted_answer == correct_answer
        })
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    
    print(f"评估结果:")
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.4f} ({correct}/{total})")
    
    # 保存详细结果
    with open(f"evaluation_results_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2)
    
    return accuracy

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
    args = parse_args()
    model, tokenizer = load_model(args)
    accuracy = evaluate_model(args, model, tokenizer)

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