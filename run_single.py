import os
import json
import yaml
from openai import OpenAI
from PromptFramwork import PromptFramework as pf
from datasets import load_dataset
from utils.utils import (
    format_question_output, format_rationale_output, format_distractor_output
)

def load_config():
    """加载所需的配置文件"""
    with open('./config/api.yaml', 'r') as file:
        api_config = yaml.safe_load(file)
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    with open('./config/principle.json', 'r') as file:
        principles_config = json.load(file)
    return api_config, config, principles_config

def initialize_api_client(api_config):
    """初始化API客户端"""
    api_key = api_config['api_key']
    api_model = api_config['model']['dp-qwen7b']  # 直接使用dp-qwen7b模型
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    return client, api_model

def get_response(client, api_model, prompt, temperature=0.7, presence_penalty=0.0):
    """获取API响应并打印详细信息"""
    print("\n=== 发送的Prompt ===")
    print(prompt)
    
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        presence_penalty=presence_penalty,
    )
    
    print("\n=== API响应 ===")
    response_content = response.choices[0].message.content
    print(response_content)
    print(f"\n=== Token使用情况 ===")
    print(f"总Token数: {response.usage.total_tokens}")
    
    return response_content, response.usage.total_tokens

def process_single_question(dataset_name, split='test', index=0):
    """处理单个问题并显示详细过程"""
    # 加载配置
    api_config, config, principles_config = load_config()
    prompt_config = config['prompt_types']['rule']  # 使用COT提示
    distractor_principle = principles_config['distractor_principle']
    
    # 初始化客户端
    client, api_model = initialize_api_client(api_config)
    
    # 加载数据集
    print(f"\n=== 加载数据集 {dataset_name} ===")
    dataset = load_dataset(dataset_name, split=split)
    sample = dataset[index]
    
    # 转换数据格式
    print("\n=== 处理的问题数据 ===")
    correct_answer_index = sample['answer']
    all_choices = sample['choices']
    
    question_data = {
        'question': sample['question'],
        'subject': sample['subject'],
        'support': f"{sample['lecture']}. {sample['solution']}",
        'correct_answer': all_choices[correct_answer_index]
    }
    
    # 添加现有的干扰项
    distractors = []
    for idx, choice in enumerate(all_choices):
        if idx != correct_answer_index:
            distractors.append(choice)
            question_data[f'distractor{len(distractors)}'] = choice
    
    print("\n【原始问题数据】")
    print(f"问题: {question_data['question']}")
    print(f"学科: {question_data['subject']}")
    print(f"支持文本: {question_data['support']}")
    print(f"正确答案: {question_data['correct_answer']}")
    print("\n【现有干扰项】")
    for i, dist in enumerate(distractors, 1):
        print(f"干扰项{i}: {dist}")
    
    # 生成错误推理
    print("\n=== 生成错误推理 ===")
    rg_prompt = pf.producePrompt(prompt_config['rg'], question_data, distractor_principle)
    rationale_response, tokens_rg = get_response(client, api_model, rg_prompt)
    inference = format_rationale_output(rationale_response, prompt_config['format'])
    print("\n格式化后的推理:")
    print(json.dumps(inference, indent=2, ensure_ascii=False))
    
    # 生成干扰项
    print("\n=== 生成干扰项 ===")
    dg_prompt = pf.producePrompt(prompt_config['dg'], question_data, inference)
    distractor_response, tokens_dg = get_response(client, api_model, dg_prompt)
    
    # 统计需要的干扰项数量
    distractor_count = pf.count_distractors(question_data)
    extracted_distractors = format_distractor_output(distractor_response, distractor_count)
    
    print("\n=== 最终结果 ===")
    result = {
        "question": question_data['question'],
        "correct_answer": question_data['correct_answer']
    }
    for i in range(1, distractor_count + 1):
        result[f'distractor{i}'] = extracted_distractors.get(f'distractor{i}', '')
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n总Token消耗: {tokens_rg + tokens_dg}")

if __name__ == "__main__":
    # 使用ScienceQA数据集作为示例
    process_single_question("derek-thomas/ScienceQA", split='test', index=0)