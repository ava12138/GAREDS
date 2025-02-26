from PIL import Image
import os
from openai import OpenAI, api_key
from utils.utils import convert_image_to_base64
from datasets import load_dataset
import io

def analyze_scienceqa_image():
    """
    分析ScienceQA数据集中的第一张图片
    """
    try:
        api_key = "sk-929f75f6b8af4c0db579490dc255aa85"
        # 初始化OpenAI客户端
        client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 加载数据集
        dataset = load_dataset("derek-thomas/ScienceQA")
        train_dataset = dataset['test']
        
        # 寻找第一个包含图片的样本
        for example in train_dataset:
            if 'image' in example and example['image'] is not None:
                # 获取图片数据
                image_data = example['image']
                              
                # 获取问题信息
                question = example['question']
                choices = example.get('choices', [])
                answer = example.get('answer', '')
                
                # 转换图片为base64
                base64_image = convert_image_to_base64(image_data)
                
                if base64_image is None:
                    print("图片转换失败")
                    return
                
                print("原始问题：", question)
                print("选项：", choices)
                print("正确答案：", answer)
                print("\n正在请求API分析图片...\n")
                
                # 发送API请求
                response = client.chat.completions.create(
                    model="qwen2.5-vl-72b-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "这张图片展示了什么？请用中文详细描述。"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": base64_image
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                # 打印结果
                print("API描述结果:")
                print(response.choices[0].message.content)
                break
                
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    analyze_scienceqa_image()