from datasets import load_dataset
import PIL.Image
import requests
from io import BytesIO
import base64
from io import BytesIO
from PIL import Image

def convert_image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    if not isinstance(image, Image.Image):
        return None
        
    try:
        buffered = BytesIO()
        # 保存为PNG格式，保持图像质量
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"图片转换错误: {str(e)}")
        return None
    
def test_image_processing():
    # 加载数据集
    dataset = load_dataset("derek-thomas/ScienceQA", split="test")
    
    # 获取第一个有图片的样本
    for sample in dataset:
        if 'image' in sample and sample['image'] is not None:
            # 检查图片格式
            print(f"原始图片数据类型: {type(sample['image'])}")
            
            if isinstance(sample['image'], str):
                # 如果是URL，下载图片
                try:
                    response = requests.get(sample['image'])
                    image = PIL.Image.open(BytesIO(response.content))
                except Exception as e:
                    print(f"下载/加载图片出错: {e}")
                    continue
            else:
                # 如果已经是PIL Image对象
                image = sample['image']
            
            print(f"图片类型: {type(image)}")
            if isinstance(image, PIL.Image.Image):
                print(f"图片大小: {image.size}")
                print(f"图片模式: {image.mode}")
            
            # 测试base64转换

            base64_str = convert_image_to_base64(image)
            print(f"Base64字符串长度: {len(base64_str) if base64_str else 0}")
            break

if __name__ == "__main__":
    test_image_processing()