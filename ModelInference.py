from swift.llm import PtEngine, InferRequest, RequestConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from swift.plugin import InferStats
import torch
import os
from utils.utils import convert_image_to_base64

class ModelInference:
    def __init__(self, model_path, inference_type="pt", device_id=0, max_batch_size=1, model_name=None):
        self.model_path = model_path
        self.inference_type = inference_type
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.engine = None
        self.model_name = model_name
        self._initialize_engine()

    def _initialize_engine(self):
        
        if self.inference_type == "pt":
            self.engine = PtEngine(
                self.model_path,
                device_map={'': f'{self.device}'}
            )
        elif self.inference_type == "vllm":
            from swift.llm import VllmEngine
            # 检查模型是否支持多模态
            is_multimodal = any(name in self.model_path.lower() for name in ['vl', 'visual', 'qwen-vl'])
            
            if is_multimodal:
                # 多模态模型配置
                os.environ['MAX_PIXELS'] = '1003520'
                os.environ['VIDEO_MAX_PIXELS'] = '50176'
                os.environ['FPS_MAX_FRAMES'] = '12'
                self.engine = VllmEngine(
                    self.model_path, 
                    max_model_len=32768,            
                    limit_mm_per_prompt={'image': 5, 'video': 2}
                )
            else:
                # 非多模态模型配置
                self.engine = VllmEngine(
                    self.model_path,
                    max_model_len=32768,
                    tensor_parallel_size=1  # 可以根据需要调整并行大小
                )
        else:
            raise ValueError(f"不支持的推理类型: {self.inference_type}")
               

    def generate_response(self, prompt, image=None, temperature=0.7, presence_penalty=0.0, max_tokens=2048):
        try:
            # 处理图片
            if image is not None:
                image_base64 = convert_image_to_base64(image)
                if image_base64:
                    prompt = f"<img>{image_base64}</img>\n{prompt}"

            # 构建请求
            messages = [{'role': 'user', 'content': prompt}]
            request = InferRequest(messages=messages)
            request_config = RequestConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=presence_penalty
            )

            metric = InferStats()
            response = self.engine.infer(
                [request], 
                request_config, 
                metrics=[metric])[0]
            metrics = metric.compute()

            print(f"指标: {metrics}")

            return response.choices[0].message.content

        except Exception as e:
            print(f"生成响应时出错: {str(e)}")
            raise
