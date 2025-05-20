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
        # 修改设备指定逻辑，当设置CUDA_VISIBLE_DEVICES后，应该使用相对索引0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        self.engine = None
        self.model_name = model_name
        self._initialize_engine()

    def _initialize_engine(self):
        
        if self.inference_type == "pt":
            # 检查是否是LoRA微调后的模型
            if "checkpoint-" in self.model_path:
                print(f"检测到LoRA微调模型: {self.model_path}")
                # 首先确定基础模型路径
                if "qwen" in self.model_path.lower() and "vl" in self.model_path.lower():
                    base_model = "/data/lzx/model/qwen2.5-vl-7b-instruct"  # 基础模型路径
                    model_type = "qwen2.5-vl-7b-instruct"
                else:
                    # 读取args.json确定基础模型
                    import json
                    import os
                    args_path = os.path.join(self.model_path, "args.json")
                    if os.path.exists(args_path):
                        with open(args_path, "r") as f:
                            args = json.load(f)
                        base_model = args.get("model", "")
                        print(f"从args.json获取基础模型: {base_model}")
                    else:
                        raise ValueError(f"无法确定基础模型，请检查{self.model_path}是否包含args.json")
                
                # 使用Swift加载基础模型和LoRA权重
                from swift.llm import get_model_tokenizer, get_template
                from swift.tuners import Swift
                
                print(f"加载基础模型: {base_model}")
                # 在环境变量CUDA_VISIBLE_DEVICES已设置的情况下，只需指定"cuda"或"cuda:0"
                model, tokenizer = get_model_tokenizer(base_model)
                # 在加载LoRA权重时不指定device_map，让Swift自动处理
                model = Swift.from_pretrained(model, self.model_path)
                template_type = model.model_meta.template if hasattr(model, 'model_meta') and hasattr(model.model_meta, 'template') else None
                template = get_template(template_type, tokenizer)
                
                # 使用from_model_template方法创建引擎
                self.engine = PtEngine.from_model_template(
                    model, 
                    template, 
                    max_batch_size=2
                )
            else:
                # 普通模型直接使用PtEngine
                # 在环境变量CUDA_VISIBLE_DEVICES已设置的情况下，让PyTorch自动分配设备
                self.engine = PtEngine(
                    self.model_path,
                    max_batch_size=2
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
                    # VLLM会自动使用CUDA_VISIBLE_DEVICES环境变量指定的GPU
                    limit_mm_per_prompt={'image': 5, 'video': 2}
                )
            else:
                # 非多模态模型配置
                self.engine = VllmEngine(
                    self.model_path,
                    max_model_len=32768,
                    # 根据环境变量自动选择GPU
                    tensor_parallel_size=1  # 可以根据需要调整并行大小
                )
        else:
            raise ValueError(f"不支持的推理类型: {self.inference_type}")
               

    def generate_response(self, prompt, image=None, temperature=0.7, presence_penalty=0.0, max_tokens=2048):
        try:
            # 构建请求
            messages = [{'role': 'user', 'content': prompt}]
            request = InferRequest(messages=messages)
            
            # 处理图像 - 根据模型类型选择不同的处理方式
            if image is not None:
                print(f"处理图像输入...")
                if "checkpoint-" in self.model_path:
                    # 对于LoRA微调模型：直接设置images字段
                    print(f"使用直接传递图像方式 (LoRA模型)")
                    request.images = [image]
                else:
                    # 对于普通模型：转换为base64嵌入到内容中
                    print(f"使用base64编码图像方式 (非LoRA模型)")
                    image_base64 = convert_image_to_base64(image)
                    if image_base64:
                        request.messages[0]['content'] = f"<img>{image_base64}</img>\n{prompt}"
                    else:
                        print(f"警告: 图像转换为base64失败")

            request_config = RequestConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=presence_penalty
            )

            metric = InferStats()
            
            # 调用推理引擎
            print(f"调用模型进行推理...")
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
