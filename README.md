# QG-DG Framework

一个专业的问题生成(Question Generation)和干扰项生成(Distractor Generation)框架，支持多模态输入和基于大型语言模型的推理。

## 功能特点

- 支持基于规则(Rule-based)和思维链(Chain-of-Thought)的推理生成
- 多模态支持，可处理文本和图像输入
- 提供完整的检索框架，实现相似问题检索和样例利用
- 兼容多种大语言模型，包括Qwen-7B和QwenVL等
- 支持API和本地推理两种部署方式
- 内置完善的评估系统

## 项目结构

- `PromptFramwork.py`: 提示工程框架，支持多种提示策略
- `RetrieveFramework.py`: 检索框架，用于找到相似问题
- `run-api.py`: API调用的主入口
- `run.py`: 本地推理的主入口
- `eval.py`/`eval-new.py`: 评估模块
- `dg_eval_llm.py`: 使用LLM进行评估的工具
- `config/`: 配置文件目录

## 运行指令

```bash
# API调用方式
PYTHONIOENCODING=utf8 && nohup python run-api.py -d scienceqa -m qwen7b -p rule 2>&1 | tee log/run-api.log &

# 本地VL模型运行方式
PYTHONIOENCODING=utf8 && nohup python run.py -d scienceqa -m qwenvl -p rule -i pt -g 7 > log/run-api-vl-local.log 2>&1 &

# VLLM加速方式运行
python run.py -d scienceqa -m qwen7b -i vllm -p rule -g 7 --split validation
```

## 评估方法

```bash
# LLM评估
python eval.py -d scienceqa -m qwenvl -p rule -v 1 -w api -c 0 
python eval.py -d scienceqa -m coe -p cot -v 0 -w api -c 0
# 干扰项LLM评估
python eval_llm.py -d scienceqa -m qwenvl -p rule -v 1 -w api --llm_model_key deepseek-v3 --output_file /home/lzx/lib/pro/evaluation/llm_eval.json --token_log_file /home/lzx/lib/pro/log/token.log
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- 支持CUDA的GPU环境(推荐)
- 用于多模态处理的相关依赖库

## 开源协议

请参阅项目中的LICENSE文件。