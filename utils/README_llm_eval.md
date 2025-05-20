# 干扰项评估工具使用指南

本文档介绍了干扰项评估工具的使用方法，包括自动评估和大模型评估功能。

## 1. 文件结构

项目包含三个核心评估文件：

1. `eval.py` - 自动评估脚本，使用BLEU/ROUGE和自动指标评估干扰项
2. `eval_llm.py` - 大模型评估脚本，使用LLM进行评委评测和多维度打分
3. `utils/llm_eval_utils.py` - 大模型评估工具函数库

## 2. 自动评估 (eval.py)

自动评估使用标准的NLP指标和自动计算的指标来评估干扰项质量：

- **BLEU/ROUGE**: 测量生成干扰项与参考干扰项的相似度
- **新颖性**: 测量干扰项与训练集的差异程度
- **多样性**: 测量同一问题不同干扰项之间的差异程度

### 使用方法

```bash
python eval.py -d scienceqa -m gpt-4 -p cot -v 1 -c 0 --train_set --max_samples 100
```

参数说明：
- `-d, --dataset`: 数据集名称
- `-m, --model`: 模型名称
- `-p, --prompt`: 提示类型 (rule/cot/non)
- `-v, --multimodal`: 是否评估多模态模式 (1: 是, 0: 否)
- `-c, --include_context`: 是否包含上下文 (1: 是, 0: 否)
- `--train_set`: 加载训练集用于新颖性计算
- `--max_samples`: 最大评估样本数量

## 3. 大模型评估 (eval_llm.py)

大模型评估使用LLM来评估干扰项质量，提供两种评估方式：

1. **评委评测**: 让大模型选择正确答案，正确率越高表示干扰项质量越差
2. **多维度打分**: 对干扰项的四个维度进行1-5分打分
   - 貌似合理性 (Plausibility)
   - 绝对错误性 (Incorrectness)
   - 区分度 (Distinctiveness)
   - 诊断价值 (Diagnostic value)

### 使用方法

```bash
python eval_llm.py -d scienceqa -m gpt-4 -p cot -v 1 --llm_model_key deepseek --max_samples 100 --token_budget 100000
```

参数说明：
- `-d, --dataset`: 数据集名称
- `-m, --model`: 模型名称
- `-p, --prompt`: 提示类型 (rule/cot/non)
- `-v, --multimodal`: 是否评估多模态模式 (1: 是, 0: 否)
- `--llm_model_key`: LLM模型键名 (api.yaml中的配置)
- `--max_samples`: 最大评估样本数量
- `--token_budget`: Token预算上限
- `--token_log_file`: Token消耗日志文件路径
- `--output_file`: 自定义结果输出文件路径

## 4. 评估工具函数 (llm_eval_utils.py)

`llm_eval_utils.py` 提供了大模型评估所需的核心工具函数：

- `llm_mcq_judge`: 评委评测单个问题
- `batch_llm_mcq_judge`: 批量评委评测
- `evaluate_distractor_multi_dim_llm`: 单个干扰项多维度打分
- `batch_llm_multi_dim_score`: 批量多维度打分
- `summarize_judge_results_by_dimensions`: 汇总评委评测结果
- `summarize_multi_dim_scores`: 汇总多维度打分结果

这些函数支持并发处理、进度条显示、token预算控制等功能。

## 5. 示例评估流程

1. 首先运行自动评估来获取基础指标:
   ```bash
   python eval.py -d scienceqa -m gpt-4 -p cot -v 1 --train_set
   ```

2. 然后运行大模型评估获取深度分析:
   ```bash
   python eval_llm.py -d scienceqa -m gpt-4 -p cot -v 1 --max_samples 200 --token_budget 50000
   ```

3. 查看评估结果和统计数据:
   ```
   # 自动评估结果
   cat ./evaluation/scienceqa-gpt-4-cot-test.json
   
   # 大模型评估结果
   cat ./evaluation/llm_eval-scienceqa-gpt-4-cot-test.json
   ```

## 6. 资源控制和优化

- `--max_samples`: 限制评估样本数量，适用于调试或控制成本
- `--token_budget`: 设置LLM token预算上限
- 大模型评估支持并发处理，可以通过`batch_llm_mcq_judge`和`batch_llm_multi_dim_score`中的`max_workers`参数调整 