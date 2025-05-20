import random
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os
import threading

# =====================
# 大模型评估工具函数
# =====================

def llm_mcq_judge(client, api_model, question, correct_answer, distractors, max_retries=2):
    """
    让大模型在干扰项和正确答案中选择，返回是否选对。
    自动兼容流式和非流式API。
    
    Args:
        client: OpenAI兼容的API客户端
        api_model: 模型名称
        question: 问题文本
        correct_answer: 正确答案
        distractors: 干扰项列表
        max_retries: 最大重试次数
        
    Returns:
        tuple: (是否选对, 输入token数, 输出token数)
    """
    # 构建多选题prompt，将正确答案和干扰项随机排列
    options = distractors + [correct_answer]
    random.shuffle(options)
    option_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    prompt = f"Question: {question}\nOptions:\n{option_str}\nPlease select the MOST correct answer, and only return the letter of the option."
    correct_index = options.index(correct_answer)
    
    retries = 0
    while retries <= max_retries:
        try:
            # 优先尝试非流式
            input_tokens = 0
            output_tokens = 0
            try:
                response = client.chat.completions.create(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    stream=False
                )
                answer = response.choices[0].message.content.strip().upper()
                if hasattr(response, 'usage'):
                    if hasattr(response.usage, 'prompt_tokens'):
                        input_tokens = response.usage.prompt_tokens
                    if hasattr(response.usage, 'completion_tokens'):
                        output_tokens = response.usage.completion_tokens
                    # 如果只有total_tokens但没有细分，进行估算
                    if hasattr(response.usage, 'total_tokens') and (input_tokens == 0 or output_tokens == 0):
                        input_tokens = int(response.usage.total_tokens * 0.8)  # 估算输入占80%
                        output_tokens = response.usage.total_tokens - input_tokens  # 估算输出占20%
            except Exception as e:
                # 检查是否需要流式
                if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400:
                    err_msg = str(e)
                    if 'only support stream mode' in err_msg or 'please enable the stream parameter' in err_msg:
                        # 切换为流式
                        response = client.chat.completions.create(
                            model=api_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                            stream=True
                        )
                        answer = ""
                        for chunk in response:
                            if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                                delta = chunk.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    answer += delta.content
                        answer = answer.strip().upper()
                        # 流式API通常不返回token计数，使用近似估计
                        # 估算输入token（每个token约4个字符）
                        input_tokens = len(prompt) // 4
                        # 估算输出token
                        output_tokens = len(answer) // 4
                    else:
                        raise e
                else:
                    raise e
            # 只取第一个大写字母
            match = re.search(r"[A-Z]", answer)
            if match:
                pred_index = ord(match.group(0)) - 65
                return pred_index == correct_index, input_tokens, output_tokens
            else:
                return False, input_tokens, output_tokens
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"\033[91mLLM评委API调用失败: {e}\033[0m")
                return False, 0, 0


def batch_llm_mcq_judge(client, api_model, qa_list, max_workers=8, max_samples=None, token_budget=None, 
                        save_path=None, batch_size=10):
    """
    批量评委评测，支持多线程。
    
    Args:
        client: OpenAI API客户端
        api_model: 模型名称
        qa_list: [(question, correct_answer, distractors), ...]
        max_workers: 并发线程数量
        max_samples: 最大评估样本数，用于调试或控制成本
        token_budget: token预算限制，超过则停止评估
        save_path: 结果保存路径，用于断点重传
        batch_size: 批次保存大小，每处理多少个样本保存一次
        
    Returns:
        tuple: (正确数, 总数, 每题是否正确列表, 输入token消耗, 输出token消耗)
    """
    # 设定断点重传
    start_idx = 0
    results = []
    input_tokens_total = 0
    output_tokens_total = 0
    
    # 如果有保存路径，尝试加载之前的结果
    if save_path and os.path.exists(save_path):
        try:
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
                start_idx = saved_data.get('processed_count', 0)
                results = saved_data.get('results', [])
                input_tokens_total = saved_data.get('input_tokens', 0)
                output_tokens_total = saved_data.get('output_tokens', 0)
                print(f"断点重传: 从第{start_idx}个样本继续，已有{len(results)}个结果")
        except Exception as e:
            print(f"加载保存点失败: {e}，将从头开始评估")
            start_idx = 0
            results = []
            input_tokens_total = 0
            output_tokens_total = 0
    
    # 如果结果为空，初始化结果列表
    if not results:
        results = [None] * len(qa_list)
    
    # 如果已经评估完所有样本，直接返回
    if start_idx >= len(qa_list):
        correct_count = sum(1 for r in results if r is True)
        answered_count = sum(1 for r in results if r is not None)
        return correct_count, answered_count, results[:answered_count], input_tokens_total, output_tokens_total
    
    # 限制最大样本数
    if max_samples and max_samples < len(qa_list) - start_idx:
        print(f"⚠️ 已限制评估数量: 最多再评估{max_samples}个样本")
        end_idx = start_idx + max_samples
    else:
        end_idx = len(qa_list)
    
    # 互斥锁保护token计数
    token_mutex = threading.Lock()
    
    # 用于显示token消耗和预估成本的函数
    def show_token_stats(input_tokens, output_tokens):
        input_cost = input_tokens / 1000 * 0.002  # 每千token输入0.004元人民币
        output_cost = output_tokens / 1000 * 0.08  # 每千token输出0.016元人民币
        total_cost = input_cost + output_cost
        return f"消耗tokens: 输入{input_tokens}, 输出{output_tokens}, 总费用约人民币{total_cost:.2f}元"
    
    def worker(idx, q, a, ds):
        nonlocal input_tokens_total, output_tokens_total
        is_correct, input_tokens, output_tokens = llm_mcq_judge(client, api_model, q, a, ds)
        with token_mutex:
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens
        return idx, is_correct, input_tokens, output_tokens
    
    # 优化save_batch_results函数，添加可选的非阻塞选项
    def save_batch_results(current_idx, non_blocking=False):
        nonlocal results, input_tokens_total, output_tokens_total
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 准备要保存的数据
            save_data = {
                'processed_count': current_idx,
                'results': results[:current_idx],
                'input_tokens': input_tokens_total,
                'output_tokens': output_tokens_total,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if non_blocking:
                # 使用线程池进行非阻塞保存
                def _save_func():
                    try:
                        with open(save_path, 'w') as f:
                            json.dump(save_data, f, indent=2)
                    except Exception as e:
                        print(f"\033[93m警告: 保存数据时出错: {e}\033[0m")
                
                threading.Thread(target=_save_func).start()
                print(f"\n保存点(异步): 已处理{current_idx}个样本，{show_token_stats(input_tokens_total, output_tokens_total)}")
            else:
                # 常规阻塞保存
                try:
                    with open(save_path, 'w') as f:
                        json.dump(save_data, f, indent=2)
                    print(f"\n保存点: 已处理{current_idx}个样本，{show_token_stats(input_tokens_total, output_tokens_total)}")
                except Exception as e:
                    print(f"\033[93m警告: 保存数据时出错: {e}\033[0m")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        to_process = [(idx, *qa_list[idx]) for idx in range(start_idx, end_idx)]
        futures = [executor.submit(worker, idx, q, a, ds) for idx, q, a, ds in to_process]
        correct_count = sum(1 for r in results[:start_idx] if r is True)
        total = len(futures)
        
        last_save_idx = start_idx
        
        with tqdm(total=total, desc="LLM评委评测中") as pbar:
            # 使用更高效的队列处理方式
            completed = 0
            for fut in as_completed(futures):
                try:
                    idx, res, input_t, output_t = fut.result()
                    results[idx] = res
                    if res:
                        correct_count += 1
                    
                    # 更新进度条信息
                    completed += 1
                    pbar.update(1)
                    processed_count = start_idx + completed
                    pbar.set_postfix({
                        "正确率": f"{correct_count}/{processed_count} = {correct_count/processed_count:.2f}",
                        "Tokens": f"输入{input_tokens_total},输出{output_tokens_total}"
                    })
                    
                    # 修改批次保存的调用
                    if save_path and processed_count - last_save_idx >= batch_size:
                        # 非阻塞保存，避免进度条卡住
                        save_batch_results(processed_count, non_blocking=True)
                        last_save_idx = processed_count
                    
                    # 检查是否超出token预算
                    if token_budget and (input_tokens_total + output_tokens_total) >= token_budget:
                        print(f"\n⚠️ 已达到token预算上限: {input_tokens_total + output_tokens_total} >= {token_budget}")
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                except Exception as e:
                    print(f"\033[91m处理结果时出错: {e}\033[0m")
                    pbar.update(1)
        
        # 修改最后的保存逻辑
        # 最后保存一次
        if save_path and last_save_idx < start_idx + completed:
            # 最后一次保存使用阻塞方式确保数据被写入
            save_batch_results(start_idx + completed, non_blocking=False)
    
    # 计算最终结果
    correct_count = sum(1 for r in results if r is True)
    answered_count = sum(1 for r in results if r is not None)
    
    # 显示token和成本统计
    print(f"\n📊 {show_token_stats(input_tokens_total, output_tokens_total)}")
    
    return correct_count, answered_count, results[:answered_count], input_tokens_total, output_tokens_total

# =====================
# 大模型多维度打分批量工具
# =====================

def evaluate_distractor_multi_dim_llm(client, api_model, question, correct_answer, distractors, max_retries=2):
    """
    一次性让大模型对四个维度打分，返回四个分数。
    维度：貌似合理性、绝对错误性、区分度、诊断价值。
    自动兼容流式和非流式API。
    
    Args:
        client: OpenAI API客户端 
        api_model: 模型名称
        question: 问题文本
        correct_answer: 正确答案
        distractor: 待评估的干扰项
        max_retries: 最大重试次数
    
    Returns：
        tuple: (dict或None, 输入token消耗, 输出token消耗)
          dict的key为维度名，value为分数（1-5）
    """
    if isinstance(distractors, str):
        distractors = [distractors]
        
    # 构建所有干扰项文本
    distractor_text = ""
    for i, d in enumerate(distractors, 1):
        distractor_text += f"Distractor {i}: {d}\n"
        
    prompt = f"""
Question: {question}
Correct answer: {correct_answer}
Distractor to evaluate: {distractor_text}

Please rate the following four dimensions on a scale of 1-5 (1 being the lowest, 5 being the highest), and ONLY return four numbers separated by commas without any explanation:
1. Plausibility (how reasonable the distractor seems)
2. Incorrectness (how clearly wrong the distractor is)
3. Distinctiveness (how different it is from the correct answer)
4. Diagnostic value (how well it reveals student misconceptions)

Return format example: 3,5,4,2
"""
    retries = 0
    while retries <= max_retries:
        try:
            # 优先尝试非流式
            input_tokens = 0
            output_tokens = 0
            try:
                response = client.chat.completions.create(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    stream=False
                )
                content = response.choices[0].message.content.strip()
                if hasattr(response, 'usage'):
                    if hasattr(response.usage, 'prompt_tokens'):
                        input_tokens = response.usage.prompt_tokens
                    if hasattr(response.usage, 'completion_tokens'):
                        output_tokens = response.usage.completion_tokens
                    # 如果只有total_tokens但没有细分，进行估算
                    if hasattr(response.usage, 'total_tokens') and (input_tokens == 0 or output_tokens == 0):
                        input_tokens = int(response.usage.total_tokens * 0.8)  # 估算输入占80%
                        output_tokens = response.usage.total_tokens - input_tokens  # 估算输出占20%
                else:
                    # 近似估计token数量
                    input_tokens = len(prompt) // 4
                    output_tokens = len(content) // 4
            except Exception as e:
                # 检查是否需要流式
                if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400:
                    err_msg = str(e)
                    if 'only support stream mode' in err_msg or 'please enable the stream parameter' in err_msg:
                        # 切换为流式
                        response = client.chat.completions.create(
                            model=api_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                            stream=True
                        )
                        content = ""
                        for chunk in response:
                            if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                                delta = chunk.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    content += delta.content
                        content = content.strip()
                        # 流式API通常不返回usage，近似估计
                        input_tokens = len(prompt) // 4
                        output_tokens = len(content) // 4
                    else:
                        raise e
                else:
                    raise e
            # 用正则提取四个1-5的数字
            match = re.search(r"([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])", content)
            if match:
                scores = [int(match.group(i)) for i in range(1, 5)]
                return {
                    "plausibility": {"score": scores[0]},
                    "incorrectness": {"score": scores[1]},
                    "distinctiveness": {"score": scores[2]},
                    "diagnostic_value": {"score": scores[3]}
                }, input_tokens, output_tokens
            else:
                print(f"\033[93m警告: 未能从LLM响应中解析出四个分数。响应: {content[:100]}...\033[0m")
                return None, input_tokens, output_tokens
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"\033[91mLLM多维度打分API调用失败: {e}\033[0m")
                return None, 0, 0

def batch_llm_multi_dim_score(client, api_model, qa_list, max_workers=8, max_samples=None, token_budget=None, 
                              log_file=None, save_path=None, batch_size=10):
    """
    批量多维度打分工具
    
    Args:
        client: OpenAI API客户端
        api_model: 模型名称
        qa_list: [(question, correct_answer, distractors列表), ...]
        max_workers: 并发线程数量
        max_samples: 最大评估样本数，用于调试或控制成本
        token_budget: token预算限制，超过则停止评估
        log_file: token消耗日志文件路径
        save_path: 结果保存路径，用于断点重传
        batch_size: 批次保存大小，每处理多少个样本保存一次
        
    Returns:
        tuple: (每题四维度分数字典列表, 输入token消耗, 输出token消耗)
    """
    # 设定断点重传
    start_idx = 0
    results = []
    input_tokens_total = 0
    output_tokens_total = 0
    
    # 如果有保存路径，尝试加载之前的结果
    if save_path and os.path.exists(save_path):
        try:
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
                start_idx = saved_data.get('processed_count', 0)
                results = saved_data.get('results', [])
                input_tokens_total = saved_data.get('input_tokens', 0)
                output_tokens_total = saved_data.get('output_tokens', 0)
                print(f"断点重传: 从第{start_idx}个样本继续，已有{len(results)}个结果")
        except Exception as e:
            print(f"加载保存点失败: {e}，将从头开始评估")
            start_idx = 0
            results = []
            input_tokens_total = 0
            output_tokens_total = 0
    
    # 如果结果为空，初始化结果列表
    if not results:
        results = [None] * len(qa_list)
        
    # 如果已经评估完所有样本，直接返回
    if start_idx >= len(qa_list):
        answered_count = sum(1 for r in results if r is not None)
        return results[:answered_count], input_tokens_total, output_tokens_total
    
    # 限制最大样本数
    if max_samples and max_samples < len(qa_list) - start_idx:
        print(f"⚠️ 已限制评估数量: 最多再评估{max_samples}个样本")
        end_idx = start_idx + max_samples
    else:
        end_idx = len(qa_list)
    
    token_mutex = threading.Lock()  # 用于保护total_tokens的互斥锁
    
    # 创建日志文件目录
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 开始多维度评分，预计处理 {end_idx - start_idx} 个样本\n")
        
    # 记录起始时间，用于周期性记录token消耗
    start_time = time.time()
    last_log_time = start_time
    
    # 用于显示token统计和预估成本的函数
    def show_token_stats(input_tokens, output_tokens):
        input_cost = input_tokens / 1000 * 0.002  # 每千token输入0.004元人民币
        output_cost = output_tokens / 1000 * 0.008  # 每千token输出0.016元人民币
        total_cost = input_cost + output_cost
        return f"消耗tokens: 输入{input_tokens}, 输出{output_tokens}, 总费用约人民币{total_cost:.2f}元"
    
    # 保存批次结果的函数
    def save_batch_results(current_idx, non_blocking=False):
        nonlocal results, input_tokens_total, output_tokens_total
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 准备要保存的数据
            save_data = {
                'processed_count': current_idx,
                'results': results[:current_idx],
                'input_tokens': input_tokens_total,
                'output_tokens': output_tokens_total,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if non_blocking:
                # 使用线程池进行非阻塞保存
                def _save_func():
                    try:
                        with open(save_path, 'w') as f:
                            json.dump(save_data, f, indent=2)
                    except Exception as e:
                        print(f"\033[93m警告: 保存数据时出错: {e}\033[0m")
                
                threading.Thread(target=_save_func).start()
                print(f"\n保存点(异步): 已处理{current_idx}个样本，{show_token_stats(input_tokens_total, output_tokens_total)}")
            else:
                # 常规阻塞保存
                try:
                    with open(save_path, 'w') as f:
                        json.dump(save_data, f, indent=2)
                    print(f"\n保存点: 已处理{current_idx}个样本，{show_token_stats(input_tokens_total, output_tokens_total)}")
                except Exception as e:
                    print(f"\033[93m警告: 保存数据时出错: {e}\033[0m")
    
    def worker(idx, q, a, d):
        nonlocal input_tokens_total, output_tokens_total, last_log_time
        res, input_t, output_t = evaluate_distractor_multi_dim_llm(client, api_model, q, a, d)
        with token_mutex:
            input_tokens_total += input_t
            output_tokens_total += output_t
            # 每60秒记录一次token消耗
            current_time = time.time()
            if log_file and (current_time - last_log_time > 60):
                with open(log_file, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {show_token_stats(input_tokens_total, output_tokens_total)}\n")
                last_log_time = current_time
        return idx, res, input_t, output_t
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        to_process = [(idx, *qa_list[idx]) for idx in range(start_idx, end_idx)]
        futures = [executor.submit(worker, idx, q, a, d) for idx, q, a, d in to_process]
        valid_results = sum(1 for r in results[:start_idx] if r is not None)
        total = len(futures)
        
        last_save_idx = start_idx
        
        with tqdm(total=total, desc="LLM多维度打分中") as pbar:
            # 使用更高效的队列处理方式
            completed = 0
            for fut in as_completed(futures):
                try:
                    idx, res, input_t, output_t = fut.result()
                    results[idx] = res
                    if res:
                        valid_results += 1
                    
                    # 更新进度条信息
                    completed += 1
                    pbar.update(1)
                    processed_count = start_idx + completed
                    pbar.set_postfix({
                        "有效比例": f"{valid_results}/{processed_count} = {valid_results/processed_count:.2f}",
                        "Tokens": f"输入{input_tokens_total},输出{output_tokens_total}"
                    })
                    
                    # 修改批次保存的调用
                    if save_path and processed_count - last_save_idx >= batch_size:
                        # 非阻塞保存，避免进度条卡住
                        save_batch_results(processed_count, non_blocking=True)
                        last_save_idx = processed_count
                    
                    # 检查是否超出token预算
                    if token_budget and (input_tokens_total + output_tokens_total) >= token_budget:
                        print(f"\n⚠️ 已达到token预算上限: {input_tokens_total + output_tokens_total} >= {token_budget}")
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                except Exception as e:
                    print(f"\033[91m处理结果时出错: {e}\033[0m")
                    pbar.update(1)
        
        # 修改最后的保存逻辑
        # 最后保存一次
        if save_path and last_save_idx < start_idx + completed:
            # 最后一次保存使用阻塞方式确保数据被写入
            save_batch_results(start_idx + completed, non_blocking=False)
    
    # 记录最终的token消耗
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 最终{show_token_stats(input_tokens_total, output_tokens_total)}\n")
            valid_count = sum(1 for r in results if r is not None)
            if valid_count > 0:
                effective_count = sum(1 for r in results if r is not None and r)
                f.write(f"有效数据: {effective_count}/{valid_count} = {effective_count/valid_count:.2f}\n")
    
    # 显示token消耗和成本估计
    print(f"\n📊 {show_token_stats(input_tokens_total, output_tokens_total)}")
    
    # 只返回有效结果
    answered_count = sum(1 for r in results if r is not None)
    
    return results[:answered_count], input_tokens_total, output_tokens_total 