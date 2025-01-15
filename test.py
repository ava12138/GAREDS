import json

def process_data(data):
    processed = []  # 使用列表存储处理后的数据
    for item in data:  # 直接遍历列表
        # 跳过没有必要字段的数据
        if not all(field in item for field in ['question', 'choices', 'answer', 'lecture', 'subject', 'topic', 'category']):
            continue
            
        # 创建新的数据项
        new_item = {
            'question': item['question'],
            'correct_answer': item['choices'][item['answer']],
            'support': item['lecture'],
            'subject': item['subject'],
            'topic': item['topic'],
            'category': item['category']
        }
        
        # 添加干扰项
        choices = item['choices']
        distractor_idx = 1
        for i, choice in enumerate(choices):
            if i != item['answer']:
                new_item[f'distractor{distractor_idx}'] = choice
                distractor_idx += 1
                
        processed.append(new_item)
    return processed

def split_and_save_data(data):
    train_data = []
    valid_data = []
    test_data = []
    
    for key, item in data.items():
        if item.get('split') == 'train':
            train_data.append(item)
        elif item.get('split') == 'valid':
            valid_data.append(item)
        elif item.get('split') == 'test':
            test_data.append(item)
            
    # 处理每个数据集
    processed_train = process_data(train_data)
    processed_valid = process_data(valid_data)
    processed_test = process_data(test_data)
    
    # 保存处理后的数据
    with open('./evaluation/sciqa-train.json', 'w', encoding='utf-8') as f:
        json.dump(processed_train, f, indent=2, ensure_ascii=False)
    with open('./evaluation/sciqa-valid.json', 'w', encoding='utf-8') as f:
        json.dump(processed_valid, f, indent=2, ensure_ascii=False)
    with open('./evaluation/sciqa-test.json', 'w', encoding='utf-8') as f:
        json.dump(processed_test, f, indent=2, ensure_ascii=False)

# 读取原始数据
with open('./evaluation/problems.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理并保存数据
split_and_save_data(data)