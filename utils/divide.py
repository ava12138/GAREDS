import json

def filter_json_data(input_file):
    """筛选有三个干扰项且没有distractor4的数据"""
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤数据
    filtered_data = []
    for item in data:
        if all(key in item for key in ['distractor1', 'distractor2', 'distractor3']) and 'distractor4' not in item:
            filtered_data.append(item)
    
    # 保存过滤后的数据
    output_file = './data_divided/sciqa-test-text.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    print(f"Filtered data: {len(filtered_data)} items")
    return len(filtered_data)

def process_sciqa_data(input_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按subject分类
    categorized_data = {
        'language_science': [],
        'natural_science': [],
        'social_science': []
    }
    
    # 过滤和分类数据
    for item in data:
        # 检查条件：
        # 1. 必须有3个干扰项
        # 2. 不能有distractor4
        if not all(key in item for key in ['distractor1', 'distractor2', 'distractor3']) or 'distractor4' in item:
            continue
            
        # 按subject分类
        if item['subject'] == 'language science':
            categorized_data['language_science'].append(item)
        elif item['subject'] == 'natural science':
            categorized_data['natural_science'].append(item)
        elif item['subject'] == 'social science':
            categorized_data['social_science'].append(item)
    
    # 保存分类后的数据
    for category, items in categorized_data.items():
        output_file = f'./data_divided/sciqa-test-{category}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"{category}: {len(items)} items")

if __name__ == "__main__":
    input_file = './data_divided/sciqa-test.json'
    # process_sciqa_data(input_file)
    filter_json_data(input_file)