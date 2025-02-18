import json

def process_sciqa_data(input_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按subject分类（使用缩写）
    categorized_data = {
        'lan': [],  # language science
        'nat': [],  # natural science
        'soc': []   # social science
    }
    
    # 只按subject分类数据
    for item in data:
        # 按subject分类
        if item['subject'] == 'language science':
            categorized_data['lan'].append(item)
        elif item['subject'] == 'natural science':
            categorized_data['nat'].append(item)
        elif item['subject'] == 'social science':
            categorized_data['soc'].append(item)
    
    # 保存分类后的数据
    for category, items in categorized_data.items():
        output_file = f'/home/lzx/lib/pro/data_divided/sciqa-test-{category}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"{category}: {len(items)} items")

if __name__ == "__main__":
    input_file = '/home/lzx/lib/pro/data_divided/sciqa-test.json'
    process_sciqa_data(input_file)