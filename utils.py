import json
import os

# 构建文件路径
file_path = os.path.expanduser('/data/lzx/sciq/test.json')

# 读取JSON文件
with open(file_path, 'r') as file:
    data = json.load(file)

# 输出第一个项
if data:
    first_item = data[0]
    print("问题:", first_item.get('question'))
    print("干扰项1:", first_item.get('distractor1'))
    print("干扰项2:", first_item.get('distractor2'))
    print("干扰项3:", first_item.get('distractor3'))
    print("正确答案:", first_item.get('correct_answer'))
else:
    print("文件为空或没有数据")
