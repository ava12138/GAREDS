from calendar import c
import json

def count_json_objects(file_path):
    """统计JSON文件中的对象数量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            count = len(data)
            print(f"文件 {file_path} 包含 {count} 个JSON对象")
            return count
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return 0
# def count_distractors(question_data: dict) -> int:
#     """
#     统计问题数据中的干扰项数量
#     Args:
#         question_data: JSON格式的问题数据字典，包含 question, correct_answer 和 distractor1,2,3... 等字段
#     Returns:
#         int: 干扰项数量
#     """
#     # 方法 1: 直接计数具有 distractor 前缀的键
#     return len([key for key in question_data.keys() if key.startswith('distractor')])

# data = {
#     "question": "Which figure of speech is used in this text?",
#     "correct_answer": "apostrophe",
#     "distractor1": "chiasmus"
# }
# count = count_distractors(data)  # 返回 1
# print(count)
# data2 = {
#     "question": "Which of the following could Gordon's test show?",
#     "correct_answer": "how steady a parachute with a 1 m vent was at 200 km per hour",
#     "distractor1": "if the spacecraft was damaged...",
#     "distractor2": "whether a parachute with a 1 m vent would..."
# }
# count = count_distractors(data2)
# print(count) 
if __name__ == "__main__":
    file_path = '/home/lzx/lib/pro/data_divided/sciqa-test-copy.json'
    count_json_objects(file_path)
    file_path2 = '/home/lzx/lib/pro/output/output_dg-sciqa-all-qwen7b-rule.json'
    count_json_objects(file_path2)