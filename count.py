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

if __name__ == "__main__":
    file_path = '/data/lzx/sciq/test.json'
    count_json_objects(file_path)