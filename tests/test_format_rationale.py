import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import format_rationale_output

def test_format_rationale():
    # 测试样例
    test_response = """ Explanation: The mushroom in the food web has an arrow pointing to it from the lichen, indicating that the mushroom consumes the lichen. Therefore, the mushroom contains matter that was once part of the lichen.

Incorrect Inference1: The bilberry contains matter that was once part of the lichen.
Incorrect Inference2: The barren-ground caribou contains matter that was once part of the lichen.
Incorrect Inference3: The grizzly bear contains matter that was once part of the lichen.
Incorrect Inference4: The Arctic fox contains matter that was once part of the lichen.
Incorrect Inference5: The earthworm contains matter that was once part of the lichen.
Incorrect Inference6: The collared lemming contains matter that was once part of the lichen."""

    # 测试rule_format格式化
    result_rule = format_rationale_output(test_response, "rule_format")
    print("\n=== Testing rule_format ===")
    print("Explanation:", result_rule.get('explanation', ''))
    print("Incorrect inferences:\n", result_rule.get('incorrect_inferences', ''))
    
    # 测试cot_format格式化
    result_cot = format_rationale_output(test_response, "cot_format")
    print("\n=== Testing cot_format ===")
    print("CoT output:", result_cot)

def main():
    print("开始测试推理格式化函数...")
    test_format_rationale()
    print("\n测试完成!")

if __name__ == "__main__":
    main()