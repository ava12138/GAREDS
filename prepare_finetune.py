# def prepare_reasoning_data(problems):
#     """准备推理路径生成的训练数据"""
#     training_data = []
#     for problem in problems:
#         # 基于六个原则生成的多样化推理路径
#         for principle in PRINCIPLES:
#             example = {
#                 "conversations": [
#                     {
#                         "from": "user",
#                         "value": f"基于{principle}原则，为以下问题生成一个推理路径:\n{problem['question']}"
#                     },
#                     {
#                         "from": "assistant",
#                         "value": problem['reasoning_paths'][principle]
#                     }
#                 ]
#             }
#             training_data.append(example)
#     return training_data

# def prepare_distractor_data(problems):
#     """准备干扰项生成的训练数据"""
#     training_data = []
#     for problem in problems:
#         # 每个推理路径对应的高质量干扰项
#         for reasoning in problem['reasoning_paths']:
#             example = {
#                 "conversations": [
#                     {
#                         "from": "user",
#                         "value": f"根据以下推理路径生成与正确答案最相似的干扰项:\n推理:{reasoning}\n正确答案:{problem['answer']}"
#                     },
#                     {
#                         "from": "assistant",
#                         "value": problem['distractors'][reasoning]
#                     }
#                 ]
#             }
#             training_data.append(example)
#     return training_data