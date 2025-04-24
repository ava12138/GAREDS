import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第7号GPU
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Lhh123/coe_multitask_blip2xl_angle_2ep', trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(
    'Lhh123/coe_multitask_blip2xl_angle_2ep', # path to the output directory
    device_map="cuda",
    trust_remote_code=True
).eval()

# query = "Picture: <img>image.png</img>\nGenerate a question based on the picture."
# query_question = "Answer: yes\nRefer to the following example, generate a question based on the corresponding answer.\nExample:\nAnswer: yes\nQuestion: Is this a sentence fragment?\nIn the fifth and sixth centuries, more than fifty thousand Buddhist statues carved into the rock of the Yungang Grottoes of China"
query_rationale = "Question: Would you find the word scissors on a dictionary page with the following guide words? slam -s1ip \nAnswer: yes\nRefer to the following example, how to make a reasoning to answer the question based on the above question with its answer?\nExample:\nQuestion: Is this a sentence fragment?\nIn the fifth and sixth centuries, more than fifty thousand Buddhist statues carved into the rock of the Yungang Grottoes of China.\nAnswer: yes\nReasoning: A sentence is a group of words that expresses a complete thought.\nThe band I'm in has been rehearsing daily because we have a concert in two weeks.\nA sentence fragment is a group of words that does not express a complete thought.\nRehearsing daily because we have a concert in two weeks.\nThis fragment is missing a subject. It doesn't tell who is rehearsing.\nThe band I'm in.\nThis fragment is missing a verb. It doesn't tell what the band I'm in is doing.\nBecause we have a concert in two weeks.\nThis fragment is missing an independent clause. It doesn't tell what happened because of the concert.This is a sentence fragment. It does not express a complete thought.\nIn the fifth and sixth centuries, more than fifty thousand Buddhist statues carved into the rock of the Yungang Grottoes of China.\nHere is one way to fix the sentence fragment:\nIn the fifth and sixth centuries, more than fifty thousand Buddhist statues were carved into the rock of the Yungang Grottoes of China."
query_distractor = "Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n—Homer, The Iliad"
response, history = model.chat(tokenizer, query=query_distractor, history=None)
print(response)

