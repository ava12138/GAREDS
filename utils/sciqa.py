from datasets import load_dataset

ds = load_dataset("derek-thomas/ScienceQA")
print(ds["train"][0])  # 查看第一条数据样例