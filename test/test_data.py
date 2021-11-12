import pickle

# 读取字典
def dict_load(name="rsd.pkl"):
    with open(name, 'rb') as f:
        return pickle.load(f)

pkl = dict_load("../data/retrieval_system_data/rsd/rsd.pkl")

for k,v in pkl.items():
    print(k)
    print(v)
    print("========")