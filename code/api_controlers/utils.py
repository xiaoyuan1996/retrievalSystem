import yaml
import numpy as np
import pickle
import os

# 加载参数
def get_config(path_opt="common/config.yaml"):
    with open(path_opt, 'r') as handle:
        options = yaml.load(handle)
    return options

# 储存为npy文件
def save_to_npy(info, filename):
    np.save(filename,info,allow_pickle=True)

# 从npy中读取
def load_from_npy(filename):
    info = np.load(filename, allow_pickle=True)
    return info

# 得到标准返回
def get_stand_return(flag, message):
    code = 200 if flag else 400
    return_json = {
        'code':code,
        "message": message
    }
    return return_json

# ==========================   字典操作   =============================
# 保存字典
def dict_save(obj, name="rsd.pkl"):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# 读取字典
def dict_load(name="rsd.pkl"):
    with open(name, 'rb') as f:
        return pickle.load(f)

# 删除键值
def dict_delete(k, rsd):
    rsd.pop(k)
    return rsd

# 插入键值
def dict_insert(k, v, rsd):
    rsd[k] = v
    return rsd

# 按照键值进行排序  输出排序好的键
def sort_based_values(sims_dict):
    """
    :param sims_dict: dict
    {
    '1': 0.2,
    "2": 0.4,
    "3": 0.3,
    "4": 0.23
    }
    :return: key 降序  ['2', '3', '4', '1']
    """
    sims_dict = sorted(sims_dict.items(), key=lambda item: item[1])[::-1]
    return [i[0] for i in sims_dict]


# ============================ 加载或新建 retrieval system data =========================
def init_rsd(file_path):
    if os.path.exists(file_path):
        rsd = dict_load(file_path)
    else:
        rsd = {}
    return rsd