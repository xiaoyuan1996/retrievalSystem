import yaml
import numpy as np
import pickle
import os,random

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

# 创建文件夹
def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

# 删除文件夹及对应文件
def delete_dire(dire):
    dir_list = []
    for root, dirs, files in os.walk(dire):
        for afile in files:
            os.remove(os.path.join(root, afile))
        for adir in dirs:
            dir_list.append(os.path.join(root, adir))
    for bdir in dir_list:
        os.rmdir(bdir)

# 保存结果到txt文件
def log_to_txt( contexts=None,filename="save.txt", mark=False,encoding='UTF-8',add_n=False,mode='a'):
    f = open(filename, mode,encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c)+" | "+ str(contexts[c]) +"\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts,list):
            tmp = ""
            for c in contexts:
                if add_n:
                    tmp += str(c) + " " + "\n"
                else:
                    tmp += str(c) + " "
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)


    f.close()

# 从txt中读取行
def load_from_txt(filename, encoding="utf-8"):
    f = open(filename,'r' ,encoding=encoding)
    contexts = f.readlines()
    return contexts

# 创建随机文件夹
def create_random_dirs_name(dir_path):
    dirs = os.listdir(dir_path)
    new_dir = ""
    while (new_dir == "") or (new_dir in dirs):
        new_dir = "".join(random.sample('1234567890qwertyuiopasdfghjklzxcvbnm', 8))
    return new_dir