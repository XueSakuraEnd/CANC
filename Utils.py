import pandas as pd
import numpy as np
# 数据处理部分

# 缺省值处理
def DefaultHandle(index):
    file = pd.read_csv("./无创中心动脉压入组患者数据.csv");
    file.dropna(subset=index, inplace = True)
    return file

# 数据洗牌
def Shuffle(data):
    np.random.shuffle(data)
    return data

# 归一化
def Normalized(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    data = (data - mean) / std
    return data

