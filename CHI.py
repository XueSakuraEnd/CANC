import numpy as np
import collections

def canlinski_harabasz_index(data,label):
    # 不同的label
    y_label = [i for i in list(collections.Counter(label).values())]
    # label数量
    n_label = len(y_label)
    # 初始化簇间,簇内协方差的值
    bk_tr,wk_tr = 0.0,0.0
    # 全局平均值
    mean = np.mean(data,axis=0)
    # 样本数量
    num = data.shape[0]
    for i in range(n_label):
        # 当前分类的样本
        current_clust = data[label == i]
        # 当前簇中心
        clust_mean = np.mean(current_clust,axis=0)
        # 簇间协方差计算
        bk_tr += len(current_clust) * np.sum((clust_mean-mean) ** 2)
        # 簇内协方差计算
        wk_tr += np.sum((current_clust-clust_mean) ** 2)
    # CHI计算
    s = (bk_tr * (num-n_label)) / (wk_tr * (n_label-1))
    return s
