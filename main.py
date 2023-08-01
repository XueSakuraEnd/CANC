import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KMeans import Kmeans
import time
import collections
from  scipy.stats import chi2_contingency
from Utils import DefaultHandle, Shuffle, Normalized
from CHI import canlinski_harabasz_index
index = ["CSBP(mmHg)", "SBP(mmHg)", "DBP(mmHg)", "PP(mmHg)",
         "SBP2反射波峰的压力值(mmHg)", "AI(%)", "AI P75脉率为75时的AI值（%）", "pulse(bpm)"]
xlabel = ["CSBP(mmHg)", "SBP(mmHg)", "DBP(mmHg)", "PP(mmHg)",
          "SBP2(mmHg)", "AI(%)", "AI P75", "pulse(bpm)"]


def getData(normalized=True):
    # 得到数据清洗后的文件
    file = DefaultHandle(index)
    init_data = np.array(file.loc[:,:])
    # 获得目标数据
    data = np.array(file.loc[:, "CSBP(mmHg)":"pulse(bpm)"])
    # plt.subplot(1, 2, 1)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.title("init")
    # plt.xlabel(index[0])
    # plt.ylabel(index[1])
    # 进行洗牌
    data = Shuffle(data)
    if normalized:
        # 进行标准化
        data = Normalized(data)
    # plt.subplot(1, 2, 2)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.title("processed")
    # plt.xlabel(index[0])
    # plt.ylabel(index[1])
    # plt.xlim((0, 20))
    # plt.ylim((0, 20))
    # plt.subplots_adjust(wspace=1, hspace=0)  # 调整子图间距
    # plt.show()
    return data,init_data

# 测试test


def testk(data):
    dit = {i: 0 for i in range(2, 9)}
    for i in range(50):
        print(f"第{i+1}轮")
        for k in range(2, 9):
            kmeansModel = Kmeans(data, k)
            (centroids,
             closest_centroids_ids) = kmeansModel.train(100)
            s = canlinski_harabasz_index(
                data, closest_centroids_ids.flatten())
            dit[k] += s
    ls = list(dit.values())
    ls.insert(0, 0)
    plt.plot([i for i in range(1, 9)], [i/50 for i in ls])
    plt.title("CHI with k")
    plt.xlabel("K")
    plt.ylabel("CHI")
    plt.show()

# 测试初始化方法


def testInit(data, k):
    dit = {i: 0 for i in range(2)}
    for i in range(50):
        print(f"第{i+1}轮")
        kmeansModel = Kmeans(data, k)
        (centroids,
         closest_centroids_ids) = kmeansModel.train(100, init_function="random_init")
        s = canlinski_harabasz_index(
            data, closest_centroids_ids.flatten())
        dit[0] += s
        (centroids,
         closest_centroids_ids) = kmeansModel.train(100, init_function="kmeans++")
        s = canlinski_harabasz_index(
            data, closest_centroids_ids.flatten())
        dit[1] += s

    ls = list(dit.values())
    print(ls)
    rects = plt.bar(["random_init", "kmeans++"], [i/50 for i in ls], width=0.3)
    plt.title("different init function")
    for rect in rects:  # rects 是三根柱子的集合
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height,
                 str(height), size=8, ha='center', va='bottom')
    plt.show()

# 测试时间


def testTime(data, k):
    dit = {i: 0 for i in range(2)}
    for i in range(50):
        print(f"第{i+1}轮")
        start = time.time()
        kmeansModel = Kmeans(data, k)
        (centroids,
         closest_centroids_ids) = kmeansModel.train(100, init_function="random_init")
        end = time.time()
        dit[0] += end-start
        start = time.time()
        (centroids,
         closest_centroids_ids) = kmeansModel.train(100, init_function="kmeans++")
        end = time.time()
        dit[1] += end-start

    ls = list(dit.values())
    print(ls)
    rects = plt.bar(["random_init", "kmeans++"], [i/50 for i in ls], width=0.3)
    plt.title("different init function take time")
    for rect in rects:  # rects 是三根柱子的集合
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height,
                 str(height), size=8, ha='center', va='bottom')
    plt.show()


if __name__ == "__main__":
    train_data,init_data = getData()
    k = 2
    kmeansModel = Kmeans(train_data, k)
    (centroids,
     closest_centroids_ids) = kmeansModel.train(100, init_function="random_init")
    s = canlinski_harabasz_index(
        train_data, closest_centroids_ids.flatten())
    data,init_data = getData(normalized=False)
    for i in range(k):
        ls = np.ravel(closest_centroids_ids == i)
        for j in range(len(index)):
            plt.subplot(2, 4, j+1)
            plt.hist(data[ls, j], bins=30, alpha=0.5, label=f'Clust {i+1}')
            plt.title('Different Clust')
            plt.xlabel(f'{xlabel[j]} value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.subplots_adjust(wspace=1, hspace=1)  # 调整子图间距
    # 显示图表
    plt.show()
    kf_ls = []
    # 血压
    for i in range(k):
        ls = np.ravel(closest_centroids_ids == i)
        blood_pleasure = [i for i in list(collections.Counter(init_data[ls,6]).values())]
        plt.subplot(1,2,i+1)
        plt.pie(blood_pleasure,labels=['H','L'], # 设置饼图标签
        colors=["#d5695d", "#5d8ca8"],
        autopct='%1.1f%%') # 设置饼图颜色)
        plt.title(f"clust {i+1} blood pleasure")
        kf_ls.append(blood_pleasure)
    plt.show()
    kf = chi2_contingency(np.array(kf_ls))
    print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)