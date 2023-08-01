# K均值聚类
import numpy as np
import matplotlib.pyplot as plt
import random
class Kmeans:
    def __init__(self, data, num_clusters):

        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations,init_function="kmeans++"):
        # 初始化簇中心
        centroids = Kmeans.centroids_init(self.data, self.num_clusters,init_function)
        num_examples = self.data.shape[0]
        last_clust_result = closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # 找到每个样本点的最近簇中心
            closest_centroids_ids = Kmeans.centroids_find_closest(self.data, centroids)
            # 进行簇中心的位置更新
            centroids = Kmeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
            # 如果与上一轮结果相同则直接返回
            if (closest_centroids_ids==last_clust_result).all():
                return centroids, closest_centroids_ids
            last_clust_result = closest_centroids_ids
        return centroids, closest_centroids_ids

    # 初始化簇中心
    @staticmethod
    def centroids_init(data, num_clusters,init_function):
        if init_function == "random_init":
            num_examples = data.shape[0]
            random_ids = np.random.permutation(num_examples)
            centroids = data[random_ids[:num_clusters], :]
            # plt.scatter(centroids[:,0],centroids[:,1])
            # plt.xlim((-5,5))
            # plt.ylim((-5,5))
            # plt.title("init")
            # plt.show()
            return centroids
        elif init_function == "kmeans++":
            num_sample,num_feature = data.shape
            choice_ls = []
            # 随机选取一个作为初始点
            init_choice = np.random.choice(range(num_sample))
            choice_ls.append(init_choice)
            centroids = []
            centroids.append(data[init_choice])

            for i in range(num_clusters-1):
                dists = {i:0 for i in range(num_sample)}
                total = 0
                for j in range(num_sample):
                    if j in choice_ls:
                        dists[j] = 0
                        continue
                    dists[j] = Kmeans.get_distance(data[j],centroids[i])
                    total += dists[j]
                # 概率
                seed = random.random()
                
                for key,value in dists.items():
                    dists[key] = value / total
                    if key > 0:
                        dists[key] += dists[key-1]
                    if dists[key] > seed:
                        centroids.append(data[key])
                        break

            centroids = np.array(centroids).reshape(len(centroids),num_feature)
            # plt.scatter(centroids[:,0],centroids[:,1])
            # plt.title("kmeans++")
            # plt.xlim((-5,5))
            # plt.ylim((-5,5))
            # plt.show()
            return centroids
            


    # 计算每个样本点的簇
    @staticmethod
    def centroids_find_closest(data, centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                # distance_diff = data[example_index, :] - centroids[centroid_index, :]
                # distance[centroid_index] = np.sum(distance_diff**2)
                distance[centroid_index] = Kmeans.get_distance(data[example_index, :],centroids[centroid_index, :])
            closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids

    # 进行簇中心位置更新
    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters, num_features))
        for centroid_id in range(num_clusters):
            closest_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centroids

    # 求欧式距离
    @staticmethod
    def get_distance(x1,x2):
        return np.sqrt(np.sum(np.square(x1-x2)))