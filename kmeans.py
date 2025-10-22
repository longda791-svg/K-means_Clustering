import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        '''
        初始化 KMeans 聚类算法的参数。
        
        参数：
        ----------
        n_clusters : int, 默认=3
            要划分的簇（类别）数量。
        max_iter : int, 默认=100
            最大迭代次数，用于防止算法陷入死循环。
        random_state : int, 默认=42
            随机种子，保证每次运行结果一致。
        
        属性：
        ----------
        centroids : ndarray
            每个簇的中心（质心）坐标。
        labels : ndarray
            每个样本所属的簇标签。
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        '''
        对输入数据 X 进行聚类训练，找到每个簇的质心。
        
        参数：
        ----------
        X : ndarray, 形状为 (n_samples, n_features)
            输入数据，每一行是一个样本。
        
        功能说明：
        ----------
        1. 随机选择初始质心；
        2. 重复以下步骤直到收敛或达到最大迭代次数：
            - 根据当前质心分配样本；
            - 更新每个簇的质心；
            - 如果质心不再变化则提前停止。
        '''
        random.seed(self.random_state)
        
        # 随机初始化质心
        initial_centroid_indices = random.sample(range(X.shape[0]), self.n_clusters)
        self.centroids = X[initial_centroid_indices]

        for _ in range(self.max_iter):
            # 分配簇
            self.labels = self._assign_clusters(X)
            
            # 更新质心
            new_centroids = self._update_centroids(X)
            
            # 检查是否收敛（质心不再变化）
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        '''
        根据当前质心，将每个样本分配到最近的簇。
        
        参数：
        ----------
        X : ndarray, 形状为 (n_samples, n_features)
            输入样本数据。
        
        返回：
        ----------
        labels : ndarray, 形状为 (n_samples,)
            每个样本所属簇的索引编号（0 ~ n_clusters-1）。
        
        逻辑说明：
        ----------
        - 计算每个样本与各质心的欧氏距离；
        - 选取距离最小的质心对应的簇作为样本的类别。
        '''
        # 计算距离矩阵，形状为 (n_clusters, n_samples)
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X):
        '''
        更新每个簇的质心。
        
        参数：
        ----------
        X : ndarray, 形状为 (n_samples, n_features)
            输入样本数据。
        
        返回：
        ----------
        new_centroids : ndarray, 形状为 (n_clusters, n_features)
            每个簇更新后的质心。
        
        逻辑说明：
        ----------
        对于每个簇 i：
        - 取出属于该簇的所有样本；
        - 计算这些样本在各维度上的平均值；
        - 作为新的质心坐标。
        '''
        new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def predict(self, X):
        '''
        对新数据 X 进行预测，返回每个样本所属的簇标签。
        
        参数：
        ----------
        X : ndarray, 形状为 (n_samples, n_features)
            待预测的数据。
        
        返回：
        ----------
        labels : ndarray, 形状为 (n_samples,)
            每个样本的簇标签。
        
        功能说明：
        ----------
        根据训练得到的质心，计算输入样本与质心的距离，
        将样本分配到最近的簇中。
        '''
        return self._assign_clusters(X)

