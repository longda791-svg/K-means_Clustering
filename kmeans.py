import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        random.seed(self.random_state)
        
        # 随机初始化质心
        initial_centroid_indices = random.sample(range(X.shape[0]), self.n_clusters)
        self.centroids = X[initial_centroid_indices]

        for _ in range(self.max_iter):
            # 分配簇
            self.labels = self._assign_clusters(X)
            
            # 更新质心
            new_centroids = self._update_centroids(X)
            
            # 检查是否收敛
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X):
        new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def predict(self, X):
        return self._assign_clusters(X)
