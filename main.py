import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kmeans import KMeans

def main():
    # 生成样本数据
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.0)

    # 创建并拟合K-means模型
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # 获取预测结果和质心
    labels = kmeans.predict(X)
    centroids = kmeans.centroids

    # 绘制结果
    plt.figure(figsize=(8, 6))
    
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.7, label='数据点')
    
    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolor='k', label='质心')
    
    plt.title('K-means 聚类')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
