import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kmeans import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 解决负号 '-' 显示为方块的问题

def main():
    '''
    主函数：演示自定义 KMeans 类的聚类效果。
    
    功能概述：
    ----------
    1. 生成二维样本数据；
    2. 使用自定义的 KMeans 算法进行聚类；
    3. 获取聚类结果和质心；
    4. 绘制聚类分布图。
    '''
    
    # ① 生成模拟样本数据
    # make_blobs 用于生成具有指定簇中心的高斯分布样本
    X, y = make_blobs(
        n_samples=300,     # 样本数量
        centers=3,         # 簇（类别）数量
        n_features=2,      # 每个样本的特征维度
        random_state=42,   # 随机种子
        cluster_std=1.0    # 每个簇的标准差（控制簇的离散程度）
    )

    # ② 创建并拟合 KMeans 模型
    # 初始化自定义的 KMeans 聚类器
    kmeans = KMeans(n_clusters=3, random_state=42)
    # 使用训练数据进行聚类
    kmeans.fit(X)

    # ③ 获取聚类结果
    # 每个样本对应的聚类标签（0, 1, 2）
    labels = kmeans.predict(X)
    # 聚类中心（质心）坐标
    centroids = kmeans.centroids

    # ④ 绘制聚类结果
    plt.figure(figsize=(8, 6))
    
    # 绘制每个数据点，颜色由聚类标签决定
    plt.scatter(
        X[:, 0], X[:, 1],
        c=labels,                # 按标签上色
        cmap='viridis',          # 颜色映射
        marker='o',              # 数据点形状
        edgecolor='k',           # 边框颜色
        s=50,                    # 点的大小
        alpha=0.7,               # 透明度
        label='数据点'
    )
    
    # 绘制聚类中心（红色大叉）
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='red',                 # 颜色
        marker='X',              # 标记样式
        s=200,                   # 点大小
        edgecolor='k',           # 边框
        label='质心'
    )
    
    # 图表标题与标签
    plt.title('K-means 聚类结果', fontsize=14)
    plt.xlabel('特征 1', fontsize=12)
    plt.ylabel('特征 2', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # 展示绘制结果
    plt.show()

# 当文件被直接运行时，执行 main()
if __name__ == '__main__':
    main()
