# gtc_align/centrality.py
import numpy as np
import networkx as nx
import scipy.sparse as sp


def compute_centrality_scores(adj: sp.spmatrix):
    """
    按 GTCAlign 论文计算综合中心性 Gamma：
    - Degree centrality
    - Closeness centrality
    - Betweenness centrality
    - Eigenvector centrality
    输入:
        adj: scipy.sparse 稀疏邻接矩阵 (n x n)，无向图
    返回:
        gamma: np.ndarray, shape (n,), 每个节点的综合中心性
    """
    # 转为 NetworkX 图
    # 如果你的 networkx 版本较旧，用 from_scipy_sparse_matrix 也可以
    G = nx.from_scipy_sparse_array(adj)

    deg_c = nx.degree_centrality(G)            # 度中心性
    clo_c = nx.closeness_centrality(G)         # 接近中心性
    bet_c = nx.betweenness_centrality(G)       # 介数中心性
    eig_c = nx.eigenvector_centrality_numpy(G) # 特征向量中心性

    n = adj.shape[0]
    gamma = np.zeros(n, dtype=np.float64)
    for i in range(n):
        gamma[i] = (
            deg_c.get(i, 0.0)
            + clo_c.get(i, 0.0)
            + bet_c.get(i, 0.0)
            + eig_c.get(i, 0.0)
        ) / 4.0
    return gamma
