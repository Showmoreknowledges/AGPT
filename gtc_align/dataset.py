import os
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp


def build_adj_from_edge_index(
    edge_index: np.ndarray,
    num_nodes: int,
    undirected: bool = True,
):
    """
    根据 edge_index 构建邻接矩阵 A。
    edge_index: shape (2, E)，第一行是源节点，第二行是目标节点。
    num_nodes: 节点总数。
    undirected: 若为 True，则构建无向图（A 对称）。
    """
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index 形状应为 (2, E)，当前为 {edge_index.shape}")

    row = edge_index[0]
    col = edge_index[1]
    data = np.ones(row.shape[0], dtype=np.float32)

    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    if undirected:
        A = A + A.T
        A[A > 0] = 1.0
    return A.tocsr()


def load_alignment_npz(
    dataset_name: str,
    root_dir: str = "data",
    use_features: bool = True,
) -> Tuple[sp.csr_matrix, sp.csr_matrix,
           Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    通用加载函数：只需给出数据集名字，即可从 {root_dir}/{dataset_name}.npz 中加载图对齐数据。

    约定 .npz 内统一包含以下键：
        - edge_index1: 第一张图的边，形状 (2, E1)
        - edge_index2: 第二张图的边，形状 (2, E2)
        - x1: 第一张图的节点特征，形状 (N1, F)
        - x2: 第二张图的节点特征，形状 (N2, F)
        - aligned_pair: 真实对齐对，形状 (M, 2)

    参数:
        dataset_name: 数据集名（如 "ACM-DBLP", "douban"）
        root_dir: 存放 npz 的目录（默认为 "data"）
        use_features: 是否使用 x1/x2 作为 GCN 输入特征

    返回:
        adj_s: 第一张图邻接矩阵 (csr_matrix)
        adj_t: 第二张图邻接矩阵 (csr_matrix)
        feats_s: 第一张图特征 (np.ndarray or None)
        feats_t: 第二张图特征 (np.ndarray or None)
        aligned_pair: 真实对齐对 (np.ndarray, shape (M, 2))
    """
    filename = f"{dataset_name}.npz"
    path = os.path.join(root_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据集文件: {path}")

    data = np.load(path, allow_pickle=True)

    required_keys = ["edge_index1", "edge_index2", "x1", "x2", "aligned_pair"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"{path} 中缺少必要键: '{k}'，请检查 npz 结构是否统一。")

    edge_index1 = data["edge_index1"]
    edge_index2 = data["edge_index2"]
    x1 = data["x1"]
    x2 = data["x2"]
    aligned_pair = data["aligned_pair"]

    n1 = x1.shape[0]
    n2 = x2.shape[0]

    adj_s = build_adj_from_edge_index(edge_index1, num_nodes=n1, undirected=True)
    adj_t = build_adj_from_edge_index(edge_index2, num_nodes=n2, undirected=True)

    if use_features:
        feats_s = x1.astype(np.float32)
        feats_t = x2.astype(np.float32)
    else:
        feats_s = None
        feats_t = None

    return adj_s, adj_t, feats_s, feats_t, aligned_pair