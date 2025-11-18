# gtc_align/gcn_encoder.py
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


def normalize_adj(adj: sp.spmatrix) -> torch.Tensor:
    """
    归一化邻接矩阵: A_hat = D^{-1/2} (A + I) D^{-1/2}
    返回 PyTorch 稀疏张量 A_norm
    """
    A = adj.tocsr()
    n = A.shape[0]
    A = A + sp.eye(n, format="csr")  # 加自环

    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    row, col = A.nonzero()
    data = A.data * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    indices = torch.LongTensor([row, col])
    values = torch.FloatTensor(data)
    A_norm = torch.sparse_coo_tensor(indices, values, (n, n))
    return A_norm


def pad_features(X: Optional[np.ndarray], target_dim: int, n_nodes: int):
    """
    把特征矩阵 pad 到统一的输入维度 target_dim。
    X: (n, f) 或 None
    """
    if X is None:
        # 用常数特征
        out = torch.zeros((n_nodes, target_dim), dtype=torch.float32)
        out[:, 0] = 1.0
        return out

    X = np.asarray(X, dtype=np.float32)
    f = X.shape[1]
    if f == target_dim:
        return torch.from_numpy(X)

    if f > target_dim:
        # 截断
        return torch.from_numpy(X[:, :target_dim])

    # f < target_dim，右侧补零
    out = np.zeros((n_nodes, target_dim), dtype=np.float32)
    out[:, :f] = X
    return torch.from_numpy(out)


def two_layer_gcn_encode(
    adj: sp.spmatrix,
    features: Optional[np.ndarray],
    W0: torch.Tensor,
    W1: torch.Tensor,
    in_dim: int,
):
    """
    单图两层 GCN 编码:
    H1 = ReLU(A_norm @ X @ W0)
    H2 = A_norm @ H1 @ W1
    """
    n = adj.shape[0]
    A_norm = normalize_adj(adj)
    X = pad_features(features, target_dim=in_dim, n_nodes=n)  # (n, in_dim)

    # 稀疏乘稠密: (n,n) x (n,in_dim) -> (n,in_dim)
    H = torch.relu(A_norm.matmul(X).matmul(W0))  # (n, hidden_dim)
    H = A_norm.matmul(H).matmul(W1)             # (n, hidden_dim)
    return H


def encode_graphs_with_gcn(
    adj_s: sp.spmatrix,
    adj_t: sp.spmatrix,
    feats_s: Optional[np.ndarray],
    feats_t: Optional[np.ndarray],
    hidden_dim: int = 128,
    seed: int = 42,
):
    """
    对源图和目标图使用共享权重的两层 GCN 编码。
    返回:
        H_s: (n_s, hidden_dim)
        H_t: (n_t, hidden_dim)
    """
    n_s = adj_s.shape[0]
    n_t = adj_t.shape[0]
    f_s = feats_s.shape[1] if feats_s is not None else 1
    f_t = feats_t.shape[1] if feats_t is not None else 1
    in_dim = max(f_s, f_t)

    torch.manual_seed(seed)
    W0 = torch.randn(in_dim, hidden_dim) * 0.1
    W1 = torch.randn(hidden_dim, hidden_dim) * 0.1

    H_s = two_layer_gcn_encode(adj_s, feats_s, W0, W1, in_dim=in_dim)
    H_t = two_layer_gcn_encode(adj_t, feats_t, W0, W1, in_dim=in_dim)

    return H_s, H_t
