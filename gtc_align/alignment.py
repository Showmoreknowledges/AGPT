# gtc_align/alignment.py
from typing import Optional, List, Dict

import numpy as np
import scipy.sparse as sp
import torch

from .centrality import compute_centrality_scores
from .gcn_encoder import encode_graphs_with_gcn


def build_gtc_mask(gamma_s: np.ndarray, gamma_t: np.ndarray, L1: float = 1.05) -> np.ndarray:
    """
    构建全局拓扑一致性掩码 M: shape (n_s, n_t), dtype=bool
    满足 max(Gs/Gt, Gt/Gs) <= L1 的节点对为 True。
    """
    gs = gamma_s[:, None]  # (n_s,1)
    gt = gamma_t[None, :]  # (1,n_t)

    valid = (gs > 0) & (gt > 0)
    ratio = np.maximum(gs / gt, gt / gs)
    mask = (ratio <= L1) & valid
    return mask


def compute_topk_candidates(
    H_s: torch.Tensor,
    H_t: torch.Tensor,
    mask: np.ndarray,
    k: int = 5,
) -> List[List[int]]:
    """
    在嵌入空间中计算相似度并选取 Top-k 候选。
    H_s: (n_s, d), H_t: (n_t, d), mask: (n_s, n_t) bool
    返回: candidates[u] = [v1, v2, ...] (长度<=k)
    """
    n_s, d = H_s.shape
    n_t = H_t.shape[0]

    # 归一化
    H_s_norm = H_s / (H_s.norm(dim=1, keepdim=True) + 1e-8)
    H_t_norm = H_t / (H_t.norm(dim=1, keepdim=True) + 1e-8)

    sim = H_s_norm.matmul(H_t_norm.T)   # (n_s, n_t)
    mask_t = torch.from_numpy(mask)     # bool tensor
    neg_inf = -1e9
    sim[~mask_t] = neg_inf

    # 每行 top-k
    topk_vals, topk_idx = torch.topk(sim, k=min(k, n_t), dim=1)
    candidates: List[List[int]] = []
    for u in range(n_s):
        # 过滤掉 -inf（说明该源点在 mask 下可匹配节点 < k）
        vals_u = topk_vals[u]
        idx_u = topk_idx[u]
        valid_pos = vals_u > neg_inf / 2
        candidates.append(idx_u[valid_pos].cpu().tolist())
    return candidates


def gtc_align(
    adj_s: sp.spmatrix,
    adj_t: sp.spmatrix,
    feats_s: Optional[np.ndarray] = None,
    feats_t: Optional[np.ndarray] = None,
    L1: float = 1.05,
    hidden_dim: int = 128,
    topk: int = 10,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    一步到位的对齐主干函数：
    输入两张图的邻接 & 可选特征，输出每个源节点的候选目标节点列表。
    返回:
        mapping: dict {u_src: [v_tgt1, v_tgt2, ...]}
    """
    # 1. 计算全局中心性 Gamma
    gamma_s = compute_centrality_scores(adj_s)
    gamma_t = compute_centrality_scores(adj_t)

    # 2. 构建 GTC 掩码
    mask = build_gtc_mask(gamma_s, gamma_t, L1=L1)  # (n_s, n_t), bool

    # 3. 两层共享 GCN 编码
    H_s, H_t = encode_graphs_with_gcn(
        adj_s=adj_s,
        adj_t=adj_t,
        feats_s=feats_s,
        feats_t=feats_t,
        hidden_dim=hidden_dim,
        seed=seed,
    )

    # 4. 相似度 + Top-k 候选
    cands = compute_topk_candidates(H_s, H_t, mask, k=topk)

    mapping = {u: cands[u] for u in range(len(cands))}
    return mapping
