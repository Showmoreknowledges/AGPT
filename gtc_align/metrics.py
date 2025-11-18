# gtc_align/metrics.py
from typing import Dict, List, Sequence, Tuple
import numpy as np


def evaluate_hit_mrr(
    mapping: Dict[int, List[int]],
    aligned_pair: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
):
    """
    使用 Hit@k 和 MRR 对对齐结果进行评估。

    参数:
        mapping: dict {u_src: [v_tgt1, v_tgt2, ...]}，候选列表须按相似度降序排序
        aligned_pair: np.ndarray, shape (M, 2)，每行是 (u_gt, v_gt)
        ks: 需要计算的 k 值列表，例如 (1,5,10)

    返回:
        hit_at_k: dict {k: value}，每个 k 对应一个 Hit@k
        mrr: float，整体 MRR
    """
    ks = sorted(set(int(k) for k in ks))
    hit_cnt = {k: 0 for k in ks}
    rr_sum = 0.0
    total = aligned_pair.shape[0]

    for u_gt, v_gt in aligned_pair:
        u = int(u_gt)
        v = int(v_gt)
        cand_list = mapping.get(u, [])

        rank = None
        # 找 v 在 cand_list 中的排名（从 1 开始）
        for idx, v_pred in enumerate(cand_list):
            if v_pred == v:
                rank = idx + 1
                break

        # Hit@k
        for k in ks:
            if rank is not None and rank <= k:
                hit_cnt[k] += 1

        # MRR
        if rank is not None:
            rr_sum += 1.0 / rank
        else:
            rr_sum += 0.0

    hit_at_k = {k: hit_cnt[k] / total for k in ks}
    mrr = rr_sum / total
    return hit_at_k, mrr