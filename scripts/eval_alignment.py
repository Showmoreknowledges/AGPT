"""
Evaluate alignment mapping with Hit@k and MRR.

示例用法：
    python eval_alignment.py \
      --npz_path data/douban.npz \
      --mapping_json results/douban_reranked_gpt4.1.json \
      --ks 1,5,10
"""

import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Sequence, Tuple

import numpy as np


def load_aligned_pairs(npz_path: str, key: str = "aligned_pair") -> np.ndarray:
    """
    从 .npz 中读取真实对齐对 aligned_pair，形状应为 (M, 2)
    每一行是 (u_src, v_tgt)
    """
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"npz file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path}. Available keys: {data.files}")

    arr = data[key]
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"aligned_pair should have shape (M, 2), got shape {arr.shape} from key '{key}'."
        )
    return arr.astype(int)


def load_mapping(mapping_json_path: str):
    """
    从精排结果 JSON 中读取 mapping。
    兼容两种格式：
      1) 直接就是 { "0": [3, 5, 10], ... }
      2) 包在顶层的 "mapping" 字段里：
         { "mapping": { "0": [3, 5, 10], ... }, "scores": ... }
    """
    if not osp.exists(mapping_json_path):
        raise FileNotFoundError(f"mapping_json not found: {mapping_json_path}")

    with open(mapping_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 如果有 "mapping" 字段，就用它；否则假定最外层就是 mapping
    if isinstance(raw, dict) and "mapping" in raw:
        raw_map = raw["mapping"]
    else:
        raw_map = raw

    mapping: Dict[int, List[int]] = {}
    for k, vs in raw_map.items():
        u = int(k)
        mapping[u] = [int(x) for x in vs]
    return mapping


def evaluate_hit_mrr(
    mapping: Dict[int, List[int]],
    aligned_pair: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
):
    """
    使用 Hit@k 和 MRR 对对齐结果进行评估。

    参数:
        mapping: dict {u_src: [v_tgt1, v_tgt2, ...]}，
                 候选列表需要按“相似度/得分降序排序”
        aligned_pair: np.ndarray, shape (M, 2)，每行是 (u_gt, v_gt)
        ks: 需要计算的 k 值列表，例如 (1,5,10)

    返回:
        hit_at_k: dict {k: value}，每个 k 对应一个 Hit@k（0~1）
        mrr: float，整体 MRR（0~1）
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
        for idx, v_pred in enumerate(cand_list):
            if v_pred == v:
                rank = idx + 1  # 从 1 开始计
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alignment mapping with Hit@k and MRR.")
    parser.add_argument("--npz_path", type=str, required=True,
                        help="Path to dataset .npz file containing aligned_pair.")
    parser.add_argument("--aligned_key", type=str, default="aligned_pair",
                        help="Key name for aligned pairs inside npz (default: aligned_pair).")
    parser.add_argument("--mapping_json", type=str, required=True,
                        help="Path to JSON file with mapping (e.g., reranked_mapping JSON).")
    parser.add_argument("--ks", type=str, default="1,5,10",
                        help="Comma-separated list of k values to compute Hit@k, e.g. '1,5,10'.")
    return parser.parse_args()


def main():
    args = parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    aligned_pair = load_aligned_pairs(args.npz_path, key=args.aligned_key)
    mapping = load_mapping(args.mapping_json)

    print("===== Evaluation settings =====")
    print(f"npz_path      : {args.npz_path}")
    print(f"aligned_key   : {args.aligned_key}")
    print(f"mapping_json  : {args.mapping_json}")
    print(f"Ks            : {ks}")
    print(f"#aligned pairs: {aligned_pair.shape[0]}")
    print(f"#src nodes in mapping: {len(mapping)}")
    print("================================")

    hit_at_k, mrr = evaluate_hit_mrr(mapping, aligned_pair, ks=ks)

    print("===== Final scores =====")
    for k in ks:
        print(f"Hit@{k}: {hit_at_k[k]:.4f}")
    print(f"MRR  : {mrr:.4f}")


if __name__ == "__main__":
    main()