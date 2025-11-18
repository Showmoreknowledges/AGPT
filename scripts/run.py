"""
Full alignment pipeline controller:

1) 使用 GTCAlign 在结构空间做粗筛，生成每个源节点的候选集合 mapping_gtc
2) 用真实对齐对 aligned_pair 评估一次 (Hit@k, MRR) —— GTC-only
3) 调用 rerank_with_llm.py，用节点文本 + LLM 对候选进行精排
4) 读回精排结果，再评一次 (Hit@k, MRR) —— GTC + LLM

示例用法：

    python scripts/run_full_alignment_pipeline.py \
      --dataset_name douban \
      --root_dir data \
      --texts_path data/douban_node_texts.json \
      --src_layer layer1 \
      --tgt_layer layer2 \
      --provider openai \
      --model gpt-4.1-mini \
      --gtc_topk 50 \
      --ks 1,5,10 \
      --output_dir results/douban_exp1

注意：
  - prompt 模板在 rerank_with_llm.py 里的 build_prompt() 中修改
  - 这里假设通用 loader 在 gtc_align.dataset_generic.load_alignment_npz
  - 这里假设 gtc_align.__init__ 中有 gtc_align(adj_s, adj_t, feats_s, feats_t, ...)
"""

import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Sequence, Tuple

import numpy as np
import subprocess

from gtc_align.dataset_generic import load_alignment_npz
from gtc_align import gtc_align


# ========== 0. 评估函数（和 eval_alignment.py 一致） ==========

def evaluate_hit_mrr(
    mapping: Dict[int, List[int]],
    aligned_pair: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> Tuple[Dict[int, float], float]:
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


# ========== 1. GTC 粗筛部分：直接调用 gtc_align 模块 ==========

def run_gtc_alignment(
    dataset_name: str,
    root_dir: str,
    gtc_topk: int,
    L1: float = 1.05,
    hidden_dim: int = 128,
    use_features: bool = True,
):
    """
    使用通用加载器 + gtc_align 进行粗对齐，返回：
      - mapping_gtc: {u: [v1, v2, ...]} (按相似度降序)
      - aligned_pair: np.ndarray, shape (M, 2)
    """

    adj_s, adj_t, feats_s, feats_t, aligned_pair = load_alignment_npz(
        dataset_name=dataset_name,
        root_dir=root_dir,
        use_features=use_features,
    )

    mapping = gtc_align(
        adj_s=adj_s,
        adj_t=adj_t,
        feats_s=feats_s,
        feats_t=feats_t,
        L1=L1,
        hidden_dim=hidden_dim,
        topk=gtc_topk,
        seed=42,
    )

    # mapping 已经是 {u: [v1, v2, ...]} 形式
    # 为保险起见，把键和值都转成 int
    mapping_int: Dict[int, List[int]] = {}
    for u, vs in mapping.items():
        mapping_int[int(u)] = [int(v) for v in vs]

    return mapping_int, aligned_pair


# ========== 2. 调用 rerank_with_llm.py 精排 ==========

def call_rerank_with_llm(
    candidates_path: str,
    texts_path: str,
    src_layer: str,
    tgt_layer: str,
    provider: str,
    model: str,
    output_json: str,
    output_csv: str,
    max_candidates_per_src: int,
    rerank_script: str = "scripts/rerank_with_llm.py",
    ):
    """
    通过 subprocess 调用你之前写好的 rerank_with_llm.py。
    prompt 模板在 rerank_with_llm.py 的 build_prompt() 中修改。
    """
    cmd = [
        "python",
        rerank_script,
        "--candidates_path", candidates_path,
        "--texts_path", texts_path,
        "--src_layer", src_layer,
        "--tgt_layer", tgt_layer,
        "--provider", provider,
        "--model", model,
        "--output_json", output_json,
    ]
    if output_csv:
        cmd += ["--output_csv", output_csv]
    if max_candidates_per_src > 0:
        cmd += ["--max_candidates_per_src", str(max_candidates_per_src)]

    print("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_mapping_from_json(path: str) -> Dict[int, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "mapping" in raw:
        raw_map = raw["mapping"]
    else:
        raw_map = raw
    mapping: Dict[int, List[int]] = {}
    for k, vs in raw_map.items():
        mapping[int(k)] = [int(x) for x in vs]
    return mapping


# ========== 3. CLI & 主流程 ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full pipeline: GTC → LLM rerank → eval.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="数据集名称（对应 {root_dir}/{dataset_name}.npz）")
    parser.add_argument("--root_dir", type=str, default="data",
                        help="存放 npz 的目录，默认 data")
    parser.add_argument("--texts_path", type=str, required=True,
                        help="节点文本 JSON 路径（generate_text 脚本的输出 text_path）")
    parser.add_argument("--src_layer", type=str, required=True,
                        help="texts_json 中源图对应的 layer 名")
    parser.add_argument("--tgt_layer", type=str, required=True,
                        help="texts_json 中目标图对应的 layer 名")
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "openrouter", "deepseek", "hf_local"],
                        help="LLM 提供方")
    parser.add_argument("--model", type=str, required=True,
                        help="LLM 模型名 / id（与 provider 对应）")
    parser.add_argument("--gtc_topk", type=int, default=50,
                        help="GTC 粗筛阶段每个源节点的候选数 (top-k)")
    parser.add_argument("--L1", type=float, default=1.05,
                        help="GTC 中心性比值阈值 L1")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="GCN 隐层维度")
    parser.add_argument("--use_features", action="store_true",
                        help="是否使用 x1/x2 作为 GCN 输入特征（默认关闭只用结构）")
    parser.add_argument("--ks", type=str, default="1,5,10",
                        help="Hit@k 里的 k 值列表，如 '1,5,10'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="结果输出目录（候选 JSON、精排 JSON 等都会放在这里）")
    parser.add_argument("--rerank_script", type=str, default="scripts/rerank_with_llm.py",
                        help="rerank_with_llm.py 的路径")
    parser.add_argument("--max_candidates_per_src", type=int, default=0,
                        help="精排阶段每个源节点最多精排多少个候选（0 表示全部）")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    npz_path = osp.join(args.root_dir, f"{args.dataset_name}.npz")

    print("===== Step 1: GTCAlign 粗筛 =====")
    mapping_gtc, aligned_pair = run_gtc_alignment(
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        gtc_topk=args.gtc_topk,
        L1=args.L1,
        hidden_dim=args.hidden_dim,
        use_features=args.use_features,
    )

    print(f"  Loaded {aligned_pair.shape[0]} aligned pairs for evaluation.")
    print(f"  Got {len(mapping_gtc)} source nodes from GTC mapping.")

    hit_gtc, mrr_gtc = evaluate_hit_mrr(mapping_gtc, aligned_pair, ks=ks)
    print("  --- GTC-only scores ---")
    for k in ks:
        print(f"  Hit@{k}: {hit_gtc[k]:.4f}")
    print(f"  MRR   : {mrr_gtc:.4f}")

    # 保存 GTC 候选 JSON，给 rerank 用 & 记录下来
    gtc_candidates_path = osp.join(args.output_dir, f"{args.dataset_name}_gtc_candidates.json")
    with open(gtc_candidates_path, "w", encoding="utf-8") as f:
        json.dump({str(u): vs for u, vs in mapping_gtc.items()}, f, ensure_ascii=False, indent=2)
    print(f"  Saved GTC candidates to {gtc_candidates_path}")

    print("\n===== Step 2: 调用 rerank_with_llm.py 做精排 =====")
    rerank_json = osp.join(args.output_dir, f"{args.dataset_name}_reranked_{args.model}.json")
    rerank_csv = osp.join(args.output_dir, f"{args.dataset_name}_reranked_{args.model}.csv")

    call_rerank_with_llm(
        candidates_path=gtc_candidates_path,
        texts_path=args.texts_path,
        src_layer=args.src_layer,
        tgt_layer=args.tgt_layer,
        provider=args.provider,
        model=args.model,
        output_json=rerank_json,
        output_csv=rerank_csv,
        max_candidates_per_src=args.max_candidates_per_src,
        rerank_script=args.rerank_script,
    )

    print("\n===== Step 3: 读取精排结果再评一次 =====")
    mapping_reranked = load_mapping_from_json(rerank_json)
    print(f"  Got {len(mapping_reranked)} source nodes from reranked mapping.")

    hit_llm, mrr_llm = evaluate_hit_mrr(mapping_reranked, aligned_pair, ks=ks)
    print("  --- GTC + LLM rerank scores ---")
    for k in ks:
        print(f"  Hit@{k}: {hit_llm[k]:.4f}")
    print(f"  MRR   : {mrr_llm:.4f}")

    print("\n===== Summary =====")
    print("  Ks:", ks)
    print("  GTC-only:")
    for k in ks:
        print(f"    Hit@{k}: {hit_gtc[k]:.4f}")
    print(f"    MRR   : {mrr_gtc:.4f}")
    print("  GTC + LLM rerank:")
    for k in ks:
        print(f"    Hit@{k}: {hit_llm[k]:.4f}")
    print(f"    MRR   : {mrr_llm:.4f}")


if __name__ == "__main__":
    main()