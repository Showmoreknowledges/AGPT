import argparse

from gtc_align import gtc_align
from gtc_align.dataset_generic import load_alignment_npz
from gtc_align.metrics import evaluate_hit_mrr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GTCAlign on different datasets by name."
    )
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="数据集名称，如 ACM-DBLP, douban, amazon 等。")
    parser.add_argument("--root_dir", type=str, default="data",
                        help="存放各数据集 npz 的目录，默认 data")
    parser.add_argument("--no_features", action="store_true",
                        help="若指定，则忽略 x1/x2 特征，使用常数特征作为 GCN 输入。")
    parser.add_argument("--L1", type=float, default=1.05,
                        help="中心性比值阈值 L1。")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="GCN 隐层维度。")
    parser.add_argument("--topk", type=int, default=10,
                        help="每个源节点保留的候选目标节点数。")
    parser.add_argument("--seed", type=int, default=42,
                        help="GCN 权重初始化随机种子。")
    return parser.parse_args()


def main():
    args = parse_args()

    adj_s, adj_t, feats_s, feats_t, aligned_pair = load_alignment_npz(
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        use_features=not args.no_features,
    )

    mapping = gtc_align(
        adj_s=adj_s,
        adj_t=adj_t,
        feats_s=feats_s,
        feats_t=feats_t,
        L1=args.L1,
        hidden_dim=args.hidden_dim,
        topk=args.topk,
        seed=args.seed,
    )

    print(f"===== 数据集 {args.dataset_name}：前 5 个源节点的候选集合 =====")
    for u in range(min(5, adj_s.shape[0])):
        print(f"源节点 {u} -> 候选目标节点: {mapping.get(u, [])}")

    if aligned_pair is not None:
    hit_at_k, mrr = evaluate_hit_mrr(
        mapping=mapping,
        aligned_pair=aligned_pair,
        ks=(1, 5, 10),
    )
    print("===== 评估结果 =====")
    for k, v in hit_at_k.items():
        print(f"Hit@{k}: {v:.4f}")
    print(f"MRR: {mrr:.4f}")

    # # 用真实对齐对做一个简单 Hit@k 评估
    # if aligned_pair is not None:
    #     hit_at_k = 0
    #     total = aligned_pair.shape[0]
    #     for u_gt, v_gt in aligned_pair:
    #         cands = mapping.get(int(u_gt), [])
    #         if int(v_gt) in cands:
    #             hit_at_k += 1
    #     print(
    #         f"在 top-{args.topk} 上的简单 Hit@{args.topk}: "
    #         f"{hit_at_k}/{total} = {hit_at_k / total:.4f}"
    #     )


if __name__ == "__main__":
    main()

