import os
import argparse
import pickle
import json
import torch

# 导入 dataset.py 中定义的类和函数
from dataset import (
    MultiLayerGraphDataset,
    merge_graphs,
    process_node_features_for_cgtp,
    process_alignment_pairs,
    TAGDatasetForLM
)

# <--- 此处开始 --->
# 从 dataset.py 剪切过来的 __main__ 逻辑

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="多层网络数据的 .npz 文件路径")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/prepared_data", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = MultiLayerGraphDataset(args.data_path)

    merged_edge_index, merged_x, mapping_g2_to_merged, n1, n2, n_total = merge_graphs(
        dataset.edge_index1, dataset.edge_index2, dataset.x1, dataset.x2
    )
    gnid2text, merged_features = process_node_features_for_cgtp(
        dataset.x1, dataset.x2, n1, n2, mapping_g2_to_merged
    )
    train_pairs_merged, test_pairs_merged = process_alignment_pairs(dataset.pos_pairs, dataset.test_pairs, n1)
    tag_dataset = TAGDatasetForLM(merged_edge_index, gnid2text, merged_features)

    # === 方案2: 清理 features 再保存对象 ===
    output_dir = args.output_dir

    # 1️⃣ 保存基础数据
    torch.save(merged_edge_index, os.path.join(output_dir, "merged_edge_index.pt"))
    torch.save(train_pairs_merged, os.path.join(output_dir, "train_pairs_merged.pt"))
    torch.save(test_pairs_merged, os.path.join(output_dir, "test_pairs_merged.pt"))

    # 2️⃣ 保存大矩阵独立文件
    if merged_features is not None and merged_features.numel() > 0:
        torch.save(merged_features, os.path.join(output_dir, "merged_features.pt"))
        print(f"✅ merged_features 已单独保存 ({merged_features.shape})")

    # 3️⃣ 清空特征再保存 dataset 对象
    tag_dataset.features = None
    with open(os.path.join(output_dir, "dataset_for_lm.pkl"), "wb") as f:
        pickle.dump(tag_dataset, f)
    print(f"✅ dataset_for_lm.pkl 保存完成（不含特征矩阵）")

    # 4️⃣ 其他可选保存
    if gnid2text is not None:
        with open(os.path.join(output_dir, "gnid2text.json"), "w", encoding="utf-8") as f:
            json.dump(gnid2text, f, ensure_ascii=False, indent=2)

    print(f"\n🎯 所有数据已保存到: {os.path.abspath(args.output_dir)}")
    print("数据准备完毕，可直接进入 CGTP 预训练阶段。")
