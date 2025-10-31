import os
import argparse
import pickle
import json
import torch
from typing import Dict, List, Tuple


from dataset import TAGDatasetForLM
# 导入 dataset.py 中定义的类和函数
from Mul_dataset import (
    MultiLayerGraphDataset,
    merge_graphs,
    process_node_features_for_cgtp,
    process_alignment_pairs,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="多层网络数据")
    parser.add_argument("--data_path", type=str, required=True, help="多层网络数据的 .npz 文件路径")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/prepared_data", help="输出目录")
    args = parser.parse_args()

    os.makedirs(f"{args.output_dir}/{args.data}", exist_ok=True)
    dataset = MultiLayerGraphDataset(args.data_path)

    merged_edge_index, _, layer_offsets, num_nodes_per_layer, total_num_nodes = merge_graphs(
        dataset.edge_indices, dataset.features
    )
    gnid2text, merged_features = process_node_features_for_cgtp(
        dataset.features, num_nodes_per_layer, layer_offsets
    )
    
    def _tensor_pairs_to_list(pairs: torch.Tensor) -> List[Tuple[int, int]]:
        if pairs.numel() == 0:
            return []
        return [(int(src), int(tgt)) for src, tgt in pairs.tolist()]

    combined_pairs: List[Tuple[int, int]] = []
    for tensor in (train_pairs_merged, test_pairs_merged):
        combined_pairs.extend(_tensor_pairs_to_list(tensor))

    node_records: Dict[int, Dict[str, object]] = {}
    for node_id in range(total_num_nodes):
        node_entry: Dict[str, object] = {"node_id": node_id}
        if gnid2text is not None:
            node_entry["text"] = gnid2text.get(node_id, "")
        else:
            node_entry["text"] = ""
        node_records[node_id] = node_entry

    tag_dataset = TAGDatasetForLM(node_records, combined_pairs)
    tag_dataset.features = merged_features
    tag_dataset.gnid2text = gnid2text if gnid2text is not None else None
    tag_dataset.layer_offsets = layer_offsets
    tag_dataset.num_nodes_per_layer = num_nodes_per_layer
    tag_dataset.total_num_nodes = total_num_nodes

    def _build_edge_split_section(pairs: torch.Tensor) -> Dict[str, object]:
        pairs_cpu = pairs.detach().cpu()
        src_list: List[int] = pairs_cpu[:, 0].tolist() if pairs_cpu.numel() > 0 else []
        tgt_list: List[int] = pairs_cpu[:, 1].tolist() if pairs_cpu.numel() > 0 else []
        section: Dict[str, object] = {
            "source_node": src_list,
            "target_node": tgt_list,
        }
        section["edge"] = pairs_cpu.numpy()
        return section

    print("✅ 正在将合并后的对齐链接注入 'dataset_for_lm.edge_split'...")
    tag_dataset.edge_split = {
        "train": _build_edge_split_section(train_pairs_merged),
        "valid": _build_edge_split_section(test_pairs_merged),
        "test": _build_edge_split_section(test_pairs_merged),
    }
    tag_dataset.generate_gnid2neighbors_train()

    # === 清理 features 再保存对象 ===
    output_dir = f"{args.output_dir}/{args.data}"

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
    tag_dataset.gnid2text = None
    with open(os.path.join(output_dir, "dataset_for_lm.pkl"), "wb") as f:
        pickle.dump(tag_dataset, f)
    print(f"✅ dataset_for_lm.pkl 保存完成（已注入对齐链接）") 

    # 4️⃣ 其他可选保存
    if gnid2text is not None:
        with open(os.path.join(output_dir, "gnid2text.json"), "w", encoding="utf-8") as f:
            json.dump(gnid2text, f, ensure_ascii=False, indent=2)

    print(f"\n🎯 所有数据已保存到: {os.path.abspath(args.output_dir)}")
    print("数据准备完毕，可直接进入 CGTP 预训练阶段。")
