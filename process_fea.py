import torch
import numpy as np

from dataset import MultiLayerGraphDataset
from merge import merge_graphs

def process_node_features_for_cgtp(x1, x2, num_nodes_1, num_nodes_2, mapping_g2_to_merged):
    """
    将节点特征/文本转换为 LinkGPT-CGTP 格式：
    - 若 x1/x2 为文本或字符串，则生成 gnid2text 字典
    - 若 x1/x2 为特征向量，则生成合并特征矩阵 X
    
    参数：
        x1, x2: 节点特征或文本（list / np.ndarray / torch.Tensor / dict / None）
        num_nodes_1, num_nodes_2: 各图节点数量
        mapping_g2_to_merged: G2 原始 ID → 合并图 ID
    
    返回：
        gnid2text (dict or None): 节点ID → 文本描述
        merged_features (torch.Tensor or None): 节点特征矩阵
    """
    gnid2text = None
    merged_features = None

    # 判断是文本型还是特征型
    def is_textual(x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)):
            return all(isinstance(i, str) for i in x)
        if isinstance(x, dict):
            return all(isinstance(v, str) for v in x.values())
        return False

    # ======================================================
    # ① 文本情况：生成 gnid2text
    # ======================================================
    if is_textual(x1) or is_textual(x2):
        gnid2text = {}

        # 处理 G1 的节点文本
        if isinstance(x1, dict):
            for nid, text in x1.items():
                gnid2text[int(nid)] = text
        elif isinstance(x1, (list, tuple)):
            for nid, text in enumerate(x1):
                gnid2text[nid] = text

        # 处理 G2 的节点文本
        if isinstance(x2, dict):
            for orig_id, text in x2.items():
                merged_id = mapping_g2_to_merged[int(orig_id)]
                gnid2text[merged_id] = text
        elif isinstance(x2, (list, tuple)):
            for orig_id, text in enumerate(x2):
                merged_id = mapping_g2_to_merged[orig_id]
                gnid2text[merged_id] = text

        print(f"✅ 已生成 gnid2text 字典，共 {len(gnid2text)} 条文本节点")

    # ======================================================
    # ② 特征矩阵情况：生成 merged_features
    # ======================================================
    else:
        def to_tensor(x, n):
            if x is None:
                return torch.zeros((n, 1))
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            if isinstance(x, torch.Tensor):
                return x.float()
            raise TypeError(f"Unsupported feature type: {type(x)}")

        x1_tensor = to_tensor(x1, num_nodes_1)
        x2_tensor = to_tensor(x2, num_nodes_2)
        merged_features = torch.cat([x1_tensor, x2_tensor], dim=0)

        print(f"✅ 已生成 merged_features 特征矩阵，形状: {merged_features.shape}")

    return gnid2text, merged_features



if __name__ == "__main__":
     

    dataset = MultiLayerGraphDataset("/mnt/data/phone-email_0.2.npz")

    merged_edge_index, merged_x_placeholder, mapping_g2_to_merged, \
    n1, n2, n_total = merge_graphs(
        dataset.edge_index1,
        dataset.edge_index2,
        dataset.x1,
        dataset.x2
    )

    gnid2text, merged_features = process_node_features_for_cgtp(
        dataset.x1,
        dataset.x2,
        n1,
        n2,
        mapping_g2_to_merged
    )

    # 打印结果
    if gnid2text is not None:
        print(f"示例文本节点: {list(gnid2text.items())[:5]}")
    if merged_features is not None:
        print(f"示例特征: {merged_features[:5]}")
