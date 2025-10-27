import torch
from dataset import MultiLayerGraphDataset

def merge_graphs(edge_index1, edge_index2, x1=None, x2=None):
    """
    将 G1 和 G2 合并为单一图 (G_merged)，用于 LinkGPT 框架。
    
    参数：
        edge_index1 (Tensor): 图1的边索引 [2, E1]
        edge_index2 (Tensor): 图2的边索引 [2, E2]
        x1 (Tensor or None): 图1的节点特征（可为空）
        x2 (Tensor or None): 图2的节点特征（可为空）
    
    返回：
        merged_edge_index (Tensor): 合并后的边 [2, E_total]
        merged_x (Tensor): 合并后的节点特征
        mapping_g2_to_merged (dict): G2旧ID → 新ID 的映射
        num_nodes_1, num_nodes_2, num_nodes_total (int): 节点统计信息
    """
    # ----  计算节点数 ----
    num_nodes_1 = int(edge_index1.max()) + 1
    num_nodes_2 = int(edge_index2.max()) + 1
    print(f"G1 节点数: {num_nodes_1}, G2 节点数: {num_nodes_2}")

    # ----  重新编号 G2 的节点 ----
    offset = num_nodes_1
    mapping_g2_to_merged = {i: i + offset for i in range(num_nodes_2)}

    edge_index2_shifted = edge_index2 + offset

    # ----  合并边 ----
    merged_edge_index = torch.cat([edge_index1, edge_index2_shifted], dim=1)

    # ----  合并节点特征 ----
    if x1 is None:
        x1 = torch.zeros((num_nodes_1, 1))
    if x2 is None:
        x2 = torch.zeros((num_nodes_2, 1))
    merged_x = torch.cat([x1, x2], dim=0)

    num_nodes_total = num_nodes_1 + num_nodes_2

    print(f"✅ 合并完成: 共 {num_nodes_total} 个节点, "
          f"{merged_edge_index.shape[1]} 条边")

    return merged_edge_index, merged_x, mapping_g2_to_merged, \
           num_nodes_1, num_nodes_2, num_nodes_total



if __name__ == "__main__":
    
    dataset = MultiLayerGraphDataset("/mnt/data/phone-email_0.2.npz")

    merged_edge_index, merged_x, mapping_g2_to_merged, \
    n1, n2, n_total = merge_graphs(
        dataset.edge_index1,
        dataset.edge_index2,
        dataset.x1,
        dataset.x2
    )

    print(f"\n总节点数: {n_total}")
    print(f"示例映射: 前5个 G2 节点映射 → {[list(mapping_g2_to_merged.items())[:5]]}")
