import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import time
import argparse
import os
from tqdm import tqdm


def build_sparse_graph(edge_index, num_nodes):
    """
    根据边索引列表构建一个稀疏的、无向的邻接矩阵。
    edge_index: 形状为 (2, E)，表示 E 条边 (src, dst)
    """
    row = np.concatenate([edge_index[0], edge_index[1]])
    col = np.concatenate([edge_index[1], edge_index[0]])
    data = np.ones(row.shape[0], dtype=np.float32)
    return sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))


def load_graph_npz(path):
    """
    从 .npz 文件中读取多层网络及对齐信息。
    要求至少包含：
        - edge_index1, edge_index2
        - x1, x2
        - pos_pair  (训练集对齐对，用作锚点对)
    若存在 test_pair，仅作信息打印，不参与锚点构造。
    """
    print(f"Loading data from '{path}' ...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件 '{path}' 不存在。")

    data = np.load(path, allow_pickle=True)

    required_keys = ["edge_index1", "edge_index2", "x1", "x2", "pos_pairs"]
    for k in required_keys:
        if k not in data:
            raise KeyError(
                f".npz 文件中缺少必要键 '{k}'，当前已有键: {list(data.keys())}"
            )

    edge_index1 = data["edge_index1"]
    edge_index2 = data["edge_index2"]
    x1 = data["x1"]
    x2 = data["x2"]
    pos_pair = data["pos_pairs"]  # 训练集对齐对

    test_pair = data["test_pairs"] if "test_pairs" in data else None

    print(f"  读取到 edge_index1, edge_index2, x1, x2, pos_pairs")
    if test_pair is not None:
        print(f"  读取到 test_pairs (仅作信息展示，不用于锚点): {test_pair.shape[0]} 对")

    return edge_index1, edge_index2, x1, x2, pos_pair, test_pair


def compute_distances_to_anchors(sparse_graph, anchor_nodes, num_nodes, graph_name="G"):
    """
    计算：对每个 anchor 节点，图中所有节点到该 anchor 的最短路径距离。
    返回:
        dist_full: 形状 [num_unique_anchors, num_nodes]
        unique_anchors: 去重后的锚点列表 (与 dist_full 的行对应)
    """
    anchor_nodes = np.asarray(anchor_nodes, dtype=np.int64)
    unique_anchors = np.unique(anchor_nodes)

    print(f"{graph_name}: 共 {num_nodes} 个节点，"
          f"锚点节点（去重后）数量 = {len(unique_anchors)}")

    if unique_anchors.size == 0:
        print(f"  警告: {graph_name} 中没有任何锚点节点。")
        dist_full = np.full((0, num_nodes), np.inf, dtype=np.float32)
        return dist_full, unique_anchors

    print(f"  使用 SciPy shortest_path 计算从每个锚点到所有节点的最短路 ...")
    t0 = time.time()
    # dist_full[k, i] = 从 unique_anchors[k] 到节点 i 的距离
    dist_full = shortest_path(
        csgraph=sparse_graph,
        directed=False,
        indices=unique_anchors,
        unweighted=True  # 边权统一为 1
    )
    t1 = time.time()
    print(f"  shortest_path 完成，用时 {t1 - t0:.2f} 秒。")

    # 转成 float32 节省内存
    dist_full = dist_full.astype(np.float32)

    # 统计不可达节点（对所有锚点距离均为 inf）
    min_dist = np.min(dist_full, axis=0)
    num_inf = np.sum(~np.isfinite(min_dist))
    if num_inf > 0:
        print(f"  ⚠ {graph_name} 中有 {num_inf} 个节点无法到达任何锚点节点。")

    return dist_full, unique_anchors


def build_super_adjacency_H(
    dist1_full,
    dist2_full,
    anchors1_unique,
    anchors2_unique,
    train_pairs,
    num_nodes1,
    num_nodes2
):
    """
    构造多层超邻接矩阵 H，行对应 G1 节点 i，列对应 G2 节点 j。

    定义:
        对任意 (i, j)，
        H[i,j] = min_k ( d1(i, u_k) + 1 + d2(v_k, j) )
        其中 (u_k, v_k) 为训练集锚点对 train_pairs[k]

    输入:
        dist1_full: 形状 [num_unique_anchors1, num_nodes1]
                    dist1_full[a_idx, i] = d1(anchors1_unique[a_idx], i)
        dist2_full: 形状 [num_unique_anchors2, num_nodes2]
        anchors1_unique: 去重后的锚点节点列表 (G1 的锚点)
        anchors2_unique: 去重后的锚点节点列表 (G2 的锚点)
        train_pairs: 形状 [A, 2]，每一行是 (u_k, v_k)
        num_nodes1, num_nodes2: 两图节点数
    """
    print("开始构造超邻接矩阵 H ...")
    H = np.full((num_nodes1, num_nodes2), np.inf, dtype=np.float32)

    # 构建 node_id -> anchor 行索引 的映射
    anchor_to_row_1 = {int(node): idx for idx, node in enumerate(anchors1_unique)}
    anchor_to_row_2 = {int(node): idx for idx, node in enumerate(anchors2_unique)}

    num_pairs = train_pairs.shape[0]
    print(f"训练集锚点对数量: {num_pairs}")

    for k in tqdm(range(num_pairs), desc="  累加每个锚点对的贡献", ncols=80):
        u_k = int(train_pairs[k, 0])  # G1 中的锚点
        v_k = int(train_pairs[k, 1])  # G2 中的锚点

        if u_k not in anchor_to_row_1 or v_k not in anchor_to_row_2:
            # 理论上不会发生（因为 anchors*_unique 就来自 train_pairs 的集合）
            # 但为了鲁棒性，还是加个判断
            continue

        row_idx_1 = anchor_to_row_1[u_k]
        row_idx_2 = anchor_to_row_2[v_k]

        # d1[i] = d1(i, u_k) = dist1_full[row_idx_1, i]
        d1_to_u = dist1_full[row_idx_1]          # 形状 [num_nodes1]
        # d2[j] = d2(v_k, j) = dist2_full[row_idx_2, j]
        d2_from_v = dist2_full[row_idx_2]        # 形状 [num_nodes2]

        # 候选矩阵: C_ij = d1(i, u_k) + 1 + d2(v_k, j)
        # 利用广播： [num_nodes1, 1] + [1] + [1, num_nodes2]
        C = d1_to_u[:, None] + 1.0 + d2_from_v[None, :]

        # 按元素取更小值
        # 注意：inf 会被任何有限值覆盖
        H = np.minimum(H, C)

    # 完成后检查锚点对位置是否为 1（如果可达）
    print("H 矩阵基本构造完成，进行锚点对位置检查 ...")
    mismatched = 0
    for k in range(num_pairs):
        u_k = int(train_pairs[k, 0])
        v_k = int(train_pairs[k, 1])
        val = H[u_k, v_k]
        # 如果可达，理论上应为 1
        if np.isfinite(val) and not np.isclose(val, 1.0):
            mismatched += 1
    if mismatched > 0:
        print(f"⚠ 有 {mismatched} 个锚点对在 H 中的值不是 1（可能由于图不连通或距离异常），请注意检查。")
    else:
        print("  所有可达的锚点对 (u_k, v_k) 在 H 中的取值均为 1 ✅")

    return H


def main(input_file, output_file_path):
    # 1. 读取数据
    edge_index1, edge_index2, x1, x2, pos_pair, test_pair = load_graph_npz(input_file)

    num_nodes1 = x1.shape[0]
    num_nodes2 = x2.shape[0]
    num_edges1 = edge_index1.shape[1]
    num_edges2 = edge_index2.shape[1]

    print(f"G1: {num_nodes1} 个节点, {num_edges1} 条边")
    print(f"G2: {num_nodes2} 个节点, {num_edges2} 条边")
    print(f"训练集锚点对 (pos_pair) 数量: {pos_pair.shape[0]}")
    if test_pair is not None:
        print(f"测试集锚点对 (test_pair) 数量: {test_pair.shape[0]}")

    # 2. 构建稀疏图
    print("构建稀疏图 ...")
    graph1_sparse = build_sparse_graph(edge_index1, num_nodes1)
    graph2_sparse = build_sparse_graph(edge_index2, num_nodes2)

    # 3. 计算从锚点到所有节点的最短路（分别在 G1 与 G2 上）
    print("计算 G1 中锚点到所有节点的最短路 ...")
    dist1_full, anchors1_unique = compute_distances_to_anchors(
        graph1_sparse, pos_pair[:, 0], num_nodes1, graph_name="G1"
    )

    print("计算 G2 中锚点到所有节点的最短路 ...")
    dist2_full, anchors2_unique = compute_distances_to_anchors(
        graph2_sparse, pos_pair[:, 1], num_nodes2, graph_name="G2"
    )

    # 4. 利用训练集锚点对构造超邻接矩阵 H
    start_time = time.time()
    H = build_super_adjacency_H(
        dist1_full=dist1_full,
        dist2_full=dist2_full,
        anchors1_unique=anchors1_unique,
        anchors2_unique=anchors2_unique,
        train_pairs=pos_pair,
        num_nodes1=num_nodes1,
        num_nodes2=num_nodes2
    )
    print(f"超邻接矩阵 H 构造完成，总耗时 {time.time() - start_time:.2f} 秒。")

    # 5. 保存结果
    try:
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.save(output_file_path, H)
        print("-" * 40)
        print(f"✅ 成功! 超邻接矩阵 H 已保存到: '{output_file_path}'")
        print(f"H 的形状: {H.shape}")
        finite_vals = H[np.isfinite(H)]
        if finite_vals.size > 0:
            print(f"H 中有限值的最小值: {finite_vals.min()}")
            print(f"H 中有限值的最大值: {finite_vals.max()}")
        else:
            print("⚠ H 中没有有限值，请检查图连通性与锚点设置。")
    except Exception as e:
        print(f"错误: 保存文件到 '{output_file_path}' 时失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "从 .npz 多层网络数据生成超邻接矩阵 H。\n"
            "要求 .npz 至少包含: edge_index1, edge_index2, x1, x2, pos_pair。\n"
            "H[i,j] = min_k ( d1(i,u_k) + 1 + d2(v_k,j) )，只使用训练集锚点对 pos_pair。"
        )
    )

    parser.add_argument("--input",type=str,required=True,help="指向输入的 .npz 数据集文件 (例如: douban.npz)")

    parser.add_argument("--output_dir",type=str,default=".",help="指定输出文件的保存目录 (默认: 当前目录)")

    args = parser.parse_args()

    base_name = os.path.basename(args.input)
    dataset_name = os.path.splitext(base_name)[0]
    output_filename = f"{dataset_name}_H.npy"
    output_path = os.path.join(args.output_dir, output_filename)

    main(args.input, output_path)