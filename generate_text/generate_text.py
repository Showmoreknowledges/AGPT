"""
Generate textual descriptions for multi-layer (two-layer) graphs by combining
structural features, anchor alignments, and cross-layer hop information.

Optimizations vs original version:
- Fix feature reshape bug (1D -> (num_nodes, 1))
- Avoid duplicate edge file loads
- More robust handling in select_cross_layer_targets
- Add "mode" option in text_options: "compact" (default) or "verbose"
- Reduce repeated computations in build_node_text
"""

import argparse
import json
import math
import os
import os.path as osp
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml

INT_METRICS = {"degree", "triangle", "core_number"}

METRIC_LABELS = {
    "degree": "degree",
    "clustering": "clustering",
    "betweenness": "betweenness",
    "closeness": "closeness",
    "pagerank": "pagerank",
    "core_number": "core number",
    "triangle": "triangle count",
}

DEFAULT_OPTIONS = {
    "structural_features": ["degree", "clustering", "betweenness"],
    "max_cross_layer_targets": 2,
    "include_paths": True,
    "show_percentile": True,
    "numeric_precision": 3,
    # 新增：文本模式，"compact" 更适合做 LLM 输入
    "mode": "compact",  # or "verbose"
}


@dataclass
class LayerRuntime:
    name: str
    prefix: str
    num_nodes: int
    graph: nx.Graph
    structural: Dict[str, Dict[int, float]]
    percentiles: Dict[str, Dict[int, float]]
    anchor_distance: Dict[int, float] = field(default_factory=dict)
    closest_anchor: Dict[int, Optional[int]] = field(default_factory=dict)
    anchor_prev: Dict[int, Optional[int]] = field(default_factory=dict)
    _anchor_route_cache: Dict[int, Dict[int, Optional[int]]] = field(default_factory=dict)

    def label(self, node_id: int) -> str:
        prefix = self.prefix or self.name
        return f"{prefix}{node_id}"

    def set_anchor_routing(
        self,
        dist: Dict[int, float],
        closest: Dict[int, Optional[int]],
        prev: Dict[int, Optional[int]],
    ):
        self.anchor_distance = dist
        self.closest_anchor = closest
        self.anchor_prev = prev

    def path_to_anchor(self, node_id: int) -> Optional[List[int]]:
        """Path from node to its closest anchor (within this layer)."""
        if not self.closest_anchor or self.closest_anchor.get(node_id) is None:
            return None
        if node_id not in self.anchor_prev:
            return None
        path = [node_id]
        visited = set()
        while True:
            prev = self.anchor_prev.get(path[-1])
            if prev is None:
                break
            if prev in visited:
                # guard against unexpected cycles
                return None
            visited.add(prev)
            path.append(prev)
        return path

    def path_from_anchor(self, anchor_id: int, target_id: int) -> Optional[List[int]]:
        """Path within this layer, from given anchor to target."""
        parents = self._anchor_route_cache.get(anchor_id)
        if parents is None:
            parents = bfs_parent_from_source(self.graph, anchor_id)
            self._anchor_route_cache[anchor_id] = parents
        if target_id not in parents:
            return None
        return reconstruct_forward_path(anchor_id, target_id, parents)




def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: str, target: Optional[str]) -> Optional[str]:
    if target is None:
        return None
    if osp.isabs(target):
        return target
    return osp.normpath(osp.join(base, target))


def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    """
    支持 .npy / .npz / .pt / .pth / .csv / .tsv / .json 的统一加载函数
    - .npz: 如果 key=None 且包含多个 array -> 报错
            如果 key=None 且只有一个 array -> 自动取唯一的那个
            如果 key 提供 -> data[key]
    """
    if path is None:
        raise ValueError("Path must be provided to load array.")

    ext = osp.splitext(path)[1].lower()

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)

        # 如果用户给了 key
        if key is not None:
            if key not in data.files:
                raise KeyError(
                    f"Key '{key}' not found in {path}. Available keys: {data.files}"
                )
            arr = data[key]
        else:
            # 用户没给 key：如果有多个数组 -> 报错
            if len(data.files) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays {data.files}, key must be specified."
                )
            # 只有一个 -> 自动取
            arr = data[data.files[0]]

        return np.asarray(arr)


    elif ext == ".npy":
        return np.load(path, allow_pickle=True)

    elif ext in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")

        if isinstance(loaded, dict):
            if key is not None:
                arr = loaded[key]
            else:
                if len(loaded) != 1:
                    raise ValueError(
                        f"{path} contains multiple tensors {list(loaded.keys())}, key must be specified."
                    )
                arr = next(iter(loaded.values()))
        else:
            arr = loaded

        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()

        return np.asarray(arr)


    elif ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        arr = pd.read_csv(path, sep=sep).values
        return np.asarray(arr)

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if key is not None:
            arr = data[key]
        else:
            arr = data
        return np.asarray(arr)

    else:
        raise ValueError(f"Unsupported file extension: {path}")



def normalize_edge_index(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.ndim != 2:
        raise ValueError("Edge index must be a 2D array.")
    if edge_index.shape[0] == 2:
        data = edge_index
    elif edge_index.shape[1] == 2:
        data = edge_index.T
    else:
        raise ValueError("Edge index should have shape (2, E) or (E, 2).")
    return data.astype(int)


def build_graph(edge_index: np.ndarray, num_nodes: int, directed: bool = False) -> nx.Graph:
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    graph.add_edges_from(zip(sources, targets))
    # 如果你的 edge_index 已经是无向边两方向都给了，可以关掉下面这行
    if not directed:
        graph.add_edges_from(zip(targets, sources))
    return graph


def compute_structural_metrics(
    graph: nx.Graph, features: Sequence[str]
) -> Dict[str, Dict[int, float]]:
    metrics = {}
    if "degree" in features:
        metrics["degree"] = {node: float(val) for node, val in graph.degree()}
    if "clustering" in features:
        metrics["clustering"] = nx.clustering(graph)
    if "betweenness" in features:
        metrics["betweenness"] = nx.betweenness_centrality(graph, normalized=True)
    if "closeness" in features:
        metrics["closeness"] = nx.closeness_centrality(graph)
    if "pagerank" in features:
        metrics["pagerank"] = nx.pagerank(graph, max_iter=200)
    if "core_number" in features:
        metrics["core_number"] = nx.core_number(graph)
    if "triangle" in features:
        metrics["triangle"] = nx.triangles(graph)
    return metrics


def compute_percentiles(metric_map: Dict[int, float]) -> Dict[int, float]:
    if not metric_map:
        return {}
    items = sorted(metric_map.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 100.0}
    percentiles = {}
    for idx, (node, _) in enumerate(items):
        percentiles[node] = 100.0 * idx / (n - 1)
    return percentiles


def multi_source_bfs(graph: nx.Graph, anchors: Sequence[int]):
    
    dist = {node: math.inf for node in graph.nodes}
    closest_anchor = {node: None for node in graph.nodes}
    prev = {node: None for node in graph.nodes}
    queue = deque()
    for anchor in anchors:
        if anchor not in graph:
            continue
        dist[anchor] = 0.0
        closest_anchor[anchor] = anchor
        prev[anchor] = None
        queue.append(anchor)
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            cand = dist[node] + 1.0
            if cand < dist[neighbor]:
                dist[neighbor] = cand
                closest_anchor[neighbor] = closest_anchor[node]
                prev[neighbor] = node
                queue.append(neighbor)
    return dist, closest_anchor, prev


def bfs_parent_from_source(graph: nx.Graph, source: int) -> Dict[int, Optional[int]]:
    parents: Dict[int, Optional[int]] = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    return parents


def reconstruct_forward_path(source: int, target: int, parents: Dict[int, Optional[int]]):
    if target not in parents:
        return None
    path = [target]
    while path[-1] != source:
        parent = parents.get(path[-1])
        if parent is None:
            return None
        path.append(parent)
    path.reverse()
    return path


def format_path(layer: LayerRuntime, nodes: Sequence[int]) -> str:
    if not nodes:
        return ""
    labels = [layer.label(int(node)) for node in nodes]
    return " -> ".join(labels)


def select_cross_layer_targets(distances: np.ndarray, max_targets: int):
    """Select top-k cross-layer targets with finite positive distances."""
    candidates: List[Tuple[int, float]] = []
    for idx, value in enumerate(distances):
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val) or val <= 0:
            continue
        candidates.append((idx, val))
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[:max_targets]


def summarize_distances(distances: np.ndarray):
    valid = [float(v) for v in distances.flatten() if math.isfinite(v) and v > 0]
    if not valid:
        return "Hyper adjacency indicates no reachable nodes in the other layer."
    valid.sort()
    best = valid[0]
    median = float(np.median(valid))
    perc90 = float(np.percentile(valid, 90)) if len(valid) > 2 else valid[-1]
    return (
        f"best {best:.1f} hops, median {median:.1f}, "
        f"p90 {perc90:.1f}, reachable {len(valid)} nodes"
    )


def read_anchor_pairs(alignment_cfg: Dict, base_dir: str, layer_names: Sequence[str]):
    """
    读取 npz/npy 格式的锚节点对齐关系。
    支持：
        - npz（内部含一个或多个 .npy 数组）
        - npy
    数组格式要求：
        - shape = (num_pairs, >=2)
        - 第一列：layer_names[0] 的节点编号
        - 第二列：layer_names[1] 的节点编号
        - 如果想换列，可在 YAML config 里加 columns 指定列号
    """

    path = resolve_path(base_dir, alignment_cfg["path"])
    fmt = alignment_cfg.get("format", "npz").lower()

    if fmt not in {"npz", "npy"}:
        raise ValueError(f"Only npz/npy supported, but got: {fmt}")

    # 使用全局 load_array()，支持 npz/npy/json/csv...
    key = alignment_cfg.get("key")
    arr = load_array(path, key)
    arr = np.asarray(arr)

    # --- 校验数组 ---
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Alignment npz/npy must contain a 2D array with at least 2 columns, got {arr.shape}"
        )

    # 默认列映射：0 -> layer1, 1 -> layer2
    if "columns" in alignment_cfg:
        cols = alignment_cfg["columns"]
        if len(cols) != 2 or set(cols.keys()) != set(layer_names):
            raise ValueError("columns must specify exactly the two layer names.")
        col_a = int(cols[layer_names[0]])
        col_b = int(cols[layer_names[1]])
    else:
        col_a, col_b = 0, 1

    # --- 构造双向映射 ---
    mapping = {layer: {} for layer in layer_names}

    for row in arr:
        a = int(row[col_a])
        b = int(row[col_b])
        mapping[layer_names[0]][a] = b
        mapping[layer_names[1]][b] = a

    return mapping


def format_structural_sentence(layer: LayerRuntime, node_id: int, options: Dict):
    feats = options.get(
        "structural_features", DEFAULT_OPTIONS["structural_features"]
    )
    precision = options.get(
        "numeric_precision", DEFAULT_OPTIONS["numeric_precision"]
    )
    show_percentile = options.get(
        "show_percentile", DEFAULT_OPTIONS["show_percentile"]
    )
    fragments = []
    for feat in feats:
        values = layer.structural.get(feat)
        if values is None or node_id not in values:
            continue
        val = values[node_id]
        if feat in INT_METRICS:
            val_str = str(int(round(val)))
        else:
            val_str = f"{val:.{precision}f}"
        if show_percentile:
            perc = layer.percentiles.get(feat, {}).get(node_id)
            if perc is not None:
                val_str += f" (p{perc:.1f})"
        fragments.append(f"{METRIC_LABELS.get(feat, feat)} {val_str}")
    if not fragments:
        return ""
    return "Structural profile: " + ", ".join(fragments) + "."


def anchor_sentence(
    layer: LayerRuntime,
    node_id: int,
    other_layer: LayerRuntime,
    anchor_map: Dict[int, int],
    options: Dict,
    ):
    anchor_id = layer.closest_anchor.get(node_id)
    dist = layer.anchor_distance.get(node_id, math.inf)
    anchor_meta = {
        "closest_anchor": anchor_id,
        "distance_to_anchor": dist,
    }
    if anchor_id is None or anchor_id not in anchor_map:
        if math.isfinite(dist):
            text = (
                f"The node has no valid anchor mapping; "
                f"closest anchor distance is {dist:.1f}."
            )
        else:
            text = "The node is not connected to any anchor."
        return text, anchor_meta

    other_anchor = anchor_map[anchor_id]
    anchor_meta["matched_anchor"] = other_anchor
    if node_id == anchor_id:
        text = (
            f"This node is an anchor aligned with "
            f"{other_layer.label(other_anchor)}."
        )
        anchor_meta["path_to_anchor"] = [int(node_id)]
        return text, anchor_meta

    path = layer.path_to_anchor(node_id)
    anchor_meta["path_to_anchor"] = [int(n) for n in path] if path else []
    if not path:
        text = (
            f"It is {dist:.1f} hops from anchor {layer.label(anchor_id)} "
            f"but the in-layer path could not be reconstructed."
        )
        return text, anchor_meta

    text = (
        f"It reaches anchor {layer.label(anchor_id)} in {dist:.1f} hops via "
        f"{format_path(layer, path)} before crossing to "
        f"{other_layer.label(other_anchor)}."
    )
    return text, anchor_meta


def build_node_text(
    node_id: int,
    layer: LayerRuntime,
    other_layer: LayerRuntime,
    options: Dict,
    hyper_distances: np.ndarray,
    anchor_map: Dict[int, int],
    ):
    """Build textual description & metadata for a single node."""
    mode = options.get("mode", DEFAULT_OPTIONS["mode"])

    struct_sentence = format_structural_sentence(layer, node_id, options)
    anchor_text, anchor_meta = anchor_sentence(
        layer, node_id, other_layer, anchor_map, options
    )

    # shared pre-computations
    path_start = layer.path_to_anchor(node_id)
    matched_anchor = layer.closest_anchor.get(node_id)
    other_anchor = anchor_map.get(matched_anchor) if matched_anchor is not None else None

    max_targets = options.get(
        "max_cross_layer_targets", DEFAULT_OPTIONS["max_cross_layer_targets"]
    )
    targets = select_cross_layer_targets(hyper_distances, max_targets)

    include_paths = options.get(
        "include_paths", DEFAULT_OPTIONS["include_paths"]
    )

    cross_sentences: List[str] = []
    cross_meta: List[Dict] = []

    for target_idx, hop in targets:
        target_label = other_layer.label(target_idx)
        path_to_target = (
            other_layer.path_from_anchor(other_anchor, target_idx)
            if other_anchor is not None
            else None
        )

        meta_entry = {
            "target_node": int(target_idx),
            "target_label": target_label,
            "hyper_distance": hop,
            "anchor_on_other_layer": other_anchor,
        }
        if include_paths:
            meta_entry["path_start_layer"] = (
                [int(n) for n in path_start] if path_start else []
            )
            meta_entry["path_other_layer"] = (
                [int(n) for n in path_to_target] if path_to_target else []
            )

        if path_start and path_to_target:
            part = (
                f"It reaches {target_label} in {hop:.1f} mixed-layer hops via "
                f"{format_path(layer, path_start)} || "
                f"{format_path(other_layer, path_to_target)}."
            )
        elif path_start:
            part = (
                f"Hyper adjacency reports {hop:.1f} hops to {target_label}, "
                f"but the path on {other_layer.name} could not be reconstructed."
            )
        else:
            part = (
                f"Hyper adjacency reports {hop:.1f} hops to {target_label}."
            )

        cross_sentences.append(part)
        cross_meta.append(meta_entry)

    stats_sentence = summarize_distances(hyper_distances)

    # 根据模式选择不同的文本拼接方式
    if mode == "compact":
        # 更适合给 LLM/LinkGPT 用：短一点，信息集中
        # 只保留一个 cross target（如果有）
        compact_cross_sent = cross_sentences[0] if cross_sentences else ""
        pieces = [
            f"Node {layer.label(node_id)} in graph {layer.name}.",
            struct_sentence,
            anchor_text,
            compact_cross_sent,
        ]
    else:
        # verbose 模式：保留所有统计信息和 cross-layer 描述
        pieces = [
            f"Node {layer.label(node_id)} in layer {layer.name}.",
            struct_sentence,
            anchor_text,
            "Cross-layer reachability summary: " + stats_sentence + ".",
            " ".join(cross_sentences),
        ]

    text = " ".join(piece for piece in pieces if piece)

    metadata = {
        "layer": layer.name,
        "node_id": int(node_id),
        "node_label": layer.label(node_id),
        "structural_features": {
            feat: layer.structural.get(feat, {}).get(node_id)
            for feat in layer.structural
        },
        "structural_percentiles": {
            feat: layer.percentiles.get(feat, {}).get(node_id)
            for feat in layer.structural
        },
        "anchor": anchor_meta,
        "cross_layer_targets": cross_meta,
    }
    return text, metadata


def prepare_layers(config: Dict, config_path: str):
    layers_cfg = config["layers"]
    if len(layers_cfg) != 2:
        raise ValueError("The current implementation expects exactly two layers.")
    base_dir = osp.dirname(osp.abspath(config_path))

    layer_runtimes: Dict[str, LayerRuntime] = {}

    for layer_cfg in layers_cfg:
        name = layer_cfg["name"]
        prefix = layer_cfg.get("prefix", name)
        edge_path = resolve_path(base_dir, layer_cfg["edge_path"])
        edge_key = layer_cfg.get("edge_key")
        feature_path = layer_cfg.get("feature_path")
        feature_key = layer_cfg.get("feature_key")

        features = None
        num_nodes = layer_cfg.get("num_nodes")

        if feature_path:
            features = load_array(resolve_path(base_dir, feature_path), feature_key)
            if features.ndim == 1:
                # 修正：一维特征视作 (num_nodes, 1)，而不是 (1, num_features)
                features = features.reshape(-1, 1)
            num_nodes = features.shape[0]

        # 只 load 一次 edge_array
        edge_array = load_array(edge_path, edge_key)
        edge_index = normalize_edge_index(edge_array)

        if num_nodes is None:
            max_idx = int(edge_index.max())
            num_nodes = max_idx + 1

        graph = build_graph(
            edge_index, num_nodes, directed=layer_cfg.get("is_directed", False)
        )
        feats = config.get("text_options", {}).get(
            "structural_features", DEFAULT_OPTIONS["structural_features"]
        )
        structural = compute_structural_metrics(graph, feats)
        percentiles = {
            feat: compute_percentiles(values) for feat, values in structural.items()
        }
        layer_runtimes[name] = LayerRuntime(
            name=name,
            prefix=prefix,
            num_nodes=num_nodes,
            graph=graph,
            structural=structural,
            percentiles=percentiles,
        )

    layer_names = [layer_cfg["name"] for layer_cfg in layers_cfg]
    alignment = read_anchor_pairs(config["alignment"], base_dir, layer_names)

    # multi-source BFS from anchors on each layer
    for layer_name, runtime in layer_runtimes.items():
        anchors = list(alignment[layer_name].keys())
        dist, closest, prev = multi_source_bfs(runtime.graph, anchors)
        runtime.set_anchor_routing(dist, closest, prev)

    hyper_cfg = config["hyper_adjacency"]
    hyper_path = resolve_path(base_dir, hyper_cfg["path"])
    hyper_key = hyper_cfg.get("key")
    hyper_matrix = load_array(hyper_path, hyper_key)

    axis_order = hyper_cfg.get("axis_order", layer_names)
    if axis_order != layer_names and axis_order != layer_names[::-1]:
        raise ValueError("axis_order should list the same two layer names.")

    row_layer, col_layer = axis_order
    runtime_row = layer_runtimes[row_layer]
    runtime_col = layer_runtimes[col_layer]

    if hyper_matrix.shape != (runtime_row.num_nodes, runtime_col.num_nodes):
        raise ValueError(
            f"Hyper adjacency shape {hyper_matrix.shape} does not match layer sizes "
            f"({runtime_row.num_nodes}, {runtime_col.num_nodes})."
        )

    return layer_runtimes, alignment, hyper_matrix, row_layer, col_layer


def generate_texts(config: Dict, config_path: str):
    layers, anchor_maps, hyper_matrix, row_layer, col_layer = prepare_layers(
        config, config_path
    )
    options = DEFAULT_OPTIONS.copy()
    options.update(config.get("text_options", {}))

    row_runtime = layers[row_layer]
    col_runtime = layers[col_layer]

    text_output: Dict[str, Dict[int, str]] = {row_layer: {}, col_layer: {}}
    metadata: List[Dict] = []

    # row → col direction
    for node_id in range(row_runtime.num_nodes):
        text, meta = build_node_text(
            node_id,
            row_runtime,
            col_runtime,
            options,
            hyper_matrix[node_id],
            anchor_maps[row_layer],
        )
        text_output[row_layer][int(node_id)] = text
        metadata.append(meta)

    # col → row direction (using transpose of hyper adjacency)
    for node_id in range(col_runtime.num_nodes):
        text, meta = build_node_text(
            node_id,
            col_runtime,
            row_runtime,
            options,
            hyper_matrix[:, node_id],
            anchor_maps[col_layer],
        )
        text_output[col_layer][int(node_id)] = text
        metadata.append(meta)

    return text_output, metadata

def parse_args():
    parser = argparse.ArgumentParser(description="Generate node text for multi-layer graphs.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print N sample texts without truncating output files.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Do not write outputs, only preview.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    texts, metadata = generate_texts(config, args.config)

    # preview 部分
    if args.preview > 0:
        count = 0
        for layer_name, node_map in texts.items():
            for node_id, text in node_map.items():
                print(f"[{layer_name}] node {node_id}: {text}\n")
                count += 1
                if count >= args.preview:
                    break
            if count >= args.preview:
                break

    if args.skip_save:
        return

    output_cfg = config["output"]
    base_dir = osp.dirname(osp.abspath(args.config))

    text_path = resolve_path(base_dir, output_cfg["text_path"])
    metadata_path = resolve_path(base_dir, output_cfg["metadata_path"])

    os.makedirs(osp.dirname(text_path), exist_ok=True)
    os.makedirs(osp.dirname(metadata_path), exist_ok=True)

    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved node texts to {text_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()