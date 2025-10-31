import os
import argparse
import pickle
import json
import random
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_sparse import SparseTensor  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SparseTensor = None


class MultiLayerGraphDataset(Dataset):
    """
    加载并封装多层网络（含锚点对），用于 LinkGPT 前置数据处理。
    支持字段：
        - edge_index{layer}：每一层图的边索引（层号从 1 开始）
        - x{layer}：对应层的节点特征或文本（可为空）
        - pos_pairs, test_pairs：训练与测试节点对
    """
    def __init__(self, npz_path):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)

        def _layer_index_from_key(key: str) -> int:
            suffix = key[len("edge_index"):]
            digits = ''.join(ch for ch in suffix if ch.isdigit())
            return int(digits) if digits else 1

        def _ensure_edge_tensor(edge_value) -> torch.Tensor:
            if isinstance(edge_value, torch.Tensor):
                return edge_value.long()
            if isinstance(edge_value, np.ndarray):
                return torch.from_numpy(edge_value).long()
            raise TypeError(f"Unsupported edge_index type: {type(edge_value)}")

        def _normalise_feature(raw_value, n_nodes):
            if isinstance(raw_value, np.ndarray):
                if raw_value.shape == ():
                    return _normalise_feature(raw_value.item(), n_nodes)
                if raw_value.dtype.kind in {'U', 'S', 'O'}:
                    return raw_value.tolist()
                return torch.from_numpy(raw_value).float()
            if isinstance(raw_value, torch.Tensor):
                return raw_value.float()
            if isinstance(raw_value, (list, tuple)):
                try:
                    tensor = torch.tensor(raw_value, dtype=torch.float32)
                except (TypeError, ValueError):
                    return list(raw_value)
                if tensor.dim() == 1:
                    tensor = tensor.view(n_nodes, -1)
                return tensor
            if isinstance(raw_value, dict):
                return raw_value
            if raw_value is None:
                return torch.zeros((n_nodes, 1), dtype=torch.float32)
            return raw_value

        edge_keys = sorted(
            [key for key in data.files if key.startswith("edge_index")],
            key=_layer_index_from_key,
        )
        if not edge_keys:
            raise ValueError(f"未在 {npz_path} 中找到任何以 'edge_index' 开头的键。")

        self.layer_ids = []
        self.edge_indices = []
        self.features = []
        self.num_nodes_per_layer = []

        for order, edge_key in enumerate(edge_keys, 1):
            layer_id = _layer_index_from_key(edge_key)
            edge_index = _ensure_edge_tensor(data[edge_key])

            feature_key = f"x{layer_id}"
            feature_raw = data[feature_key] if feature_key in data.files else None
            if isinstance(feature_raw, np.ndarray) and feature_raw.shape == ():
                feature_raw = feature_raw.item()

            num_nodes = None
            if isinstance(feature_raw, torch.Tensor):
                num_nodes = feature_raw.size(0)
            elif isinstance(feature_raw, np.ndarray) and feature_raw.ndim >= 1:
                num_nodes = feature_raw.shape[0]
            elif isinstance(feature_raw, (list, tuple)):
                num_nodes = len(feature_raw)
            elif isinstance(feature_raw, dict):
                num_nodes = len(feature_raw)

            if num_nodes is None:
                num_nodes_key = f"num_nodes{layer_id}"
                if num_nodes_key in data.files:
                    num_nodes = int(data[num_nodes_key])
                elif edge_index.numel() > 0:
                    num_nodes = int(edge_index.max().item()) + 1
                else:
                    num_nodes = 0

            feature = _normalise_feature(feature_raw, num_nodes)

            self.layer_ids.append(layer_id)
            self.edge_indices.append(edge_index)
            self.features.append(feature)
            self.num_nodes_per_layer.append(int(num_nodes))

        self.layer_offsets = []
        running = 0
        for n in self.num_nodes_per_layer:
            self.layer_offsets.append(running)
            running += n
        self.total_num_nodes = running
        self.num_layers = len(self.edge_indices)

        self.edge_index1 = self.edge_indices[0]
        self.x1 = self.features[0]
        self.num_nodes_1 = self.num_nodes_per_layer[0]
        if self.num_layers > 1:
            self.edge_index2 = self.edge_indices[1]
            self.x2 = self.features[1]
            self.num_nodes_2 = self.num_nodes_per_layer[1]
        else:
            self.edge_index2 = None
            self.x2 = None
            self.num_nodes_2 = 0

        
        self.pos_pairs = torch.as_tensor(data["pos_pairs"], dtype=torch.long)
        self.test_pairs = torch.as_tensor(data["test_pairs"], dtype=torch.long)

        print(f"✅ 数据加载成功: {npz_path}")
        for idx, (n_nodes, edge_index) in enumerate(zip(self.num_nodes_per_layer, self.edge_indices), 1):
            num_edges = edge_index.size(1) if edge_index.dim() == 2 else edge_index.numel() // 2
            print(f"图{idx}: {n_nodes} 个节点, {num_edges} 条边")
        print(f"共 {self.num_layers} 层, 总节点 {self.total_num_nodes}")

        print(f"训练对: {self.pos_pairs.shape[0]}, 测试对: {self.test_pairs.shape[0]}")

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        pair = self.pos_pairs[idx]
        if isinstance(pair, torch.Tensor):
            pair = pair.clone()
            if pair.dim() == 1 and pair.size(0) == 2:
                src, tgt = pair
                return {"src_id": src.item(), "tgt_id": tgt.item()}
        return {"pair": pair}


def merge_graphs(edge_index1, edge_index2=None, x1=None, x2=None):
    def _to_edge_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.long()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).long()
        raise TypeError(f"Unsupported edge_index type: {type(value)}")

    if isinstance(edge_index1, (list, tuple)):
        edge_indices_raw = list(edge_index1)
        if edge_index2 is None:
            feature_list = [None] * len(edge_indices_raw)
        elif isinstance(edge_index2, (list, tuple)):
            feature_list = list(edge_index2)
        else:
            raise TypeError("When passing multiple edge indices, `edge_index2` should be a sequence of features or None.")
    else:
        if edge_index2 is None:
            raise ValueError("`edge_index2` must be provided when passing individual tensors.")
        edge_indices_raw = [edge_index1, edge_index2]
        feature_list = [x1, x2]

    if len(feature_list) < len(edge_indices_raw):
        feature_list.extend([None] * (len(edge_indices_raw) - len(feature_list)))
    elif len(feature_list) > len(edge_indices_raw):
        feature_list = feature_list[:len(edge_indices_raw)]

    edge_indices = [_to_edge_tensor(idx) for idx in edge_indices_raw]
    num_nodes_per_layer = []
    for edge_index, feature in zip(edge_indices, feature_list):
        num_nodes = None
        if isinstance(feature, torch.Tensor):
            num_nodes = feature.size(0)
        elif isinstance(feature, np.ndarray) and feature.ndim >= 1:
            num_nodes = feature.shape[0]
        elif isinstance(feature, (list, tuple)):
            num_nodes = len(feature)
        elif isinstance(feature, dict):
            num_nodes = len(feature)

        if num_nodes is None:
            if edge_index.numel() > 0:
                num_nodes = int(edge_index.max().item()) + 1
            else:
                num_nodes = 0
        num_nodes_per_layer.append(int(num_nodes))

    offsets = []
    running = 0
    for n in num_nodes_per_layer:
        offsets.append(running)
        running += n
    total_nodes = running

    merged_edges = []
    for edge_index, offset in zip(edge_indices, offsets):
        if edge_index.numel() == 0:
            continue
        merged_edges.append(edge_index + offset)
    if merged_edges:
        merged_edge_index = torch.cat(merged_edges, dim=1)
    else:
        merged_edge_index = torch.zeros((2, 0), dtype=torch.long)

    def _feature_to_tensor(feature, n_nodes):
        if feature is None:
            return torch.zeros((n_nodes, 1), dtype=torch.float32)
        if isinstance(feature, torch.Tensor):
            return feature.float()
        if isinstance(feature, np.ndarray):
            if feature.shape == ():
                return torch.zeros((n_nodes, 1), dtype=torch.float32)
            if feature.dtype.kind in {"U", "S", "O"}:
                return None
            return torch.from_numpy(feature).float()
        if isinstance(feature, (list, tuple)):
            try:
                tensor = torch.tensor(feature, dtype=torch.float32)
            except (TypeError, ValueError):
                return None
            if tensor.dim() == 1:
                tensor = tensor.view(n_nodes, -1)
            return tensor
        if isinstance(feature, dict):
            return None
        return None

    feature_tensors = []
    numeric_features = True
    for feature, n_nodes in zip(feature_list, num_nodes_per_layer):
        tensor = _feature_to_tensor(feature, n_nodes)
        if tensor is None:
            numeric_features = False
            break
        feature_tensors.append(tensor)

    if numeric_features:
        if feature_tensors:
            merged_x = torch.cat(feature_tensors, dim=0)
        else:
            merged_x = torch.zeros((0, 1), dtype=torch.float32)
    else:
        merged_x = None

    print(f"✅ 合并完成: 总节点 {total_nodes}, 总边 {merged_edge_index.shape[1]}")
    
    return merged_edge_index, merged_x, offsets, num_nodes_per_layer, total_nodes

def process_node_features_for_cgtp(layer_features, num_nodes_per_layer, layer_offsets):
    gnid2text, merged_features = None, None

    def is_textual(value):
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            return False
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return isinstance(value.item(), str)
            return value.dtype.kind in {"U", "S", "O"}
        if isinstance(value, (list, tuple)):
            return all(isinstance(item, str) for item in value)
        if isinstance(value, dict):
            return all(isinstance(v, str) for v in value.values())
        return False

    if any(is_textual(feature) for feature in layer_features if feature is not None):
        gnid2text = {}
        for layer_idx, feature in enumerate(layer_features):
            if feature is None:
                continue
            offset = layer_offsets[layer_idx]
            if isinstance(feature, dict):
                for nid, text in feature.items():
                    gnid2text[offset + int(nid)] = text
            elif isinstance(feature, (list, tuple)):
                for nid, text in enumerate(feature):
                    gnid2text[offset + nid] = text
            elif isinstance(feature, np.ndarray):
                flat = feature.reshape(-1).tolist()
                for nid, text in enumerate(flat):
                    gnid2text[offset + nid] = text
        print(f"✅ 已生成 gnid2text 字典，共 {len(gnid2text)} 条文本节点")
    else:
        tensors = []
        for feature, n_nodes in zip(layer_features, num_nodes_per_layer):
            if isinstance(feature, torch.Tensor):
                tensor = feature.float()
            elif isinstance(feature, np.ndarray):
                tensor = torch.from_numpy(feature).float()
            elif feature is None:
                tensor = torch.zeros((n_nodes, 1), dtype=torch.float32)
            else:
                try:
                    tensor = torch.tensor(feature, dtype=torch.float32)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"Unsupported feature type: {type(feature)}") from exc
            if tensor.dim() == 1:
                tensor = tensor.view(n_nodes, -1)
            tensors.append(tensor)
        merged_features = torch.cat(tensors, dim=0) if tensors else torch.zeros((0, 1), dtype=torch.float32)
        print(f"✅ 已生成 merged_features 特征矩阵，形状: {merged_features.shape}")
    return gnid2text, merged_features


def process_alignment_pairs(pos_pairs, test_pairs, layer_offsets):
    if not isinstance(pos_pairs, torch.Tensor):
        pos_pairs = torch.as_tensor(pos_pairs, dtype=torch.long)
    if not isinstance(test_pairs, torch.Tensor):
        test_pairs = torch.as_tensor(test_pairs, dtype=torch.long)

    offsets = torch.tensor(layer_offsets, dtype=torch.long, device=pos_pairs.device)
    if offsets.numel() == 0:
        raise ValueError("`layer_offsets` should contain at least one entry.")

    def _apply_offsets(pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return pairs.new_zeros((0, 2))
        if pairs.dim() != 2:
            raise ValueError("`pairs` should be a 2D tensor.")
        if pairs.size(1) == 2:
            if offsets.numel() < 2:
                raise ValueError("At least two layers are required for 2-column alignment pairs.")
            src = pairs[:, 0] + offsets[0]
            tgt = pairs[:, 1] + offsets[1]
        elif pairs.size(1) == 3:
            tgt_layers = pairs[:, 2].long()
            if (tgt_layers < 0).any():
                raise ValueError("Layer indices must be non-negative.")
            if offsets.numel() <= tgt_layers.max().item():
                raise ValueError("Target layer index out of range in alignment pairs.")
            src = pairs[:, 0] + offsets[0]
            tgt = pairs[:, 1] + offsets[tgt_layers]
        elif pairs.size(1) == 4:
            src_layers = pairs[:, 0].long()
            tgt_layers = pairs[:, 2].long()
            if (src_layers < 0).any() or (tgt_layers < 0).any():
                raise ValueError("Layer indices must be non-negative.")
            if offsets.numel() <= max(src_layers.max().item(), tgt_layers.max().item()):
                raise ValueError("Layer index out of range in alignment pairs.")
            src = pairs[:, 1] + offsets[src_layers]
            tgt = pairs[:, 3] + offsets[tgt_layers]
        else:
            raise ValueError("`pairs` should have 2, 3 or 4 columns.")
        return torch.stack([src, tgt], dim=1)

    train_pairs_merged = _apply_offsets(pos_pairs)
    test_pairs_merged = _apply_offsets(test_pairs)
    print(f"✅ 节点对映射完成: train={train_pairs_merged.shape}, test={test_pairs_merged.shape}")
    return train_pairs_merged, test_pairs_merged


class AlignmentDataset(Dataset):
    """
    网络对齐数据集。

    既兼容旧版的 `AlignmentDataset(node_emb_path, pair_path, ppr_path)` 调用方式，
    也支持直接传入已加载的张量，且会在初始化阶段生成负样本并缓存，方便 Trainer 直接迭代。
    """

    def __init__(
        self,
        *legacy_paths: str,
        node_embeddings: Optional[torch.Tensor] = None,
        pairs: Optional[torch.Tensor] = None,
        ppr: Optional[Union[torch.Tensor, "SparseTensor", Dict[Tuple[int, int], float]]] = None,
        node_emb_path: Optional[str] = None,
        pair_path: Optional[str] = None,
        ppr_path: Optional[str] = None,
        neg_ratio: int = 1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if legacy_paths:
            if len(legacy_paths) != 3:
                raise ValueError("Legacy mode expects (node_emb_path, pair_path, ppr_path) as three positional arguments.")
            node_emb_path, pair_path, ppr_path = legacy_paths  # type: ignore[misc]

        self.node_emb = self._load_tensor(node_embeddings, node_emb_path, "node_embeddings")
        self.pairs = self._load_tensor(pairs, pair_path, "pairs", dtype=torch.long)
        self.ppr_lookup = self._prepare_ppr(self._load_optional_ppr(ppr, ppr_path))

        if self.pairs.dim() != 2 or self.pairs.size(1) != 2:
            raise ValueError("`pairs` should be a 2D tensor with shape [N, 2].")

        self.pos_pairs = self.pairs.long().cpu()
        self.neg_ratio = max(int(neg_ratio), 0)
        self.seed = seed
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        self.target_candidates = self._infer_target_candidates(self.pos_pairs)
        self.neg_pairs = self._build_negative_pairs()

        pos_labels = torch.ones(self.pos_pairs.size(0), dtype=torch.float32)
        if self.neg_pairs.numel() > 0:
            neg_labels = torch.zeros(self.neg_pairs.size(0), dtype=torch.float32)
            self.all_pairs = torch.cat([self.pos_pairs, self.neg_pairs], dim=0)
            self.labels = torch.cat([pos_labels, neg_labels], dim=0)
        else:
            self.all_pairs = self.pos_pairs
            self.labels = pos_labels

        if self.shuffle and len(self.all_pairs) > 1:
            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(self.all_pairs), generator=generator)
            self.all_pairs = self.all_pairs[perm]
            self.labels = self.labels[perm]

        print(
            f"✅ AlignmentDataset 准备完成: 正样本 {self.pos_pairs.size(0)}, "
            f"负样本 {self.neg_pairs.size(0)}, 嵌入维度 {self.node_emb.shape[1]}"
        )

    def __len__(self) -> int:
        return self.all_pairs.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u, v = self.all_pairs[idx]
        label = self.labels[idx]

        sample = {
            "u_id": u.long(),
            "v_id": v.long(),
            "u_emb": self.node_emb[u].float(),
            "v_emb": self.node_emb[v].float(),
            "labels": label,
        }

        sample["ppr_score"] = self._lookup_ppr(int(u), int(v))
        return sample

    def resample_negatives(self) -> None:
        """重新采样负样本，可在每个 epoch 前调用以提升训练稳定性。"""
        self.neg_pairs = self._build_negative_pairs()
        pos_labels = torch.ones(self.pos_pairs.size(0), dtype=torch.float32)
        if self.neg_pairs.numel() > 0:
            neg_labels = torch.zeros(self.neg_pairs.size(0), dtype=torch.float32)
            self.all_pairs = torch.cat([self.pos_pairs, self.neg_pairs], dim=0)
            self.labels = torch.cat([pos_labels, neg_labels], dim=0)
        else:
            self.all_pairs = self.pos_pairs
            self.labels = pos_labels
        if self.shuffle and len(self.all_pairs) > 1:
            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(self.all_pairs), generator=generator)
            self.all_pairs = self.all_pairs[perm]
            self.labels = self.labels[perm]

    @staticmethod
    def _load_tensor(
        tensor: Optional[torch.Tensor],
        path: Optional[str],
        name: str,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if tensor is None:
            if path is None:
                raise ValueError(f"`{name}` must be provided either as a tensor or a file path.")
            tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor, but got {type(tensor)}.")
        tensor = tensor.detach().cpu()
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    @staticmethod
    def _load_optional_ppr(
        ppr: Optional[Union[torch.Tensor, "SparseTensor", Dict[Tuple[int, int], float]]],
        path: Optional[str],
    ) -> Optional[Union[torch.Tensor, "SparseTensor", Dict[Tuple[int, int], float]]]:
        if ppr is None and path is not None:
            ppr = torch.load(path, map_location="cpu")
        return ppr

    def _prepare_ppr(
        self,
        ppr: Optional[Union[torch.Tensor, "SparseTensor", Dict[Tuple[int, int], float]]],
    ) -> Optional[Union[torch.Tensor, "SparseTensor", Dict[Tuple[int, int], float]]]: # <--- 修改返回类型
        if ppr is None:
            return None
        if isinstance(ppr, dict):
            # 如果已经是字典，直接使用
            return {(int(k[0]), int(k[1])): float(v) for k, v in ppr.items()}
        if isinstance(ppr, torch.Tensor):
            if ppr.is_sparse:
                # <--- 不再转换为字典 --->
                print("⚠️ 警告: PPR 是稀疏张量。保持稀疏格式以节省内存，但查找可能会变慢。")
                return ppr.coalesce().cpu() # 保持为稀疏张量
            return ppr.float().cpu() # 保持为密集张量
        if SparseTensor is not None and isinstance(ppr, SparseTensor):
            # <--- 不再转换为字典 --->
            print("⚠️ 警告: PPR 是 SparseTensor。保持稀疏格式以节省内存，但查找可能会变慢。")
            return ppr.cpu() # 保持为 SparseTensor
        raise TypeError(f"Unsupported PPR type: {type(ppr)}")

    def _lookup_ppr(self, u: int, v: int) -> torch.Tensor:
        if self.ppr_lookup is None:
            return torch.tensor(0.0, dtype=torch.float32)
        if isinstance(self.ppr_lookup, torch.Tensor):
            # 适用于密集张量，或（低效的）稀疏张量查找
            return self.ppr_lookup[u, v].view(1).float() 
        if SparseTensor is not None and isinstance(self.ppr_lookup, SparseTensor):
            # <--- 处理 SparseTensor (低效) --->
            # 注意：这是低效的逐元素查找
            return self.ppr_lookup[u, v].view(1).float()
        
        # 适用于字典
        return torch.tensor(self.ppr_lookup.get((u, v), 0.0), dtype=torch.float32).view(1)

    def _infer_target_candidates(self, pairs: torch.Tensor) -> Sequence[int]:
        target_ids = torch.unique(pairs[:, 1]).cpu().tolist()
        return target_ids

    def _build_negative_pairs(self) -> torch.Tensor:
        if self.neg_ratio <= 0 or len(self.target_candidates) <= 1:
            return torch.zeros((0, 2), dtype=torch.long)

        neg_samples = []
        for src, tgt in self.pos_pairs.tolist():
            for _ in range(self.neg_ratio):
                neg_tgt = self._sample_negative_target(int(tgt))
                neg_samples.append((int(src), neg_tgt))
        if not neg_samples:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.tensor(neg_samples, dtype=torch.long)

    def _sample_negative_target(self, positive_tgt: int) -> int:
        neg_tgt = positive_tgt
        while neg_tgt == positive_tgt:
            neg_tgt = self.rng.choice(self.target_candidates)
        return neg_tgt

