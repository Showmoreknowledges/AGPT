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
        - edge_index1, edge_index2：图的边索引
        - x1, x2：节点特征或文本（可为空）
        - pos_pairs, test_pairs：训练与测试节点对
    """
    def __init__(self, npz_path):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)

        self.edge_index1 = torch.from_numpy(data["edge_index1"]).long()
        self.edge_index2 = torch.from_numpy(data["edge_index2"]).long()

        # 节点特征（或文本）
        self.x1 = data["x1"].item() if data["x1"].shape == () else data["x1"]
        self.x2 = data["x2"].item() if data["x2"].shape == () else data["x2"]
        if self.x1 is None:
            self.x1 = torch.zeros((int(self.edge_index1.max()) + 1, 1))
        if self.x2 is None:
            self.x2 = torch.zeros((int(self.edge_index2.max()) + 1, 1))

        self.pos_pairs = torch.from_numpy(data["pos_pairs"]).long()
        self.test_pairs = torch.from_numpy(data["test_pairs"]).long()

        self.num_nodes_1 = int(self.edge_index1.max()) + 1
        self.num_nodes_2 = int(self.edge_index2.max()) + 1

        print(f"✅ 数据加载成功: {npz_path}")
        print(f"图1: {self.num_nodes_1} 个节点, {self.edge_index1.shape[1]} 条边")
        print(f"图2: {self.num_nodes_2} 个节点, {self.edge_index2.shape[1]} 条边")
        print(f"训练对: {self.pos_pairs.shape[0]}, 测试对: {self.test_pairs.shape[0]}")

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        src, tgt = self.pos_pairs[idx]
        return {"src_id": src.item(), "tgt_id": tgt.item()}


def merge_graphs(edge_index1, edge_index2, x1=None, x2=None):
    def to_tensor(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, torch.Tensor):
            return x.float()
        raise TypeError(f"Unsupported feature type: {type(x)}")

    x1 = to_tensor(x1)
    x2 = to_tensor(x2)

    num_nodes_1 = int(edge_index1.max()) + 1
    num_nodes_2 = int(edge_index2.max()) + 1
    offset = num_nodes_1

    mapping_g2_to_merged = {i: i + offset for i in range(num_nodes_2)}
    edge_index2_shifted = edge_index2 + offset
    merged_edge_index = torch.cat([edge_index1, edge_index2_shifted], dim=1)

    if x1 is None:
        x1 = torch.zeros((num_nodes_1, 1))
    if x2 is None:
        x2 = torch.zeros((num_nodes_2, 1))
    merged_x = torch.cat([x1, x2], dim=0)

    print(f"✅ 合并完成: 总节点 {num_nodes_1 + num_nodes_2}, "
          f"总边 {merged_edge_index.shape[1]}")
    return merged_edge_index, merged_x, mapping_g2_to_merged, num_nodes_1, num_nodes_2, num_nodes_1 + num_nodes_2


def process_node_features_for_cgtp(x1, x2, num_nodes_1, num_nodes_2, mapping_g2_to_merged):
    gnid2text, merged_features = None, None

    def is_textual(x):
        if x is None:
            return False
        if isinstance(x, np.ndarray) and x.dtype.type is np.str_:
            return True
        if isinstance(x, (list, tuple)):
            return all(isinstance(i, str) for i in x)
        if isinstance(x, dict):
            return all(isinstance(v, str) for v in x.values())
        return False

    if is_textual(x1) or is_textual(x2):
        gnid2text = {}
        if isinstance(x1, dict):
            for nid, text in x1.items():
                gnid2text[int(nid)] = text
        elif isinstance(x1, (list, tuple)):
            for nid, text in enumerate(x1):
                gnid2text[nid] = text
        if isinstance(x2, dict):
            for orig_id, text in x2.items():
                gnid2text[mapping_g2_to_merged[int(orig_id)]] = text
        elif isinstance(x2, (list, tuple)):
            for orig_id, text in enumerate(x2):
                gnid2text[mapping_g2_to_merged[orig_id]] = text
        print(f"✅ 已生成 gnid2text 字典，共 {len(gnid2text)} 条文本节点")
    else:
        def to_tensor(x, n):
            if x is None:
                return torch.zeros((n, 1))
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            if isinstance(x, torch.Tensor):
                return x.float()
            raise TypeError(f"Unsupported feature type: {type(x)}")

        merged_features = torch.cat([to_tensor(x1, num_nodes_1), to_tensor(x2, num_nodes_2)], dim=0)
        print(f"✅ 已生成 merged_features 特征矩阵，形状: {merged_features.shape}")
    return gnid2text, merged_features


def process_alignment_pairs(pos_pairs, test_pairs, num_nodes_1):
    if not isinstance(pos_pairs, torch.Tensor):
        pos_pairs = torch.tensor(pos_pairs, dtype=torch.long)
    if not isinstance(test_pairs, torch.Tensor):
        test_pairs = torch.tensor(test_pairs, dtype=torch.long)
    train_pairs_merged, test_pairs_merged = pos_pairs.clone(), test_pairs.clone()
    train_pairs_merged[:, 1] += num_nodes_1
    test_pairs_merged[:, 1] += num_nodes_1
    print(f"✅ 节点对映射完成: train={train_pairs_merged.shape}, test={test_pairs_merged.shape}")
    return train_pairs_merged, test_pairs_merged


class TAGDatasetForLM(Dataset):
    def __init__(self, merged_edge_index, gnid2text=None, merged_features=None):
        super().__init__()
        self.edge_index = merged_edge_index
        self.gnid2text = gnid2text
        self.features = merged_features
        self.num_nodes = (
            merged_features.shape[0] if merged_features is not None else len(gnid2text)
        )
        self.data_list = []
        for i in range(self.num_nodes):
            node_info = {"node_id": i, "neighbors": self._get_neighbors(i)}
            if self.gnid2text is not None:
                node_info["text"] = self.gnid2text.get(i, "")
            if self.features is not None:
                node_info["feature"] = self.features[i]
            self.data_list.append(node_info)
        print(f"✅ TAGDatasetForLM: 共 {self.num_nodes} 个节点样本")

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        return self.data_list[idx]

    def _get_neighbors(self, idx):
        src_mask = self.edge_index[0] == idx
        dst_mask = self.edge_index[1] == idx
        neighbors = torch.cat([self.edge_index[1][src_mask], self.edge_index[0][dst_mask]]).unique().tolist()
        return neighbors

    def get_neighbors_in_training_set(self, gnid):
        return self._get_neighbors(gnid)


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

