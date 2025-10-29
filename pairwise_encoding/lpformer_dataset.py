"""
Adapted from https://github.com/HarryShomer/LPFormer/blob/master/src/util/read_datasets.py
"""

import os
import argparse
import sys
import random

import torch
import numpy as np
from torch_sparse import SparseTensor
from torch.nn.init import xavier_uniform_
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, degree
import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset
import dgl

from ..dataset.tag_dataset_for_lm import TAGDatasetForLM
from ..utils import basics

def get_lpformer_dataset(dataset_name: str, split_edge, dgl_g, ppr_data, device):
    """
    Returns:
        dataset for LPFormer
    """
    data_obj = {
        "dataset": dataset_name,
    }
    
    homo_attr = getattr(dgl_g, "is_homogeneous", None)
    if homo_attr is None:
        is_homo_graph = True
    else:
        is_homo_graph = bool(homo_attr() if callable(homo_attr) else homo_attr)

    if is_homo_graph:
        num_nodes = dgl_g.num_nodes()
        primary_ntype = None
        relation_tensor = None
        edge_src, edge_dst = dgl_g.edges()
    else:
        primary_ntype = getattr(dgl_g, "_linkgpt_primary_ntype", dgl_g.ntypes[0])
        num_nodes = dgl_g.num_nodes(primary_ntype)
        hetero_graph = dgl_g.to(device)
        data_obj['hetero_graph'] = hetero_graph
        data_obj['hetero_primary_ntype'] = primary_ntype
        data_obj['use_relational_gnn'] = True

        edata_fields = []
        for canonical_etype in hetero_graph.canonical_etypes:
            rel_data = hetero_graph.edges[canonical_etype].data
            if 'relation_type' in rel_data:
                edata_fields = ['relation_type']
                break

        homo_kwargs = {'edata': edata_fields} if edata_fields else {}
        homograph = dgl.to_homogeneous(hetero_graph, **homo_kwargs)
        edge_src, edge_dst = homograph.edges()
        if edata_fields:
            relation_tensor = homograph.edata['relation_type'].to(device)
        else:
            relation_tensor = torch.zeros(edge_src.shape[0], dtype=torch.long, device=device)

    data_obj['num_nodes'] = num_nodes

    source, target = split_edge['train']['source_node'], split_edge['train']['target_node']
    source, target = torch.tensor(source), torch.tensor(target)
    data_obj['train_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)

    source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
    source, target = torch.tensor(source), torch.tensor(target)
    data_obj['valid_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
    pair_list = []
    for src, neg_tgt_list in zip(split_edge['valid']['source_node'], split_edge['valid']['target_node_neg']):
        pair_list += [[src, tgt] for tgt in neg_tgt_list]
    selected_items = random.sample(pair_list, 10000)
    data_obj['valid_neg'] = torch.tensor(selected_items).to(device) 

    source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
    source, target = torch.tensor(source), torch.tensor(target)
    data_obj['test_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
    pair_list = []
    for src, neg_tgt_list in zip(split_edge['test']['source_node'], split_edge['test']['target_node_neg']):
        pair_list += [[src, tgt] for tgt in neg_tgt_list]
    selected_items = random.sample(pair_list, 10000)
    data_obj['test_neg'] = torch.tensor(selected_items).to(device) 
    
    idx = torch.randperm(data_obj['train_pos'].size(0))[:data_obj['valid_pos'].size(0)]
    data_obj['train_pos_val'] = data_obj['train_pos'][idx]
    
    if is_homo_graph:
        node_features = dgl_g.ndata['feat']
    else:
        node_features = dgl_g.nodes[primary_ntype].data['feat']

    data_obj['x'] = node_features.to(device).to(torch.float)

    src_ls, tgt_ls = edge_src.reshape(1, -1), edge_dst.reshape(1, -1)
    edge_index = torch.concat([src_ls, tgt_ls], dim=0).to(device)
    
    edge_weight = torch.ones(edge_index.size(1)).to(device).float()
    data_obj['adj_t'] = SparseTensor.from_edge_index(
        edge_index, 
        edge_weight.squeeze(-1),
        [data_obj['num_nodes'], data_obj['num_nodes']]
    ).to(device)
    data_obj['adj_t'] = data_obj['adj_t'].to_symmetric().coalesce().to(device)
    data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()
    data_obj['adj_mask'] = data_obj['adj_mask'].coalesce().bool().int()
    data_obj['full_adj_t'] = data_obj['adj_t']
    data_obj['full_adj_mask'] = data_obj['adj_mask']
    data_obj['degree'] = degree(edge_index[0], num_nodes=data_obj['num_nodes']).to(device)
    
    data_obj['ppr'] = ppr_data.to(device)
    data_obj['ppr'] = data_obj['ppr'].to_torch_sparse_coo_tensor()
    data_obj['ppr_test'] = data_obj['ppr']
    
    train_pos = data_obj['train_pos'].cpu().numpy().tolist()
    pair_to_edge_idx = {tuple(pair): idx for idx, pair in enumerate(train_pos)}
    data_obj['pair_to_edge_idx'] = pair_to_edge_idx

    if relation_tensor is not None:
        data_obj['edge_relation_type'] = relation_tensor

    if not is_homo_graph:
        data_obj.setdefault('use_relational_gnn', True)

    return data_obj