import os
import os.path as osp
import numpy as np
import tqdm
import torch
import networkx as nx
import re
import argparse
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to the multi-layer graph .npz file (e.g., weibo.npz)')
    return parser.parse_args()


def check_path(path):
    if not osp.exists(path):
        os.makedirs(path)


def extract_dataset_prefix(npz_path):
    return osp.splitext(osp.basename(npz_path))[0]  # e.g. weibo.npz → weibo


def extract_layers(npz):
    keys = set(npz.files)
    layer_ids = []
    for key in keys:
        match = re.match(r'x(\d+)', key)
        if match and f'edge_index{match.group(1)}' in keys:
            layer_ids.append(int(match.group(1)))
    return sorted(layer_ids)


def load_layers(npz_path):
    npz = np.load(npz_path)
    prefix = extract_dataset_prefix(npz_path)
    layers = extract_layers(npz)
    data_dict = {}
    for i in layers:
        x = torch.tensor(npz[f'x{i}'], dtype=torch.float)
        edge_index = torch.tensor(npz[f'edge_index{i}'], dtype=torch.long)
        data_dict[f'{prefix}{i}'] = Data(x=x, edge_index=edge_index)
    return data_dict


def get_one_hop_neighbors(data):
    neighbors = {}
    for i in tqdm.tqdm(range(data.num_nodes)):
        neighbors[i] = list(data.edge_index[1][data.edge_index[0] == i].numpy().astype(int))
    return neighbors


def compute_node_properties(data):
    G = to_networkx(data, to_undirected=True)
    return {
        'square_clustering': nx.square_clustering(G),
        'clustering': nx.clustering(G),
        'degree': nx.centrality.degree_centrality(G),
        'closeness': nx.centrality.closeness_centrality(G),
        'betweenness': nx.centrality.betweenness_centrality(G)
    }


def main():
    args = parse_args()
    data_dict = load_layers(args.npz_path)
    prefix = extract_dataset_prefix(args.npz_path)
    save_dir = osp.join(osp.dirname(__file__), f'/root/autodl-tmp/{prefix}/property')
    check_path(save_dir)

    for name, data in data_dict.items():
        print(f"Processing {name}...")

        neighbors = get_one_hop_neighbors(data)
        np.save(osp.join(save_dir, f'{name}_one_hop.npy'), neighbors)

        topo = compute_node_properties(data)
        np.save(osp.join(save_dir, f'{name}_topo.npy'), topo)

        print(f"✓ Saved: {name}_one_hop.npy and {name}_topo.npy")


if __name__ == "__main__":
    main()
