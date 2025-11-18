import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description='Generate prompts for multi-layer graph')
    parser.add_argument('--npz_path', type=str, default='../../data/dataset/douban/douban.npz',
                        help='Path to multi-layer .npz dataset')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    return parser.parse_args()


def check_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


templates = {
    'base': "Given a node from a general graph, where the node type is unknown with {} nodes, and the edge type is undirected with {} edges. ",
    'property': 'The value of property "{}" is {:.4f}, ranked at {} among {} nodes. ',
    'final': "Output the activity level of the node and provide reasons for your assessment. Your answer should be less than 200 words. "
}


def process_topological_features(topo_path):
    topo_features = np.load(topo_path, allow_pickle=True).item()

    rename = {
        'clustering': 'Clustering Coefficient',
        'degree': 'Node Degree',
        'square_clustering': 'Square Clustering Coefficient',
        'closeness': 'Closeness Centrality',
        'betweenness': 'Betweenness Centrality',
    }
    topo_features = {rename[k]: v for k, v in topo_features.items()}

    total_node = len(next(iter(topo_features.values())))
    topo_features_rank = {}
    for method, values in topo_features.items():
        sorted_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
        rank = 0
        pre_value = -1
        node2rank = {}
        for node, value in sorted_values.items():
            if value != pre_value:
                rank += 1
                pre_value = value
            node2rank[node] = rank
        topo_features_rank[method] = node2rank

    new_topo_features = {}
    for method, values in topo_features.items():
        for node, value in values.items():
            if node not in new_topo_features:
                new_topo_features[node] = {}
            new_topo_features[node][method] = (value, topo_features_rank[method][node])

    return new_topo_features, total_node


def generate_text(node, methods, total_node, total_edge):
    text = templates['base'].format(total_node, total_edge)
    for method, (value, rank) in methods.items():
        text += templates['property'].format(method, value, rank, total_node)
    text += templates['final']
    return text


def extract_layers(npz_file):
    """Extract unique layer indices from keys like x1, edge_index2, etc."""
    keys = npz_file.files
    layer_ids = sorted(set(int(re.search(r'\d+', k).group()) for k in keys if re.search(r'\d+', k)))
    return layer_ids

def get_dataset_prefix(npz_path):
    return osp.splitext(osp.basename(npz_path))[0]

def main():
    args = parse_args()
    base_dir = osp.dirname(args.npz_path)
    prefix = get_dataset_prefix(args.npz_path)

    npz_data = np.load(args.npz_path)
    layers = extract_layers(npz_data)

    for layer in layers:
        layer_name = f"{prefix}{layer}"
        topo_path = osp.join(base_dir, f'/root/autodl-tmp/{prefix}/property/{layer_name}_topo.npy')
        edge_key = f'edge_index{layer}'

        if not osp.exists(topo_path) or edge_key not in npz_data:
            print(f"Skipping {layer_name} due to missing files.")
            continue

        total_edge = npz_data[edge_key].shape[1]
        topo_features, total_node = process_topological_features(topo_path)

        questions = [
            generate_text(node, methods, total_node, total_edge)
            for node, methods in topo_features.items()
        ]

        df = pd.DataFrame({
            'question': questions,
            'answer': 'Error',
            'node_idx': range(len(questions))
        })

        save_dir = osp.join(base_dir, f'/root/autodl-tmp/{prefix}/response/{layer_name}')
        check_dir(save_dir)

        df['question'].to_csv(osp.join(save_dir, 'question.csv'), index=False)
        df[['node_idx', 'answer']].to_csv(
            osp.join(save_dir, f'answer_{args.model}.csv'), index=False, sep='\t'
        )

        print(f"✓ Saved prompts for {layer_name} to {save_dir}")


if __name__ == "__main__":
    main()

