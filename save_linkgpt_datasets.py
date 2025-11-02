import os
import argparse
import sys
import torch
from dataset.tag_dataset_for_lm import tag_dataset_for_lm_to_dgl_graph
from dataset.yn_dataset import YNDataset, YNDatasetConfig
from dataset.np_dataset import NPDataset, NPDatasetConfig
from model.linkgpt_model import get_tokenizer
from utils import basics


project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="包含 dataset_for_lm.pkl 的目录")
    parser.add_argument('--dataset_name', required=True, help="数据集的名称 (例如 acm-dblp)")
    parser.add_argument(
        '--model_name_or_path',default='openlm-research/open_llama_7b',
        help='用于构建提示的基础模型名称或本地路径，将用于加载分词器',
    )

    args = parser.parse_args()

    dataset_for_lm_path = os.path.join(f"{args.data_dir}/{args.dataset_name}", 'dataset_for_lm.pkl')
    print(f"正在加载: {dataset_for_lm_path}")
    dataset_for_lm = basics.load_pickle(dataset_for_lm_path)

    # 兼容不同版本的 edge_split 字段命名
    def ensure_field(entry, target_key, candidates, required=True):
        if target_key in entry:
            return
        for cand in candidates:
            if cand in entry:
                entry[target_key] = entry[cand]
                return
        if required:
            raise KeyError(
                f"edge_split 字段缺少 '{target_key}'，可接受的备选名称包括: {candidates}"
            )

    edge_split = getattr(dataset_for_lm, 'edge_split', None)
    if edge_split is None:
        raise ValueError('dataset_for_lm 缺少 edge_split，请确认预处理阶段已调用 generate_edge_split。')

    split_names = ['train', 'valid', 'test']
    for split in split_names:
        if split not in edge_split:
            continue
        entry = edge_split[split]
        ensure_field(entry, 'source_node', ['source_nodes', 'src', 'src_nodes', 'source', 'sources'])
        ensure_field(entry, 'target_node', ['target_nodes', 'dst', 'dst_nodes', 'target', 'targets'])
        # 负样本只在 valid/test 中需要
        ensure_field(
            entry,
            'target_node_neg',
            ['target_nodes_neg', 'neg_target_node', 'target_neg', 'negative_target_node', 'neg_targets', 'negative_targets'],
            required=(split != 'train'),
        )

    dgl_graph = tag_dataset_for_lm_to_dgl_graph(dataset_for_lm, device='cpu', include_valid=True)
    gnid2text = getattr(dataset_for_lm, 'gnid2text', None)

    if gnid2text is None:
        if hasattr(dataset_for_lm, 'data_list'):
            gnid2text = {
                int(item['node_id']): item.get('text', '')
                for item in dataset_for_lm.data_list
            }
        else:
            raise ValueError(
                'dataset_for_lm 中没有 gnid2text 信息，无法构建文本提示。'
                ' 请确认预处理阶段已为每个节点准备文本描述。'
            )

    tokenizer = get_tokenizer(args.model_name_or_path)

    # 1. 创建并保存 LP (网络对齐) 数据集
    # YNDataset 会自动从 dataset_for_lm.edge_split['train'] 读取正样本
    # 并自动采样负样本
    print("正在创建 LP (Alignment) YNDataset...")
    lp_config = YNDatasetConfig(
        ablate_pairwise_encoding=False,
        ablate_node_encoding=False,
        learn_text=True # 确保模型学习文本
    )
    lp_dataset = YNDataset(dgl_graph, gnid2text, lp_config, tokenizer)
    lp_dataset.config.dataset_name = args.dataset_name
    lp_output_path = os.path.join(f"{args.data_dir}/{args.dataset_name}", 'lp_dataset.pkl')
    basics.save_pickle(lp_dataset, lp_output_path)
    print(f"已保存 LP (Alignment) 数据集到: {lp_output_path}")

    # 2. 创建并保存 NP (邻居预测) 数据集
    # 即使我们只关心对齐 (LP)，train.py 也需要这个文件
    print("正在创建 NP (Neighbor Prediction) NPDataset...")
    np_config = NPDatasetConfig(
        ablate_node_encoding=False,
        learn_src_text=True # 确保模型学习文本
    )
    np_dataset = NPDataset(dgl_graph, gnid2text, np_config, tokenizer)
    np_dataset.config.dataset_name = args.dataset_name
    np_output_path = os.path.join(f"{args.data_dir}/{args.dataset_name}", 'np_dataset.pkl')
    basics.save_pickle(np_dataset, np_output_path)
    print(f"已保存 NP (Neighbor) 数据集到: {np_output_path}")

if __name__ == '__main__':
    main()