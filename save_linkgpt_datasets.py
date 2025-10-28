import os
import pickle
import argparse
import sys
from linkgpt.dataset.tag_dataset_for_lm import TAGDatasetForLM
from linkgpt.dataset.yn_dataset import YNDataset, YNDatasetConfig
from linkgpt.dataset.np_dataset import NPDataset, NPDatasetConfig
from linkgpt.utils import basics


project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="包含 dataset_for_lm.pkl 的目录")
    parser.add_argument('--dataset_name', required=True, help="数据集的名称 (例如 acm-dblp)")
    args = parser.parse_args()

    dataset_for_lm_path = os.path.join(args.data_dir, 'dataset_for_lm.pkl')
    print(f"正在加载: {dataset_for_lm_path}")
    dataset_for_lm = basics.load_pickle(dataset_for_lm_path)

    # 1. 创建并保存 LP (网络对齐) 数据集
    # YNDataset 会自动从 dataset_for_lm.edge_split['train'] 读取正样本
    # 并自动采样负样本
    print("正在创建 LP (Alignment) YNDataset...")
    lp_config = YNDatasetConfig(
        dataset_name=args.dataset_name,
        ablate_pairwise_encoding=False,
        ablate_node_encoding=False,
        learn_text=True # 确保模型学习文本
    )
    lp_dataset = YNDataset(lp_config, dataset_for_lm)
    lp_output_path = os.path.join(args.data_dir, 'lp_dataset.pkl')
    basics.save_pickle(lp_dataset, lp_output_path)
    print(f"已保存 LP (Alignment) 数据集到: {lp_output_path}")

    # 2. 创建并保存 NP (邻居预测) 数据集
    # 即使我们只关心对齐 (LP)，train.py 也需要这个文件
    print("正在创建 NP (Neighbor Prediction) NPDataset...")
    np_config = NPDatasetConfig(
        dataset_name=args.dataset_name,
        ablate_node_encoding=False,
        learn_src_text=True # 确保模型学习文本
    )
    np_dataset = NPDataset(np_config, dataset_for_lm)
    np_output_path = os.path.join(args.data_dir, 'np_dataset.pkl')
    basics.save_pickle(np_dataset, np_output_path)
    print(f"已保存 NP (Neighbor) 数据集到: {np_output_path}")

if __name__ == '__main__':
    main()