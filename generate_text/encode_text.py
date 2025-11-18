import os
import os.path as osp
import numpy as np
import pandas as pd
import tqdm
import torch
from sentence_transformers import SentenceTransformer
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description='Encode text data for multilayer graph nodes')
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to .npz file, e.g. douban.npz')
    parser.add_argument('--enc_model', type=str, default='minilm',
                        choices=['albert', 'roberta', 'minilm', 'mpnet'])
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--wo_neigh', action='store_true')
    return parser.parse_args()


def get_prefix(npz_path):
    return osp.splitext(osp.basename(npz_path))[0]


def extract_layers(npz_file):
    return sorted(set(
        int(re.search(r'\d+', k).group()) for k in npz_file.files if re.search(r'^x\d+$', k)
    ))


def load_llm_texts(data_name, llm_model, wo_neigh=False, prefix=''):
    ans_file = f'answer_{llm_model}_wo_neigh.csv' if wo_neigh else f'answer_{llm_model}.csv'
    ans_path = osp.join(f'/root/autodl-tmp/{prefix}/response', data_name, ans_file)
    df = pd.read_csv(ans_path, sep='\t')
    return df['answer'].tolist()


def encode_text_sbert(texts, model_name, batch_size=256, device=None):
    model = SentenceTransformer(model_name, device=device)  # device="cuda" / "cpu"
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,      # 直接返回 torch.Tensor
        show_progress_bar=True,
        normalize_embeddings=False    # 视任务需求
    )
    return emb  # [N, D] torch.FloatTensor



def main():
    args = parse_args()
    prefix = get_prefix(args.npz_path)
    npz_data = np.load(args.npz_path)
    layers = extract_layers(npz_data)

    # Select encoder model
    model_map = {
        'albert': 'paraphrase-albert-small-v2',
        'roberta': 'all-distilroberta-v1',
        'minilm': 'all-MiniLM-L12-v2',
        'mpnet': 'all-mpnet-base-v2'
    }
    sbert_model = model_map[args.enc_model]

    for layer in layers:
        layer_name = f'{prefix}{layer}'
        print(f'Encoding text for {layer_name}...')

        try:
            answers = load_llm_texts(layer_name, args.llm_model, args.wo_neigh, prefix=prefix)
        except FileNotFoundError:
            print(f'Skipping {layer_name}: missing answer file')
            continue

        texts = [
            f"The node type is unknown. The node description is na. The additional node description is {a}."
            for a in answers
        ]

        embeddings = encode_text_sbert(texts, sbert_model)

        save_dir = f'/root/autodl-tmp/{prefix}/text_emb'
        os.makedirs(save_dir, exist_ok=True)

        # 文件名格式：{prefix}{layer}_{llm_model}_{enc_model}.pt
        file_name = f"{prefix}{layer}_{args.llm_model}_{args.enc_model}"
        if args.wo_neigh:
            file_name += "_wo_neigh"

        save_path = osp.join(save_dir, f"{file_name}.pt")
        torch.save(embeddings, save_path)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
