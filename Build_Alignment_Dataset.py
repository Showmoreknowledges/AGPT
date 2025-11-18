import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


data_dir = "./data"
layer1 = "douban1"
layer2 = "douban2"

config = {
    "one_hop_1": f"{data_dir}/{layer1}_one_hop.npy",
    "one_hop_2": f"{data_dir}/{layer2}_one_hop.npy",
    "topo_1": f"{data_dir}/{layer1}_topo.npy",
    "topo_2": f"{data_dir}/{layer2}_topo.npy",
    "text_file": f"{data_dir}/node_text.csv",
    "embed_file": f"{data_dir}/embedding.npy",
    "pairs_file": f"{data_dir}/alignment_pairs.csv",
    "gpt_answer_file": f"{data_dir}/answer_gpt-5.csv"
}

output_file = f"{data_dir}/alignment_train.jsonl"

# ========== 模块1：加载数据 ==========
def load_all_inputs(config):
    print("加载所有输入数据...")
    data = {}
    data['onehop_1'] = np.load(config['one_hop_1'], allow_pickle=True).item()
    data['onehop_2'] = np.load(config['one_hop_2'], allow_pickle=True).item()
    data['topo_1'] = np.load(config['topo_1'], allow_pickle=True).item()
    data['topo_2'] = np.load(config['topo_2'], allow_pickle=True).item()

    text_df = pd.read_csv(config['text_file'])
    data['id2text'] = dict(zip(text_df['id'], text_df['text']))
    data['embedding'] = np.load(config['embed_file'])
    data['pairs_df'] = pd.read_csv(config['pairs_file'])

    try:
        gpt_df = pd.read_csv(config['gpt_answer_file'])
        data['gpt_df'] = gpt_df
        data['use_gpt'] = True
    except:
        print("[警告] GPT答案文件读取失败，将只使用标签作为completion")
        data['use_gpt'] = False
        data['gpt_df'] = None

    return data

# ========== 模块2：构造 Prompt 样本 ==========
def build_prompt_dataset(data):
    print("构造 Prompt 样本...")
    results = []
    pairs_df = data['pairs_df']
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        id_a, id_b, label = int(row['id_a']), int(row['id_b']), int(row['label'])

        text_a = data['id2text'].get(id_a, f"节点{id_a}")
        text_b = data['id2text'].get(id_b, f"节点{id_b}")

        nei_a = data['onehop_1'].get(id_a, [])[:5]
        nei_b = data['onehop_2'].get(id_b, [])[:5]
        nei_text_a = [data['id2text'].get(n, f"邻居{n}") for n in nei_a]
        nei_text_b = [data['id2text'].get(n, f"邻居{n}") for n in nei_b]

        topo_feat_a = data['topo_1'].get(id_a, [])
        topo_feat_b = data['topo_2'].get(id_b, [])

        emb_a = data['embedding'][id_a].reshape(1, -1)
        emb_b = data['embedding'][id_b].reshape(1, -1)
        sim = cosine_similarity(emb_a, emb_b)[0][0]

        prompt = f"""
                源节点（图A）描述：{text_a}
                邻居节点：{', '.join(nei_text_a)}
                拓扑摘要：{topo_feat_a}

                目标节点（图B）描述：{text_b}
                邻居节点：{', '.join(nei_text_b)}
                拓扑摘要：{topo_feat_b}

                语义相似度：{sim:.3f}
                请判断这两个节点是否为同一实体？回答“是”或“否”，并说明理由。
                """
                .strip()

        if data['use_gpt'] and idx < len(data['gpt_df']):
            completion = str(data['gpt_df'].iloc[idx]['answer']).strip()
        else:
            completion = "是" if label == 1 else "否"

        results.append({
            "prompt": prompt,
            "completion": completion
        })

    return results

# ========== 模块3：保存为 JSONL ==========
def save_jsonl(dataset, output_path):
    print("[阶段1-3] 写入 JSONL 文件...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 保存完成：{len(dataset)} 条 → {output_path}")


if __name__ == "__main__":
    data = load_all_inputs(config)
    dataset = build_prompt_dataset(data)
    save_jsonl(dataset, output_file)
