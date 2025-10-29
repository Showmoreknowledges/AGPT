import sys
import itertools
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
import argparse
import json
import pickle

import torch
from torch.utils import data
from tqdm import tqdm
from transformers import AutoConfig, BertTokenizer, BertModel

# === 项目路径修复 ===
project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# === 导入 LinkGPT 模块 ===
from linkgpt.text_graph_pretraining.graph_text_dataset import CGTPDataset
from linkgpt.text_graph_pretraining.graph_text_model import CGTPModel
from linkgpt.utils import basics
from Mul_dataset import TAGDatasetForLM


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    model.train()
    total_loss = 0.0
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        center_input, neighbor_input, neighbor_mask, all_input, all_mask = batch
        for part in [center_input, neighbor_input, all_input]:
            for k in part.keys():
                part[k] = part[k].to(device)
        neighbor_mask = neighbor_mask.to(device)
        all_mask = all_mask.to(device)
        
        loss = model(center_input, neighbor_input, neighbor_mask, all_input, all_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step == "batch":
            lr_scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f"平均训练损失: {avg_loss:.4f}")
    return avg_loss


# ===========================================================
# 🚀 主程序
# ===========================================================
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_for_lm_path', required=True, type=str, help="dataset_for_lm.pkl 路径")
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--ckpt_save_path', required=True, type=str, help="模型权重保存路径")
    parser.add_argument('--data_dir', type=str, default=None, help="包含 merged_features.pt 与 gnid2text.json 的目录")

    parser.add_argument('-t', '--text_encoder_name', default='bert-base-uncased')
    parser.add_argument('-g', '--graph_encoder_name', default='graphformer')
    parser.add_argument('--num_neighbor', default=5, type=int)
    parser.add_argument('--max_text_length', default=64, type=int)
    parser.add_argument('--text_encoder_num_layers_to_train', default=2, type=int)
    parser.add_argument('--text_emb_dim', default=768, type=int)
    parser.add_argument('--graph_embed_dim', default=768, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--head_lr', default=1e-3, type=float)
    parser.add_argument('--graph_encoder_lr', default=1e-4, type=float)
    parser.add_argument('--text_encoder_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--lr_scheduler_patience', default=3, type=int)
    parser.add_argument('--lr_scheduler_factor', default=0.8, type=float)
    parser.add_argument('--num_epochs', default=5, type=int)
    
    args = parser.parse_args()

    # =======================================================
    # 1️⃣ 加载 dataset_for_lm.pkl
    # =======================================================
    with open(args.dataset_for_lm_path, 'rb') as f:
        dataset_for_lm = pickle.load(f)
    print(f" 已加载 dataset_for_lm.pkl ({len(dataset_for_lm)} 个节点)")

    # 自动检测数据目录
    data_dir = args.data_dir or os.path.dirname(args.dataset_for_lm_path)

    # =======================================================
    # 2️⃣ 自动加载外部特征矩阵与文本映射
    # =======================================================
    merged_features_path = os.path.join(data_dir, "merged_features.pt")
    gnid2text_path = os.path.join(data_dir, "gnid2text.json")

    if os.path.exists(merged_features_path):
        dataset_for_lm.features = torch.load(merged_features_path)
        print(f" 已加载特征矩阵: {dataset_for_lm.features.shape}")
    else:
        dataset_for_lm.features = None
        print("⚠️ 未检测到 merged_features.pt，将仅使用文本特征")

    if os.path.exists(gnid2text_path):
        with open(gnid2text_path, "r", encoding="utf-8") as f:
            dataset_for_lm.gnid2text = json.load(f)
        print(f" 已加载 gnid2text.json ({len(dataset_for_lm.gnid2text)} 条文本)")
    else:
        dataset_for_lm.gnid2text = None
        print("⚠️ 未检测到 gnid2text.json，将仅使用数值特征")

    # 3️⃣ 自动适配 text_field（原版 LinkGPT 所需）
    dataset_for_lm.text_field = "text" if dataset_for_lm.gnid2text is not None else "feature"

    # 4️⃣ 构造 CGTPDataset
    get_text = (
        (lambda x: x.get("text", ""))
        if dataset_for_lm.gnid2text is not None
        else (lambda x: "")
    )

    train_dataset = CGTPDataset(
        get_text,
        args.num_neighbor,
        dataset_for_lm,
        args.text_encoder_name,
        args.max_text_length,
    )

    device = basics.get_device()
    cgtp_model = CGTPModel().to(device)
    total_params = sum(p.numel() for p in cgtp_model.parameters())
    trainable_params = sum(p.numel() for p in cgtp_model.parameters() if p.requires_grad)
    print(f"# total parameters: {total_params:,}\n# trainable parameters: {trainable_params:,}")

    # =======================================================
    # 5️⃣ 优化器与学习率计划
    # =======================================================
    params = [
        {"params": cgtp_model.graph_encoder.parameters(), "lr": args.graph_encoder_lr},
        {"params": cgtp_model.text_encoder.parameters(), "lr": args.text_encoder_lr},
        {
            "params": itertools.chain(
                cgtp_model.graph_proj.parameters(), cgtp_model.text_proj.parameters()
            ),
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.lr_scheduler_patience,
        factor=args.lr_scheduler_factor,
    )

    # =======================================================
    # 6️⃣ 训练循环
    # =======================================================
    for epoch in range(args.num_epochs):
        print(f"\n====================== epoch: {epoch + 1} ======================")
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        avg_loss = train_epoch(cgtp_model, train_loader, optimizer, lr_scheduler, "epoch", device)

    # =======================================================
    # 7️⃣ 保存模型
    # =======================================================
    os.makedirs(os.path.dirname(args.ckpt_save_path), exist_ok=True)
    torch.save(cgtp_model.state_dict(), args.ckpt_save_path)
    print(f"\n 模型已保存至: {args.ckpt_save_path}")


if __name__ == '__main__':
    main()

