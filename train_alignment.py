import argparse
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback, TrainingArguments

from Mul_dataset import AlignmentDataset


class AlignmentScoringModel(nn.Module):
    """
    基于节点嵌入与结构特征的网络对齐打分模型。
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1, use_ppr: bool = True) -> None:
        super().__init__()
        self.use_ppr = use_ppr
        feature_dim = embed_dim * 4 + (1 if use_ppr else 0)

        hidden_mid = max(hidden_dim // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid, 1),
        )

    def forward(
        self,
        u_emb: torch.Tensor,
        v_emb: torch.Tensor,
        ppr_score: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pair_features = [u_emb, v_emb, torch.abs(u_emb - v_emb), u_emb * v_emb]
        if self.use_ppr and ppr_score is not None:
            if ppr_score.dim() == 1:
                ppr_score = ppr_score.unsqueeze(-1)
            pair_features.append(ppr_score.float())

        pair_feature = torch.cat(pair_features, dim=-1)
        logits = self.mlp(pair_feature).squeeze(-1)

        output: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            labels = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            output["loss"] = loss
            output["labels"] = labels
        return output


class NegativeResampleCallback(TrainerCallback):
    """
    每个 epoch 开始时重新采样负样本，增强训练的鲁棒性。
    """

    def __init__(self, dataset: AlignmentDataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:  # type: ignore[override]
        self.dataset.resample_negatives()
        return control


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = 1.0 / (1.0 + np.exp(-logits))

    preds = (probs >= 0.5).astype(np.float32)
    labels = labels.astype(np.float32)

    accuracy = float((preds == labels).mean())
    tp = float(np.sum((preds == 1.0) * (labels == 1.0)))
    fp = float(np.sum((preds == 1.0) * (labels == 0.0)))
    fn = float(np.sum((preds == 0.0) * (labels == 1.0)))

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="网络对齐任务训练脚本")
    parser.add_argument("--data_path", type=str, default="lpformer_data.pt", help="包含节点嵌入/PPR等信息的 torch 文件")
    parser.add_argument("--train_pairs_path", type=str, default="train_pairs_merged.pt", help="训练锚点对")
    parser.add_argument("--test_pairs_path", type=str, default="test_pairs_merged.pt", help="测试锚点对")
    parser.add_argument("--gnid2text_path", type=str, default="gnid2text.json", help="节点文本映射，可选")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/alignment", help="模型与日志输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=512, help="每设备 batch size")
    parser.add_argument("--neg_ratio", type=int, default=4, help="每个正样本生成的负样本数量")
    parser.add_argument("--eval_neg_ratio", type=int, default=None, help="评估阶段负样本比例，默认等于 neg_ratio")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.1, help="MLP dropout 概率")
    parser.add_argument("--hidden_dim", type=int, default=512, help="MLP 隐藏维度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_steps", type=int, default=50, help="日志打印步数")
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_node_embeddings(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    for key in ("x", "node_embeddings", "embeddings"):
        if key in data:
            emb = data[key]
            if not isinstance(emb, torch.Tensor):
                raise TypeError(f"{key} should be a torch.Tensor, got {type(emb)}")
            return emb.detach().cpu().float()
    raise KeyError("无法在数据文件中找到节点嵌入 (x/node_embeddings/embeddings)")


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data_bundle = torch.load(args.data_path, map_location="cpu")
    node_embeddings = load_node_embeddings(data_bundle)
    ppr_train = data_bundle.get("ppr")
    ppr_test = data_bundle.get("ppr_test", ppr_train)

    train_pairs = torch.load(args.train_pairs_path, map_location="cpu")
    test_pairs = torch.load(args.test_pairs_path, map_location="cpu")

    if os.path.exists(args.gnid2text_path):
        with open(args.gnid2text_path, "r", encoding="utf-8") as f:
            gnid2text = json.load(f)
        print(f"📄 成功加载 gnid2text 文本映射，共 {len(gnid2text)} 条")
    else:
        gnid2text = None
        print("ℹ️ 未找到 gnid2text 文本映射，训练仅基于结构与嵌入特征")

    eval_neg_ratio = args.neg_ratio if args.eval_neg_ratio is None else max(args.eval_neg_ratio, 0)

    train_dataset = AlignmentDataset(
        node_embeddings=node_embeddings,
        pairs=train_pairs,
        ppr=ppr_train,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        shuffle=True,
    )
    eval_dataset = AlignmentDataset(
        node_embeddings=node_embeddings,
        pairs=test_pairs,
        ppr=ppr_test,
        neg_ratio=eval_neg_ratio,
        seed=args.seed,
        shuffle=False,
    )

    model = AlignmentScoringModel(
        embed_dim=node_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_ppr=train_dataset.ppr_lookup is not None,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=args.log_steps,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if train_dataset.neg_ratio > 0:
        trainer.add_callback(NegativeResampleCallback(train_dataset))

    print("🚀 开始训练网络对齐模型...")
    trainer.train()
    metrics = trainer.evaluate()
    print("📊 评估指标:", metrics)

    model_dir = os.path.join(args.output_dir, "alignment_model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embed_dim": int(node_embeddings.shape[1]),
            "use_ppr": bool(train_dataset.ppr_lookup is not None),
        },
        os.path.join(model_dir, "pytorch_model.bin"),
    )
    with open(os.path.join(model_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"✅ 模型与配置已保存在 {model_dir}")


if __name__ == "__main__":
    main()
