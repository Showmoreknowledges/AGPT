from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

import torch
from transformers import Trainer
import wandb

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llamafactory.extras.misc import count_parameters
from llamafactory.hparams.parser import get_train_args

from dataset.align_dataset import AlignDataset
from dataset.linkgpt_dataset import LinkGPTDataCollator
from model.linkgpt_model import (
    freeze_all_parameters,
    get_model_and_tokenizer,
    save_lora_model,
    unfreeze_graph_related_modules,
    unfreeze_lora_adapter,
)
from utils import basics


def _maybe_extract_text(record, field: Optional[str]) -> Optional[str]:
    if record is None or field is None:
        return None
    if isinstance(record, dict):
        value = record.get(field)
        if value is not None:
            return value
    if hasattr(record, field):
        value = getattr(record, field)
        if value is not None:
            return value
    return None


def build_gnid2text(dataset_for_lm, graph_label: str) -> Dict[int, str]:
    if graph_label not in {"A", "B"}:
        raise ValueError(f"Unsupported graph label: {graph_label}")

    attr_name = f"gnid2text_{graph_label}"
    mapping = getattr(dataset_for_lm, attr_name, None)
    if mapping:
        return {int(k): v for k, v in mapping.items()}

    generic_mapping = getattr(dataset_for_lm, "gnid2text", None)
    if generic_mapping:
        return {int(k): v for k, v in generic_mapping.items()}

    data_attr = "data_list_A" if graph_label == "A" else "data_list_B"
    data_list = getattr(dataset_for_lm, data_attr, None)
    if data_list is None:
        data_list = getattr(dataset_for_lm, "data_list", None)
    if data_list is None:
        raise ValueError(
            "dataset_for_lm is missing data_list for graph {}".format(graph_label)
        )

    text_field = getattr(dataset_for_lm, "text_field", None)
    longer_text_field = getattr(dataset_for_lm, "longer_text_field", None)

    gnid2text: Dict[int, str] = {}
    for idx, record in enumerate(data_list):
        text_val = _maybe_extract_text(record, text_field)
        if text_val is None:
            text_val = _maybe_extract_text(record, longer_text_field)
        if text_val is None:
            raise ValueError(
                f"Unable to locate textual field for node {idx} in graph {graph_label}"
            )
        gnid2text[idx] = text_val
    return gnid2text


def load_text_embeddings(folder_path: str, method: str, max_hop: int, device: str):
    embedding_list = []
    for hop in range(max_hop + 1):
        if hop == 0:
            file_name = f"text_emb_{method}.pt"
        else:
            file_name = f"text_emb_{method}_{hop}hop.pt"
        emb_path = os.path.join(folder_path, file_name)
        embedding_list.append(torch.load(emb_path, map_location=device))
    return embedding_list


def main() -> None:
    basics.set_seeds(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_embedding_method", required=True)
    parser.add_argument("--text_embedding_folder_path_A", required=True)
    parser.add_argument("--text_embedding_folder_path_B", required=True)
    parser.add_argument("--max_hop", type=int, default=0)
    parser.add_argument("--dataset_for_lm_path_A", required=True)
    parser.add_argument("--dataset_for_lm_path_B", required=True)
    parser.add_argument("--align_dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", required=True)

    parser.add_argument(
        "--device_setting",
        default=None,
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    )

    parser.add_argument("--wandb_key", default=None, type=str)
    parser.add_argument("--project_name", default="LinkGPT-align", type=str)
    parser.add_argument("--run_name", default="alignment_run", type=str)

    parser.add_argument("--finetuning_type", default="lora")
    parser.add_argument("--per_device_train_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--save_steps", default=10000, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--num_train_epochs_stage1", default=1, type=int)
    parser.add_argument("--num_train_epochs_stage2", default=1, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report_to", default=None)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--dataloader_prefetch_factor", default=8, type=int)

    parser.add_argument("--lora_target", default="q_proj,v_proj")
    parser.add_argument("--lora_alpha", default=16, type=float)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)

    parser.add_argument("--freeze_graph_related_modules_in_stage2", action="store_true")
    parser.add_argument("--node_proj_num_layers", default=1, type=int)

    args = parser.parse_args()
    print(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_key:
        if args.wandb_key == "None":
            args.report_to = None
        else:
            wandb.login(key=args.wandb_key)
            os.environ["WANDB_PROJECT"] = args.project_name
            os.environ["WANDB_RUN_NAME"] = args.run_name

    if args.device_setting is None:
        device = basics.get_device()
        print(f"No device setting is provided. Using {device}", flush=True)
    else:
        device = args.device_setting

    text_emb_list_A = load_text_embeddings(
        args.text_embedding_folder_path_A,
        args.text_embedding_method,
        args.max_hop,
        device,
    )
    text_emb_list_B = load_text_embeddings(
        args.text_embedding_folder_path_B,
        args.text_embedding_method,
        args.max_hop,
        device,
    )

    dataset_for_lm_A = basics.load_pickle(args.dataset_for_lm_path_A)
    dataset_for_lm_B = basics.load_pickle(args.dataset_for_lm_path_B)

    gnid2text_A = build_gnid2text(dataset_for_lm_A, "A")
    gnid2text_B = build_gnid2text(dataset_for_lm_B, "B")

    align_dataset: AlignDataset = basics.load_pickle(args.align_dataset_path)
    align_dataset.gnid2text_A = gnid2text_A
    align_dataset.gnid2text_B = gnid2text_B
    align_dataset.config.node_encoding_max_hop = args.max_hop
    align_dataset.config.return_tokenized = True

    num_node_types = getattr(align_dataset, "num_node_types", 1)
    num_relation_types = getattr(align_dataset, "num_relation_types", 1)

    hf_args_dict = {
        "do_train": True,
        "stage": "pt",
        "lora_target": args.lora_target,
        "overwrite_output_dir": True,
        "resize_vocab": True,
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "dataset": args.dataset_name,
        "finetuning_type": args.finetuning_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs_stage1,
        "fp16": args.fp16,
        "lora_rank": args.lora_rank,
        "lora_dropout": args.lora_dropout,
        "lora_alpha": args.lora_alpha,
        "report_to": args.report_to,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "dataloader_num_workers": args.dataloader_num_workers,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
    }
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
        hf_args_dict
    )

    model, tokenizer = get_model_and_tokenizer(
        model_args,
        finetuning_args,
        node_encoding_list_A=text_emb_list_A,
        lpformer_model=None,
        is_trainable=True,
        device=device,
        apply_lora=True,
        node_proj_num_layers=args.node_proj_num_layers,
        num_node_types=num_node_types,
        num_relation_types=num_relation_types,
        node_encoding_list_B=text_emb_list_B,
    )

    align_dataset.tokenizer = tokenizer
    align_dataset.set_lengths()

    linkgpt_data_collator = LinkGPTDataCollator(tokenizer)

    # Stage 1: train graph-related parameters
    print("Stage 1: Train graph encoder parameters", flush=True)
    freeze_all_parameters(model)
    unfreeze_graph_related_modules(model)
    trainable_param, total_param = count_parameters(model)
    print(f"Num of trainable params: {trainable_param}", flush=True)
    print(f"Total num of params: {total_param}", flush=True)
    print(f"ratio: {trainable_param / total_param * 100: .3}%", flush=True)

    training_args.num_train_epochs = args.num_train_epochs_stage1
    stage1_dataset = align_dataset
    stage1_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=linkgpt_data_collator,
        train_dataset=stage1_dataset,
    )
    stage1_trainer.train()
    save_lora_model(stage1_trainer.model, os.path.join(args.output_dir, "stage1"))

    # Stage 2: jointly tune LLM and graph modules
    print("Stage 2: Jointly tune the LLM and graph modules", flush=True)
    freeze_all_parameters(model)
    if not args.freeze_graph_related_modules_in_stage2:
        unfreeze_graph_related_modules(model)
    unfreeze_lora_adapter(model)
    trainable_param, total_param = count_parameters(model)
    print(f"Num of trainable params: {trainable_param}")
    print(f"Total num of params: {total_param}")
    print(f"ratio: {trainable_param / total_param * 100: .3}%")

    training_args.num_train_epochs = args.num_train_epochs_stage2
    stage2_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=linkgpt_data_collator,
        train_dataset=align_dataset,
    )
    stage2_trainer.train()
    save_lora_model(stage2_trainer.model, os.path.join(args.output_dir, "stage2"))


if __name__ == "__main__":
    main()
