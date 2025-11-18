import os
import os.path as osp
import re
import numpy as np
import pandas as pd
import argparse
import requests
from tqdm import tqdm

from utils.llm_wrapper import get_answer


# === CLI Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description='Query LLMs for multilayer graph prompts')
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, help='Model name or local model path')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'openrouter', 'hf_local'])
    parser.add_argument('--without_neigh', action='store_true')
    parser.add_argument('--response_base', type=str, default=None, help='Base directory for response files; defaults to <npz_dir>/response')
    return parser.parse_args()

# === Layer Extraction ===
def extract_layers(npz):
    return sorted(set(
    int(m.group()) for k in npz.files if k.startswith("x")
    if (m := re.search(r'\d+', k)) is not None
))

# === File Path Construction ===
def get_paths(prefix, layer, model, wo_neigh, response_base=None, npz_dir=None):
    dataset = f"{prefix}{layer}"
    if response_base is not None:
        base = osp.join(response_base, dataset)
    elif npz_dir is not None:
        base = osp.join(npz_dir, 'response', dataset)
    else:
        # fallback to current working dir
        base = osp.join('.', 'response', dataset)
    qfile = 'question_wo_neigh.csv' if wo_neigh else 'question.csv'
    afile = f'answer_{osp.basename(model)}_wo_neigh.csv' if wo_neigh else f'answer_{osp.basename(model)}.csv'
    return osp.join(base, qfile), osp.join(base, afile)

# === HuggingFace Local Model Loader ===
def load_local_model(model_path):
    # keep a small local cache to avoid repeated loads
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    global _local_cache
    try:
        _local_cache
    except NameError:
        _local_cache = {}
    if model_path in _local_cache:
        return _local_cache[model_path]
    print(f"🔄 Loading local HF model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    _local_cache[model_path] = pipe
    return pipe

# === Unified LLM Query Function ===
def query_llm(question, model, provider):
    # wrap the centralized llm wrapper; errors are returned as text starting with 'Error:'
    return get_answer(question, model, provider)

# === Per-layer Prompt Filling ===
def process_layer(prefix, layer, args, response_base=None, npz_dir=None):
    qpath, apath = get_paths(prefix, layer, args.model, args.without_neigh, response_base=response_base, npz_dir=npz_dir)
    if not osp.exists(qpath) or not osp.exists(apath):
        print(f"⚠️ Missing files for {prefix}{layer}, skipping...")
        return

    print(f"💬 Processing {prefix}{layer} with {args.model} via {args.provider}")
    q_df = pd.read_csv(qpath)
    a_df = pd.read_csv(apath, sep='\t')

    for i in tqdm(range(len(a_df))):
        if a_df.iloc[i]['answer'] == "Error":
            prompt = q_df.iloc[i]['question']
            answer = query_llm(prompt, args.model, args.provider)
            a_df.at[i, 'answer'] = answer

    a_df.to_csv(apath, sep='\t', index=False)
    print(f"✅ Saved: {apath}")

# === Entry point ===
def main():
    args = parse_args()
    prefix = osp.splitext(osp.basename(args.npz_path))[0]
    npz = np.load(args.npz_path)
    layers = extract_layers(npz)

    npz_dir = osp.dirname(args.npz_path)
    response_base = args.response_base

    for layer in layers:
        process_layer(prefix, layer, args, response_base=response_base, npz_dir=npz_dir)

if __name__ == "__main__":
    main()