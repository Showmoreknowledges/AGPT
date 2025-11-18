"""
Rerank GTC candidates with an LLM using node texts.

Usage example:
    python rerank_with_llm.py \
      --candidates_path results/gtc_candidates.json \
      --texts_path data/douban_node_texts.json \
      --src_layer layer1 \
      --tgt_layer layer2 \
      --provider openai \
      --model gpt-4.1-mini \
      --output_json results/reranked_mapping_gpt4.1.json
"""

import argparse
import json
import os
import os.path as osp
import re
from typing import Dict, List, Tuple

from tqdm import tqdm


# ========== 1. 这里是你要改的 Prompt 模板 ==========

def build_prompt(src_text: str, tgt_text: str):
    """
    构造给 LLM 的完整 prompt。
    👉 如果你想改模板，只需要改这个函数里的内容。
    """
    template = f"""
You are helping to align nodes between two related graphs.

Each node is described with structural and cross-layer information.

[Source node description]
{src_text}

[Target node description]
{tgt_text}

Task:
On a scale from 0 to 10, how likely is it that the source node
and the target node correspond to the same real-world entity?

Respond with only a single number between 0 and 10.
"""
    return template.strip()



def call_llm(provider: str, model: str, prompt: str) -> str:
    """
    根据 provider 调用不同的后端。
    需要你在环境变量中设置相应的 API_KEY。
    """
    provider = provider.lower()

    if provider == "openai":
        # 使用 OpenAI 官方接口
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    elif provider == "openrouter":
        # 通过 OpenRouter 调用第三方模型（如 deepseek、llama 等）
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        resp = client.chat.completions.create(
            model=model,  # 例如 "deepseek/deepseek-chat"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    elif provider == "deepseek":
        # 直接调用 DeepSeek 官方接口
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        resp = client.chat.completions.create(
            model=model,  # 例如 "deepseek-chat"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    elif provider == "hf_local":
        # 本地 HuggingFace 模型（如本地 LLaMA）
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        pipe = pipeline(
            "text-generation",
            model=model,            # 本地或缓存路径
            device_map="auto",
        )
        out = pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        return out.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")


def parse_score_from_response(text: str):
    """
    从 LLM 返回的文本中解析出一个 0-10 的分数。
    简单做法：提取第一个浮点/整数。
    """
    # 找到第一个类似 7, 7.5, 10, 0.0 这样的数字
    match = re.search(r"[-+]?\d+(\.\d+)?", text)
    if not match:
        # 找不到数字时，保守给一个中间值
        return 5.0
    try:
        score = float(match.group(0))
    except ValueError:
        return 5.0
    # 裁剪到 [0, 10]
    score = max(0.0, min(10.0, score))
    return score



def load_candidates(path: str):
    """
    期望 JSON 格式:
    {
        "0": [3, 5, 10],
        "1": [4, 7],
        ...
    }
    即: 源节点 -> 候选目标节点列表
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[int, List[int]] = {}
    for k, v_list in data.items():
        u = int(k)
        mapping[u] = [int(x) for x in v_list]
    return mapping


def load_node_texts(path: str):
    """
    期望 JSON 格式与你 generate_text 脚本输出一致:
    {
        "layer1": {
            "0": "text ...",
            "1": "text ...",
            ...
        },
        "layer2": {
            "0": "text ...",
            ...
        }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    texts: Dict[str, Dict[int, str]] = {}
    for layer_name, node_map in raw.items():
        texts[layer_name] = {}
        for node_id_str, text in node_map.items():
            texts[layer_name][int(node_id_str)] = text
    return texts



def rerank_candidates(
    candidates: Dict[int, List[int]],
    texts: Dict[str, Dict[int, str]],
    src_layer: str,
    tgt_layer: str,
    provider: str,
    model: str,
    max_candidates_per_src: int = 0,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]]]:
    """
    对每个源节点 u 和它的候选列表 v，用 LLM 打分并排序。

    返回:
        reranked_mapping: {u: [v1, v2, ...] 按得分从高到低}
        scores: {u: {v: score}}
    """
    src_texts = texts[src_layer]
    tgt_texts = texts[tgt_layer]

    reranked_mapping: Dict[int, List[int]] = {}
    scores: Dict[int, Dict[int, float]] = {}

    all_src_nodes = sorted(candidates.keys())

    for u in tqdm(all_src_nodes, desc="LLM reranking"):
        cand_vs = candidates[u]
        if max_candidates_per_src > 0:
            cand_vs = cand_vs[:max_candidates_per_src]

        src_text = src_texts.get(u, "")
        if not src_text:
            # 没有文本时，给一个占位描述
            src_text = f"Node {u} in layer {src_layer}."

        scores[u] = {}
        for v in cand_vs:
            tgt_text = tgt_texts.get(v, "")
            if not tgt_text:
                tgt_text = f"Node {v} in layer {tgt_layer}."

            prompt = build_prompt(src_text, tgt_text)
            try:
                response = call_llm(provider, model, prompt)
                score = parse_score_from_response(response)
            except Exception as e:
                print(f"[warn] LLM call failed for u={u}, v={v}: {e}")
                # 失败时给一个中等分
                score = 5.0

            scores[u][v] = score

        # 根据得分排序
        sorted_vs = sorted(scores[u].keys(), key=lambda x: scores[u][x], reverse=True)
        reranked_mapping[u] = sorted_vs

    return reranked_mapping, scores