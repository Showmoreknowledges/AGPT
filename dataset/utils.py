from typing import List, Tuple, Dict, Callable, Set, Iterable, Optional
from dataclasses import dataclass
import random
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, trange
import torch_sparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100

NODE_START_TOKEN = '<node_start>'
NODE_TOKEN = '<node>'
PAIRWISE_START_TOKEN = '<pairwise_start>'
PAIRWISE_TOKEN = '<pairwise>'

LINKGPT_SPECIAL_TOKENS = [
    NODE_START_TOKEN,
    NODE_TOKEN,
    PAIRWISE_START_TOKEN,
    PAIRWISE_TOKEN,
]

def sample_neg_tgt(num_neg_tgt: int, pos_tgt_set: Set[int], total_node_num: int, candidate_pool: Optional[Iterable[int]] = None):
    """Sample negative targets that respect optional candidate pools.

    Args:
        num_neg_tgt: Number of negatives to sample.
        pos_tgt_set: Set of positive target node ids to avoid.
        total_node_num: Total number of nodes in the (merged) graph. Used when
            ``candidate_pool`` is not provided.
        candidate_pool: Optional iterable of node ids that negatives should be
            drawn from. When provided, negatives are sampled exclusively from
            this pool after removing positives.

    Returns:
        A list of sampled negative node ids. If the candidate pool does not
        contain enough nodes, as many unique negatives as possible will be
        returned (potentially fewer than ``num_neg_tgt``).
    """

    if num_neg_tgt <= 0:
        return []

    if candidate_pool is not None:
        candidate_ls = list(dict.fromkeys(candidate_pool))
        candidate_ls = [nid for nid in candidate_ls if nid not in pos_tgt_set]
        if not candidate_ls:
            return []
        if len(candidate_ls) < num_neg_tgt:
            return random.sample(candidate_ls, len(candidate_ls))
        return random.sample(candidate_ls, num_neg_tgt)
    
    while True:
        neg_tgt_list = random.sample(range(total_node_num), num_neg_tgt)
        if not any(neg_tgt in pos_tgt_set for neg_tgt in neg_tgt_list):
            return neg_tgt_list

def get_text_with_encoding_token(center, config, gnid2text, do_pairwise=True):
    """
    Get the text with the encoding tokens for the prompt
    """
    
    ablate_node_encoding = getattr(config, 'ablate_node_encoding', True)
    ablate_pairwise_encoding = getattr(config, 'ablate_pairwise_encoding', True)
    
    node_prefix = "" if ablate_node_encoding else NODE_START_TOKEN + NODE_TOKEN * (config.node_encoding_max_hop + 1)
    pairwise_prefix = "" if (ablate_pairwise_encoding or not do_pairwise) else PAIRWISE_START_TOKEN + PAIRWISE_TOKEN
    if node_prefix + pairwise_prefix == "":
        return 'text: ' + gnid2text[center] + '\n'
    else:
        return node_prefix + pairwise_prefix + '\n' + 'text: ' + gnid2text[center] + '\n'