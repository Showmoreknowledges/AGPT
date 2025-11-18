from typing import List, Tuple, Dict, Callable, Set, Optional
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
import copy

from utils import basics
from .utils import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, \
    PAIRWISE_TOKEN, LINKGPT_SPECIAL_TOKENS, IGNORE_INDEX, get_text_with_encoding_token, sample_neg_tgt

# for Link Prediction task, using Yes/No (YN) prompt

@dataclass
class YNTargetData:
    tgt_node: int
    label: int
    tgt_type: Optional[int] = None
    relation_type: Optional[int] = None

@dataclass
class YNData:
    src_node: int
    tgt_list: List[YNTargetData]

@dataclass
class YNDatasetConfig:
    num_tgt_per_prompt: int = 4 # number of target nodes per prompt
    num_neg_per_pos: float = 1 # number of negative target nodes for each positive target node
    
    # The following four are for the prompt generation. The wording can be modified to fit different tasks.
    task_desc: str = "Determine whether there is a link between the source node and the candidate nodes.\n"
    source_node_intro: str = "Source node:\n"
    candidate_target_node_intro: str = "Candidate target node:\n"
    connection_question: str = "Is this connected to the source node?\n"
    
    ablate_pairwise_encoding: bool = False
    ablate_node_encoding: bool = False
    node_encoding_max_hop: int = 0 
    # the maximum hop to encode the node
    # if node_encoding_max_hop == n, then the node encoding will contain n+1 tokens, i.e., 
    # (itself, avg. of its 1-hop neighbors, avg. of its 2-hop neighbors, ..., avg. of its n-hop neighbors)
    # However, in this paper, node_encoding_max_hop is fixed to be 0
    
    learn_text: bool = False # whether to calculate the loss for the text part
    learn_yn: bool = True # whether to calculate the loss for the Yes/No part (obviously, this must be True)
    learn_all: bool = False # whether to calculate the loss for all tokens (if True, learn_text and learn_yn will be ignored. And the training would be ordinary FT, instead of supervised FT.)
    
    # The following two are mainly for debugging
    generate_at_initialization: bool = True # whether to automatically generate the YNData list at initialization
    return_tokenized: bool = True # whether to return tokenized data in __getitem__
    
    # Optional heterogeneous metadata
    node_id2type: Optional[Dict[int, int]] = None
    type2node_ids: Optional[Dict[int, List[int]]] = None
    relation_type_map: Optional[Dict[Tuple[int, int], int]] = None
    default_node_type: int = 0
    default_relation_type: int = 0

class YNDataset(Dataset):
    def __init__(self, dgl_graph, gnid2text: Dict[int, str], config: YNDatasetConfig, tokenizer):
        self.dgl_graph = dgl_graph
        self.gnid2text = gnid2text
        self.config = config
        self.num_nodes = dgl_graph.num_nodes()
        self.tokenizer = tokenizer
        self.lengths = None
        self.column_names = ['length'] # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
        
        self.node_id2type = dict(config.node_id2type) if config.node_id2type is not None else {}
        if config.type2node_ids is not None:
            self.type2node_ids = {k: list(v) for k, v in config.type2node_ids.items()}
            self.type2node_ids = {k: list (v) for k, v in config.type2node_ids.items()}
        elif self.node_id2type:
            type2node_ids: Dict[int, List[int]] = {}
            for nid, type_id in self.node_id2type.items():
                type2node_ids.setdefault(type_id, []).append(nid)
            self.type2node_ids = type2node_ids
        else:
            self.type2node_ids = None
        self.default_node_type = getattr(config, 'default_node_type', 0)

        self.relation_type_map = dict(config.relation_type_map) if config.relation_type_map is not None else {}
        self.default_relation_type = getattr(config, 'default_relation_type', 0)
        self.num_node_types = max(self.node_id2type.values(), default=-1) + 1 if self.node_id2type else 1
        self.num_relation_types = max(self.relation_type_map.values(), default=-1) + 1 if self.relation_type_map else 1

        # keep config references in sync so that downstream components can reuse the metadata
        self.config.node_id2type = self.node_id2type if self.node_id2type else None
        self.config.type2node_ids = self.type2node_ids if self.type2node_ids else None
        self.config.relation_type_map = self.relation_type_map

        self.yn_data_list = None
        if self.config.generate_at_initialization:
            self.generate_yn_data_list()
    
    def _lookup_node_type(self, node_id: int) -> Optional[int]:
        if not self.node_id2type:
            return None
        return self.node_id2type.get(node_id, self.default_node_type)

    def _ensure_relation_type(self, src_node: int, tgt_node: int) -> Optional[int]:
        if not self.node_id2type:
            return None
        src_type = self._lookup_node_type(src_node)
        tgt_type = self._lookup_node_type(tgt_node)
        if src_type is None or tgt_type is None:
            return None
        key = (src_type, tgt_type)
        if key not in self.relation_type_map:
            self.relation_type_map[key] = len(self.relation_type_map)
        relation_id = self.relation_type_map[key]
        self.num_relation_types = max(self.num_relation_types, relation_id + 1)
        return relation_id

    def _candidate_pool_for_src(self, src: int, pos_tgt_id_ls: List[int]) -> Optional[List[int]]:
        if not self.type2node_ids or not self.node_id2type:
            return None
        candidate_types = {self._lookup_node_type(tgt) for tgt in pos_tgt_id_ls if tgt in self.node_id2type}
        if not candidate_types:
            src_type = self._lookup_node_type(src)
            if src_type is None:
                return None
            candidate_types = {ctype for ctype in self.type2node_ids.keys() if ctype != src_type}
        candidate_pool: List[int] = []
        for ctype in candidate_types:
            candidate_pool.extend(self.type2node_ids.get(ctype, []))
        return candidate_pool if candidate_pool else None

    def _build_target_data(self, src: int, tgt: int, label: int) -> YNTargetData:
        tgt_type = self._lookup_node_type(tgt)
        relation_type = self._ensure_relation_type(src, tgt)
        if relation_type is None:
            relation_type = self.default_relation_type
        return YNTargetData(tgt_node=tgt, label=label, tgt_type=tgt_type, relation_type=relation_type)
    
    def generate_yn_data_list(self, num_src_nodes=None):
        yn_data_list = []
        
        config = self.config
        total_num = num_src_nodes if num_src_nodes is not None else self.num_nodes # for debug only
        for src in trange(total_num):
            pos_tgt_id_ls = self.dgl_graph.successors(src).cpu().tolist()
            neg_tgt_num = round(len(pos_tgt_id_ls) * config.num_neg_per_pos)
            
            candidate_pool = self._candidate_pool_for_src(src, pos_tgt_id_ls)
            neg_tgt_id_ls = sample_neg_tgt(
                num_neg_tgt=neg_tgt_num,
                pos_tgt_set=set(pos_tgt_id_ls),
                total_node_num=self.num_nodes,
                candidate_pool=candidate_pool,
            )

            pos_tgt_data_ls = [self._build_target_data(src, tgt, 1) for tgt in pos_tgt_id_ls]
            neg_tgt_data_ls = [self._build_target_data(src, tgt, 0) for tgt in neg_tgt_id_ls]
            
            all_tgt_data_ls = pos_tgt_data_ls + neg_tgt_data_ls
            random.shuffle(all_tgt_data_ls)
            
            for start in range(0, len(all_tgt_data_ls), config.num_tgt_per_prompt):
                end = min(len(all_tgt_data_ls), start + config.num_tgt_per_prompt)
                cur_tgt_data_ls = all_tgt_data_ls[start:end]
                yn_data = YNData(src_node=src, tgt_list=cur_tgt_data_ls)
                yn_data_list.append(yn_data)
        self.yn_data_list = yn_data_list
        self.num_node_types = max(self.num_node_types, max(self.node_id2type.values(), default=-1) + 1 if self.node_id2type else 1)
        self.num_relation_types = max(self.num_relation_types, len(self.relation_type_map) if self.relation_type_map else 1)
    
    def __len__(self):
        if self.yn_data_list is None:
            raise ValueError("YNData list not generated yet!")
        return len(self.yn_data_list)
    
    def __getitem__(self, index: int):
        if index == 'length':
            # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
            return self.lengths
        if self.yn_data_list is None:
            raise ValueError("YNData list not generated yet!")
        config = self.config
        yn_data = self.yn_data_list[index]
        prompt, graph_data = yn_data_to_prompt_and_graph_data(self.gnid2text, yn_data, self.config)
        if not config.return_tokenized:
            return prompt, graph_data
        tokenizer = self.tokenizer
        encoding = tokenizer(prompt, return_tensors='pt')
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        labels = generate_yn_labels(tokenizer, input_ids, learn_text=config.learn_text, learn_yn=config.learn_yn, learn_all=config.learn_all)
        return ({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }, graph_data)
    
    def set_lengths(self):
        # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
        old_config = self.config
        length_list = []
        self.config.return_tokenized = True
        for i in trange(self.__len__()):
            data = self.__getitem__(i)
            length = data[0]['input_ids'].shape[0]
            length_list.append(length)
            
        self.config = old_config
        self.lengths = length_list


def yn_data_to_prompt_and_graph_data(gnid2text: Dict[int, str], yn_data: YNData, config: YNDatasetConfig):
    graph_data = {
        'source_node': -1,
        'node_id_ls': [],
        'pairwise_target_id_ls': [],
        'node_type_ls': [],
        'pairwise_target_type_ls': [],
        'pairwise_relation_type_ls': [],
    }
    src = yn_data.src_node
    graph_data['source_node'] = src
    src_text = get_text_with_encoding_token(src, config, gnid2text, do_pairwise=False) + '\n'
    graph_data['node_id_ls'] += [(src, i) for i in range(config.node_encoding_max_hop + 1)]
    
    node_type_lookup = getattr(config, 'node_id2type', None)
    default_node_type = getattr(config, 'default_node_type', 0)
    default_relation_type = getattr(config, 'default_relation_type', 0)

    if node_type_lookup is not None:
        graph_data['node_type_ls'] += [node_type_lookup.get(src, default_node_type) for _ in range(config.node_encoding_max_hop + 1)]
    
    tgt_text_ls = []
    for tgt in yn_data.tgt_list:
        info_part = get_text_with_encoding_token(tgt.tgt_node, config, gnid2text, do_pairwise=True)
        if not config.ablate_node_encoding:
            graph_data['node_id_ls'] += [(tgt.tgt_node, i) for i in range(config.node_encoding_max_hop + 1)]
            if node_type_lookup is not None:
                graph_data['node_type_ls'] += [node_type_lookup.get(tgt.tgt_node, default_node_type) for _ in range(config.node_encoding_max_hop + 1)]
        if not config.ablate_pairwise_encoding:
            graph_data['pairwise_target_id_ls'].append(tgt.tgt_node)
            if tgt.tgt_type is not None:
                graph_data['pairwise_target_type_ls'].append(tgt.tgt_type)
            elif node_type_lookup is not None:
                graph_data['pairwise_target_type_ls'].append(node_type_lookup.get(tgt.tgt_node, default_node_type))
            else:
                graph_data['pairwise_target_type_ls'].append(default_node_type)
            if tgt.relation_type is not None:
                graph_data['pairwise_relation_type_ls'].append(tgt.relation_type)
            else:
                graph_data['pairwise_relation_type_ls'].append(default_relation_type)
        
        answer_text = "Answer: " + ("Yes" if tgt.label == 1 else "No")
        text = f'{config.candidate_target_node_intro}{info_part}{config.connection_question}{answer_text}'
        tgt_text_ls.append(text)
    tgt_text_all = '\n\n'.join(tgt_text_ls)
            
    prompt = f"{config.task_desc}{config.source_node_intro}{src_text}{tgt_text_all}"
    return prompt, graph_data


def generate_yn_labels(tokenizer, input_ids, learn_text: bool=True, learn_yn: bool=True, learn_all: bool=False):
    """
    Generate labels for the Yes/No prompt.
    Args:
        tokenizer: The tokenizer used to tokenize the text.
        input_ids: The input_ids of the Yes/No prompt.
        learn_text: Whether to calculate the loss for the text part.
        learn_yn: Whether to calculate the loss for the Yes/No part.
        learn_all: Whether to calculate the loss for all tokens (if True, learn_text and learn_yn will be ignored. And the training would be ordinary FT, instead of supervised FT.)
    """
    
    if learn_all:
        labels = input_ids[1:]
        return labels
    labels = (IGNORE_INDEX * torch.ones(len(input_ids)-1)).long()
    
    if learn_text:
        newline_id = tokenizer('x\nx', add_special_tokens=False)['input_ids'][1]
        text_prefix_ids = tokenizer.convert_tokens_to_ids(['text', ':'])
        
        potential_pos_list = torch.where(input_ids == text_prefix_ids[0])[0].tolist()
        real_pos_list = [pos for pos in potential_pos_list if pos+1 < len(input_ids) and input_ids[pos+1] == text_prefix_ids[1]]
        start_pos_list = [pos + 2 for pos in real_pos_list]
        for start_pos in start_pos_list:
            try:
                end_pos = torch.where(input_ids[start_pos:] == newline_id)[0][0].item() + start_pos + 1 # plus 1 is to include the '\n'
            except:
                end_pos = len(input_ids)
            labels[(start_pos-1):(end_pos-1)] = input_ids[start_pos:end_pos]
    
    if learn_yn:
        yn_prefix_ids = tokenizer.convert_tokens_to_ids(['Answer', ':'])
        potential_pos_list = torch.where(input_ids == yn_prefix_ids[0])[0].tolist()
        real_pos_list = [pos for pos in potential_pos_list if pos+1 < len(input_ids) and input_ids[pos+1] == yn_prefix_ids[1]]
        start_pos_list = [pos + 2 for pos in real_pos_list]
        for start_pos in start_pos_list:
            labels[start_pos-1] = input_ids[start_pos]
        
    return labels

@dataclass
class YNDatasetForEvalConfig:
    num_tgt_per_prompt: int = 4
    num_neg_per_pos: float = 1
    
    task_desc: str = "Determine whether there is a link between the source node and the candidate nodes.\n"
    source_node_intro: str = "Source node:\n"
    candidate_target_node_intro: str = "Candidate target node:\n"
    connection_question: str = "Is this connected to the source node?\n"
    
    ablate_pairwise_encoding: bool = False
    ablate_node_encoding: bool = False
    node_encoding_max_hop: int = 0
    
    generate_at_initialization: bool = True

    node_id2type: Optional[Dict[int, int]] = None
    type2node_ids: Optional[Dict[int, List[int]]] = None
    relation_type_map: Optional[Dict[Tuple[int, int], int]] = None
    default_node_type: int = 0
    default_relation_type: int = 0
    
class YNDatasetForEval(Dataset):
    def __init__(
        self,
        question_data: Dict[str, object],
        dgl_graph,
        gnid2text: Dict[int, str],
        config: YNDatasetForEvalConfig,
        tokenizer
    ):
        self.question_data = question_data
        self.dgl_graph = dgl_graph
        self.gnid2text = gnid2text
        self.config = config
        self.num_nodes = dgl_graph.num_nodes()
        self.tokenizer = tokenizer

        self.node_id2type = dict(config.node_id2type) if config.node_id2type is not None else {}
        if config.type2node_ids is not None:
            self.type2node_ids = {k: list(v) for k, v in config.type2node_ids.items()}
        elif self.node_id2type:
            type2node_ids: Dict[int, List[int]] = {}
            for nid, type_id in self.node_id2type.items():
                type2node_ids.setdefault(type_id, []).append(nid)
            self.type2node_ids = type2node_ids
        else:
            self.type2node_ids = None
        self.default_node_type = getattr(config, 'default_node_type', 0)

        self.relation_type_map = config.relation_type_map if config.relation_type_map is not None else {}
        self.default_relation_type = getattr(config, 'default_relation_type', 0)
        self.num_node_types = max(self.node_id2type.values(), default=-1) + 1 if self.node_id2type else 1
        self.num_relation_types = max(self.relation_type_map.values(), default=-1) + 1 if self.relation_type_map else 1
        
        self.yn_eval_data_list = None
        if self.config.generate_at_initialization:
            self.generate_yn_eval_data_list()
        
    def _lookup_node_type(self, node_id: int) -> Optional[int]:
        if not self.node_id2type:
            return None
        return self.node_id2type.get(node_id, self.default_node_type)

    def _ensure_relation_type(self, src_node: int, tgt_node: int) -> Optional[int]:
        if not self.node_id2type:
            return None
        src_type = self._lookup_node_type(src_node)
        tgt_type = self._lookup_node_type(tgt_node)
        if src_type is None or tgt_type is None:
            return None
        key = (src_type, tgt_type)
        if key not in self.relation_type_map:
            self.relation_type_map[key] = len(self.relation_type_map)
        relation_id = self.relation_type_map[key]
        self.num_relation_types = max(self.num_relation_types, relation_id + 1)
        return relation_id

    def _build_target_data(self, src: int, tgt: int, label: int) -> YNTargetData:
        tgt_type = self._lookup_node_type(tgt)
        relation_type = self._ensure_relation_type(src, tgt)
        if relation_type is None:
            relation_type = self.default_relation_type
        return YNTargetData(tgt_node=tgt, label=label, tgt_type=tgt_type, relation_type=relation_type)
    
    def generate_yn_eval_data_list(self, num_questions=None):
        yn_eval_data_list = []
        
        config = self.config
        total_num = num_questions if num_questions is not None else len(self.question_data['source_node']) # for debug only
        for i in trange(total_num):
            src = self.question_data['source_node'][i]
            pos_tgt = self.question_data['target_node'][i]
            neg_tgts = self.question_data['target_node_neg'][i]
            
            all_pos_neighbors = self.dgl_graph.successors(src).cpu().tolist()
            pos_neighbors = random.sample(all_pos_neighbors, min(len(all_pos_neighbors), config.num_tgt_per_prompt // 2))
            neg_neighbor_num = round(len(pos_neighbors) * config.num_neg_per_pos)
            
            candidate_pool = None
            if self.type2node_ids:
                candidate_pool = []
                src_type = self._lookup_node_type(src)
                neighbor_types = {self._lookup_node_type(nbr) for nbr in all_pos_neighbors if self._lookup_node_type(nbr) is not None}
                if not neighbor_types and src_type is not None:
                    neighbor_types = {ctype for ctype in self.type2node_ids.keys() if ctype != src_type}
                for ctype in neighbor_types:
                    candidate_pool.extend(self.type2node_ids.get(ctype, []))
            neg_neighbors = sample_neg_tgt(
                num_neg_tgt=neg_neighbor_num,
                pos_tgt_set=set(all_pos_neighbors),
                total_node_num=self.num_nodes,
                candidate_pool=candidate_pool,
            )

            pos_neighbor_data_ls = [self._build_target_data(src, tgt, 1) for tgt in pos_neighbors]
            neg_neighbor_data_ls = [self._build_target_data(src, tgt, 0) for tgt in neg_neighbors]

            all_neighbor_ls = pos_neighbor_data_ls + neg_neighbor_data_ls
            random.shuffle(all_neighbor_ls)
            
            yn_data = YNData(src_node=src, tgt_list=all_neighbor_ls)
            
            candidate_list = []
            for idx, tgt in enumerate([pos_tgt] + neg_tgts):
                label = 1 if idx == 0 else 0
                candidate_list.append(self._build_target_data(src, tgt, label))
            
            yn_eval_data_list.append((yn_data, candidate_list))
        self.yn_eval_data_list = yn_eval_data_list
    
    def __len__(self):
        if self.yn_eval_data_list is None:
            raise ValueError("Data list not generated yet!")
        return len(self.yn_eval_data_list)

    def __getitem__(self, index: int):
        if self.yn_eval_data_list is None:
            raise ValueError("Data list not generated yet!")
        
        yn_data, candidate_list = self.yn_eval_data_list[index]
        yn_data_clone = copy.deepcopy(yn_data)
        yn_data_clone.tgt_list.append(None)
        config = self.config
        
        ret = []
        for candidate_tgt in candidate_list:
            yn_data_clone.tgt_list[-1] = candidate_tgt
            prompt, graph_data = yn_data_to_prompt_and_graph_data(self.gnid2text, yn_data_clone, self.config)
            ret.append((prompt, graph_data))
        return ret