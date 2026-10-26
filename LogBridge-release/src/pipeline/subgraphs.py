#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct heterogeneous contexts from entity-linked earlier windows."""

import sys
import time
import pickle
from typing import List, Set, Optional, Dict, Tuple, Any
import torch

from torch_geometric.data import HeteroData

ANOMALY_TYPE_TO_LABEL = {
    'compaction': 0,
    'export': 1,
    'flush': 2,
    'full_cpu': 3,
    'full_memory': 4,
    'network_bandwidth2': 5,
    'normal': 6
}

MAX_TOTAL_NODES = 200000

_index = None

def _init_worker(index_path: str):
    """初始化worker进程，加载轻量级索引"""
    global _index
    with open(index_path, 'rb') as f:
        _index = pickle.load(f)
    print(f"  子进程加载轻量索引完成: {len(_index.get('all_windows', []))} 个窗口")
    sys.stdout.flush()

def _get_expanded_windows_from_precomputed(
    window_id: str,
    max_neighbors: int,
    allowed_windows: Optional[Set[str]],
    index: Dict
) -> Set[str]:
    """
    从预计算的window_top_neighbors中获取expanded_window_ids
    直接查表，O(k)复杂度，k为预计算的候选数
    """
    if index.get('protocol') == 'logbridge-context':
        from prepare_data import select_history
        return {window_id, *select_history(window_id, index, max_neighbors, allowed_windows)}
    raise ValueError('Context schema mismatch; run prepare_data.py')

def build_single_subgraph_lightweight(args_tuple: Tuple) -> Tuple[str, Any, int, float]:
    """
    基于轻量级索引构建单个子图
    
    Args:
        args_tuple: (window_id, num_hops, max_neighbors, allowed_windows)
        
    Returns:
        Tuple[window_id, hetero_data, label, execution_time]
    """
    global _index
    window_id, num_hops, max_neighbors, allowed_windows = args_tuple
    
    task_start = time.time()
    
    try:

        if window_id not in _index.get('window_entity_sets', {}):
            return (window_id, None, -1, time.time() - task_start)

        expanded_window_ids = _get_expanded_windows_from_precomputed(
            window_id, max_neighbors, allowed_windows, _index
        )

        window_to_logs = _index['window_to_logs']
        window_to_entities = _index['window_to_entities']
        log_to_entities = _index['log_to_entities']
        log_to_text = _index['log_to_text']
        entity_to_content = _index['entity_to_content']
        window_to_label = _index['window_to_label']
        
        expanded_log_ids = set()
        for wid in expanded_window_ids:
            expanded_log_ids.update(window_to_logs.get(wid, []))
        
        expanded_entity_ids = set()
        for wid in expanded_window_ids:
            expanded_entity_ids.update(window_to_entities.get(wid, set()))
        for log_id in expanded_log_ids:
            expanded_entity_ids.update(log_to_entities.get(log_id, []))

        if len(expanded_log_ids) + len(expanded_entity_ids) + len(expanded_window_ids) > MAX_TOTAL_NODES:
            return (window_id, None, -1, time.time() - task_start)

        label = window_to_label.get(window_id, ANOMALY_TYPE_TO_LABEL['normal'])

        expanded_window_ids = [window_id] + sorted(expanded_window_ids - {window_id})
        expanded_log_ids = [lid for wid in expanded_window_ids for lid in window_to_logs.get(wid, [])]
        expanded_entity_ids = sorted(expanded_entity_ids)
        window_local_idx = {wid: i for i, wid in enumerate(expanded_window_ids)}
        log_local_idx = {lid: i for i, lid in enumerate(expanded_log_ids)}
        entity_local_idx = {eid: i for i, eid in enumerate(expanded_entity_ids)}

        log_to_entities_local = {}
        for log_id in expanded_log_ids:
            kept = [e for e in log_to_entities.get(log_id, []) if e in entity_local_idx]
            if kept:
                log_to_entities_local[log_id] = kept

        window_to_log_edges = []
        log_to_entity_edges = []
        entity_to_log_edges = []
        
        for wid in expanded_window_ids:
            w_idx = window_local_idx[wid]
            for log_id in window_to_logs.get(wid, []):
                if log_id in log_local_idx:
                    l_idx = log_local_idx[log_id]
                    window_to_log_edges.append([w_idx, l_idx])
                    for entity_id in log_to_entities_local.get(log_id, []):
                        e_idx = entity_local_idx[entity_id]
                        log_to_entity_edges.append([l_idx, e_idx])
                        entity_to_log_edges.append([e_idx, l_idx])

        data = HeteroData()
        data['window'].num_nodes = len(expanded_window_ids)
        data['log'].num_nodes = len(expanded_log_ids)
        data['entity'].num_nodes = len(expanded_entity_ids)
        
        if window_to_log_edges:
            data['window', 'CONTAINS', 'log'].edge_index = torch.tensor(
                window_to_log_edges, dtype=torch.long
            ).t().contiguous()
        if log_to_entity_edges:
            data['log', 'ASSOCIATED_WITH', 'entity'].edge_index = torch.tensor(
                log_to_entity_edges, dtype=torch.long
            ).t().contiguous()
        if entity_to_log_edges:
            data['entity', 'REVERSE_ASSOCIATED_WITH', 'log'].edge_index = torch.tensor(
                entity_to_log_edges, dtype=torch.long
            ).t().contiguous()
        
        data.target_window_id = window_id
        data.target_window_idx = window_local_idx.get(window_id, 0)
        data.window_ids = list(expanded_window_ids)
        data.log_ids = list(expanded_log_ids)
        data.entity_ids = list(expanded_entity_ids)
        data.label = label

        log_template_texts = {}
        for log_id in expanded_log_ids:
            text = log_to_text.get(log_id)
            if text:
                log_template_texts[log_id] = text
        
        entity_contents = {}
        for entity_id in expanded_entity_ids:
            content = entity_to_content.get(entity_id)
            if content:
                entity_contents[entity_id] = content
            else:
                entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
        
        data.log_template_texts = log_template_texts
        data.entity_contents = entity_contents

        log_to_count = _index.get('log_to_count') if isinstance(_index, dict) else None
        if log_to_count:
            data.log_counts = {lid: log_to_count.get(lid, 1) for lid in expanded_log_ids}

        data = data.cpu()
        execution_time = time.time() - task_start
        return (window_id, data, label, execution_time)
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return (window_id, None, -1, time.time() - task_start)

