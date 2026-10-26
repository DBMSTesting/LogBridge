#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局知识图谱数据加载器：支持跨窗口信息传递
构建全局图，所有窗口共享Entity节点，实现跨窗口的实体关联建模
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

class GlobalKGDataLoader:
    """全局知识图谱数据加载器（支持跨窗口信息传递）"""

    ANOMALY_TYPE_TO_LABEL = {
        'compaction': 0,
        'export': 1,
        'flush': 2,
        'full_cpu': 3,
        'full_memory': 4,
        'network_bandwidth2': 5,
        'normal': 6
    }
    
    LABEL_TO_ANOMALY_TYPE = {v: k for k, v in ANOMALY_TYPE_TO_LABEL.items()}
    
    def __init__(self, kg_file: Path):
        """
        初始化全局知识图谱数据加载器
        
        Args:
            kg_file: 细粒度知识图谱JSON文件路径
        """
        self.kg_file = kg_file
        self.kg_data = None
        self.nodes_dict = {}
        self.edges_dict = defaultdict(list)

        self.window_idx_map = {}
        self.log_idx_map = {}
        self.entity_idx_map = {}

        self.idx_to_window = {}
        self.idx_to_log = {}
        self.idx_to_entity = {}

        self.window_to_logs = defaultdict(list)
        self.log_to_window = {}

        self.entity_to_logs = defaultdict(set)
        
    def load_kg(self):
        """加载知识图谱并构建全局索引"""
        import sys
        print(f"加载全局知识图谱: {self.kg_file}")
        sys.stdout.flush()
        with open(self.kg_file, 'r', encoding='utf-8') as f:
            self.kg_data = json.load(f)

        print("  建立全局节点索引...")
        sys.stdout.flush()
        total_nodes = len(self.kg_data['nodes'])
        print(f"  总节点数: {total_nodes:,}")
        sys.stdout.flush()
        for i, node in enumerate(self.kg_data['nodes']):
            if i % 1000000 == 0 and i > 0:
                print(f"    已处理节点: {i:,}/{total_nodes:,} ({100*i/total_nodes:.1f}%)")
                sys.stdout.flush()
            node_id = node['id']
            node_type = node['type']
            self.nodes_dict[node_id] = node
            
            if node_type == 'Window':
                idx = len(self.window_idx_map)
                self.window_idx_map[node_id] = idx
                self.idx_to_window[idx] = node_id
            elif node_type == 'LogInstance' or node_type == 'Log':

                idx = len(self.log_idx_map)
                self.log_idx_map[node_id] = idx
                self.idx_to_log[idx] = node_id
            elif node_type in ['GeneralEntity', 'Thread', 'DataRegion', 'Node', 'ConsensusGroup', 'Anomaly']:

                if node_id not in self.entity_idx_map:
                    idx = len(self.entity_idx_map)
                    self.entity_idx_map[node_id] = idx
                    self.idx_to_entity[idx] = node_id
        print(f"    节点索引建立完成: {total_nodes:,} 个节点")
        sys.stdout.flush()

        print("  建立边索引...")
        sys.stdout.flush()
        total_edges = len(self.kg_data['edges'])
        print(f"  总边数: {total_edges:,}")
        sys.stdout.flush()
        for i, edge in enumerate(self.kg_data['edges']):
            if i % 1000000 == 0 and i > 0:
                print(f"    已处理边: {i:,}/{total_edges:,} ({100*i/total_edges:.1f}%)")
                sys.stdout.flush()
            source = edge['source']
            target = edge['target']
            relation = edge['relation']
            self.edges_dict[source].append((target, relation, edge.get('properties', {})))

            if relation == 'HAS_LOG':

                window_id = source
                log_id = target
                self.window_to_logs[window_id].append(log_id)
                self.log_to_window[log_id] = window_id
            elif relation == 'CONTAINS':

                window_id = source
                target_node = self.nodes_dict.get(target)
                target_type = target_node.get('type', '') if target_node else ''
                if target_type in ['LogInstance', 'Log']:

                    log_id = target
                    self.window_to_logs[window_id].append(log_id)
                    self.log_to_window[log_id] = window_id
                else:

                    entity_id = target
                    if window_id in self.window_to_logs:
                        for log_id in self.window_to_logs[window_id]:
                            self.entity_to_logs[entity_id].add(log_id)

            if relation == 'ASSOCIATED_WITH' and (source.startswith('LogInstance:') or source.startswith('Log:')):
                log_id = source
                entity_id = target
                self.entity_to_logs[entity_id].add(log_id)
        print(f"    边索引建立完成: {total_edges:,} 条边")
        sys.stdout.flush()
        
        import sys

        windows_with_logs = sum(1 for v in self.window_to_logs.values() if len(v) > 0)
        total_log_links = sum(len(v) for v in self.window_to_logs.values())
        print(f"  节点数: Window={len(self.window_idx_map)}, "
              f"LogInstance={len(self.log_idx_map)}, "
              f"Entity={len(self.entity_idx_map)}")
        print(f"  边数: {len(self.kg_data['edges'])}")
        print(f"  窗口-日志映射: {len(self.window_to_logs)} 个窗口, "
              f"其中 {windows_with_logs} 个有日志, 共 {total_log_links:,} 条 Window->Log 链接")
        print(f"  实体到日志反向索引: {len(self.entity_to_logs)} 个实体")
        sys.stdout.flush()
    
    def get_window_label(self, window_id: str) -> int:
        """获取窗口的标签"""
        edges = self.edges_dict.get(window_id, [])
        for target, relation, props in edges:
            if relation == 'HAS_ANOMALY':
                anomaly_node = self.nodes_dict.get(target)
                if anomaly_node:
                    anomaly_type = anomaly_node.get('properties', {}).get('anomaly_type', 'normal')
                    if anomaly_type in self.ANOMALY_TYPE_TO_LABEL:
                        return self.ANOMALY_TYPE_TO_LABEL[anomaly_type]
        return self.ANOMALY_TYPE_TO_LABEL['normal']
    
    def build_global_hetero_data(
        self,
        window_ids: Optional[List[str]] = None,
        max_windows: Optional[int] = None
    ) -> HeteroData:
        """
        构建全局异构图（包含所有或指定的窗口）
        
        Args:
            window_ids: 要包含的窗口ID列表，如果为None则包含所有窗口
            max_windows: 最大窗口数（用于限制图大小，便于测试）
            
        Returns:
            HeteroData: 全局异构图
        """
        if window_ids is None:
            window_ids = list(self.window_idx_map.keys())
        
        if max_windows is not None:
            window_ids = window_ids[:max_windows]
        
        print(f"构建全局图，包含 {len(window_ids)} 个窗口...")

        log_ids_set = set()
        entity_ids_set = set()
        
        for window_id in window_ids:
            log_ids = self.window_to_logs.get(window_id, [])
            log_ids_set.update(log_ids)

            for log_id in log_ids:
                edges = self.edges_dict.get(log_id, [])
                for target, relation, props in edges:
                    if relation == 'ASSOCIATED_WITH':
                        entity_ids_set.add(target)
        
        log_ids = list(log_ids_set)
        entity_ids = list(entity_ids_set)
        
        print(f"  包含日志: {len(log_ids)}, 实体: {len(entity_ids)}")

        window_to_log_edges = []
        log_to_entity_edges = []
        entity_to_log_edges = []
        
        for window_id in window_ids:
            window_idx = self.window_idx_map[window_id]
            log_ids_in_window = self.window_to_logs.get(window_id, [])
            
            for log_id in log_ids_in_window:
                log_idx = self.log_idx_map[log_id]

                window_to_log_edges.append([window_idx, log_idx])

                edges = self.edges_dict.get(log_id, [])
                for target, relation, props in edges:
                    if relation == 'ASSOCIATED_WITH' and target in entity_ids_set:
                        entity_idx = self.entity_idx_map[target]
                        log_to_entity_edges.append([log_idx, entity_idx])

                        entity_to_log_edges.append([entity_idx, log_idx])

        data = HeteroData()

        data['window'].num_nodes = len(window_ids)
        data['log'].num_nodes = len(log_ids)
        data['entity'].num_nodes = len(entity_ids)

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

        data.window_ids = window_ids
        data.log_ids = log_ids
        data.entity_ids = entity_ids

        window_labels = [self.get_window_label(wid) for wid in window_ids]
        data.window_labels = torch.tensor(window_labels, dtype=torch.long)

        log_template_texts = {}
        for log_id in log_ids:
            log_node = self.nodes_dict.get(log_id)
            if log_node:
                raw_line = log_node.get('properties', {}).get('raw_line', '')
                template_id = log_node.get('properties', {}).get('template_id', '')
                if raw_line:
                    log_template_texts[log_id] = raw_line
                elif template_id:
                    log_template_texts[log_id] = template_id
        
        entity_contents = {}
        for entity_id in entity_ids:
            entity_node = self.nodes_dict.get(entity_id)
            if entity_node:
                if entity_node.get('type') == 'GeneralEntity':
                    token = entity_node.get('properties', {}).get('token', '')
                    if token:
                        entity_contents[entity_id] = token
                    else:
                        entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
                elif entity_node.get('type') == 'Thread':
                    thread_name = entity_node.get('properties', {}).get('thread_name', '')
                    if thread_name:
                        entity_contents[entity_id] = thread_name
                    else:
                        entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
                else:
                    entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
        
        data.log_template_texts = log_template_texts
        data.entity_contents = entity_contents
        
        return data
    
    def build_window_subgraph(
        self,
        window_id: str,
        num_hops: int = 2,
        max_neighbors: int = 50,
        allowed_windows: Optional[Set[str]] = None
    ) -> Tuple[HeteroData, int]:
        """
        为单个窗口构建子图（包含跨窗口的邻居）
        通过Entity节点连接，包含其他窗口的日志
        
        Args:
            window_id: 目标窗口ID
            num_hops: 采样跳数（通过Entity连接的窗口数）
            max_neighbors: 每个节点的最大邻居数
            
        Returns:
            Tuple[HeteroData, int]: (子图, 标签)
        """

        target_log_ids = set(self.window_to_logs.get(window_id, []))
        if not target_log_ids:
            return None, -1

        label = self.get_window_label(window_id)

        expanded_log_ids = set(target_log_ids)
        expanded_entity_ids = set()
        expanded_window_ids = {window_id}

        window_edges = self.edges_dict.get(window_id, [])
        for target, relation, props in window_edges:
            if relation == 'CONTAINS':

                entity_id = target

                entity_node = self.nodes_dict.get(entity_id)
                if entity_node and entity_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity']:
                    expanded_entity_ids.add(entity_id)

        for log_id in target_log_ids:
            edges = self.edges_dict.get(log_id, [])
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH':
                    expanded_entity_ids.add(target)

        candidate_windows = {window_id}

        for entity_id in list(expanded_entity_ids):
            associated_logs = self.entity_to_logs.get(entity_id, set())

            for log_id in associated_logs:
                if log_id in self.log_to_window:
                    window_id_for_log = self.log_to_window[log_id]

                    if allowed_windows is None or window_id_for_log in allowed_windows:
                        candidate_windows.add(window_id_for_log)

        if len(candidate_windows) > max_neighbors:

            other_windows = list(candidate_windows - {window_id})

            window_scores = []
            target_entity_set = expanded_entity_ids
            
            for other_window_id in other_windows:

                other_entity_set = set()
                other_window_edges = self.edges_dict.get(other_window_id, [])
                for target, relation, props in other_window_edges:
                    if relation == 'CONTAINS':
                        entity_id = target
                        entity_node = self.nodes_dict.get(entity_id)
                        if entity_node and entity_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity']:
                            other_entity_set.add(entity_id)

                other_log_ids = set(self.window_to_logs.get(other_window_id, []))
                for log_id in other_log_ids:
                    edges = self.edges_dict.get(log_id, [])
                    for target, relation, props in edges:
                        if relation == 'ASSOCIATED_WITH':
                            other_entity_set.add(target)

                shared_entities = target_entity_set & other_entity_set
                score = len(shared_entities)
                window_scores.append((other_window_id, score))

            window_scores.sort(key=lambda x: x[1], reverse=True)
            selected_windows = [wid for wid, _ in window_scores[:max_neighbors-1]]
            expanded_window_ids = {window_id} | set(selected_windows)
        else:
            expanded_window_ids = candidate_windows

        expanded_log_ids = set(target_log_ids)
        for entity_id in list(expanded_entity_ids):
            associated_logs = self.entity_to_logs.get(entity_id, set())
            for log_id in associated_logs:
                if log_id in self.log_to_window:
                    window_id_for_log = self.log_to_window[log_id]
                    if window_id_for_log in expanded_window_ids:
                        expanded_log_ids.add(log_id)

        expanded_entity_ids = set()

        MAX_EXPANDED_LOGS = 100000
        logs_to_process = list(expanded_log_ids)
        if len(logs_to_process) > MAX_EXPANDED_LOGS:
            import random

            target_logs_in_expanded = [lid for lid in logs_to_process if lid in target_log_ids]
            other_logs = [lid for lid in logs_to_process if lid not in target_log_ids]

            if len(other_logs) > MAX_EXPANDED_LOGS - len(target_logs_in_expanded):
                other_logs = random.sample(other_logs, MAX_EXPANDED_LOGS - len(target_logs_in_expanded))
            logs_to_process = target_logs_in_expanded + other_logs
            expanded_log_ids = set(logs_to_process)

        for wid in expanded_window_ids:
            window_edges = self.edges_dict.get(wid, [])
            for target, relation, props in window_edges:
                if relation == 'CONTAINS':
                    entity_id = target
                    entity_node = self.nodes_dict.get(entity_id)
                    if entity_node and entity_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity']:
                        expanded_entity_ids.add(entity_id)

        for log_id in logs_to_process:
            edges = self.edges_dict.get(log_id, [])
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH':
                    expanded_entity_ids.add(target)
        
        log_ids = list(expanded_log_ids)
        entity_ids = list(expanded_entity_ids)
        window_ids = list(expanded_window_ids)

        MAX_TOTAL_NODES = 200000
        if len(log_ids) + len(entity_ids) + len(window_ids) > MAX_TOTAL_NODES:
            return None, -1

        window_to_log_edges = []
        log_to_entity_edges = []
        entity_to_log_edges = []

        window_local_idx = {wid: i for i, wid in enumerate(window_ids)}
        log_local_idx = {lid: i for i, lid in enumerate(log_ids)}
        entity_local_idx = {eid: i for i, eid in enumerate(entity_ids)}

        log_to_entities = {}
        for wid in window_ids:
            window_edges = self.edges_dict.get(wid, [])
            window_logs = self.window_to_logs.get(wid, [])
            window_entities = []
            for target, relation, props in window_edges:
                if relation == 'CONTAINS' and target in entity_local_idx:
                    window_entities.append(target)

            for log_id in window_logs:
                if log_id in log_local_idx:
                    if log_id not in log_to_entities:
                        log_to_entities[log_id] = []
                    log_to_entities[log_id].extend(window_entities)

        for log_id in log_ids:
            edges = self.edges_dict.get(log_id, [])
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH' and target in entity_local_idx:
                    if log_id not in log_to_entities:
                        log_to_entities[log_id] = []
                    if target not in log_to_entities[log_id]:
                        log_to_entities[log_id].append(target)

        for wid in window_ids:
            w_local_idx = window_local_idx[wid]
            window_logs = self.window_to_logs.get(wid, [])
            for log_id in window_logs:
                if log_id in log_local_idx:
                    l_local_idx = log_local_idx[log_id]
                    window_to_log_edges.append([w_local_idx, l_local_idx])

                    if log_id in log_to_entities:
                        for entity_id in log_to_entities[log_id]:
                            e_local_idx = entity_local_idx[entity_id]
                            log_to_entity_edges.append([l_local_idx, e_local_idx])
                            entity_to_log_edges.append([e_local_idx, l_local_idx])

        data = HeteroData()
        
        data['window'].num_nodes = len(window_ids)
        data['log'].num_nodes = len(log_ids)
        data['entity'].num_nodes = len(entity_ids)
        
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
        data.window_ids = window_ids
        data.log_ids = log_ids
        data.entity_ids = entity_ids
        data.label = label

        log_template_texts = {}
        for log_id in log_ids:
            log_node = self.nodes_dict.get(log_id)
            if log_node:
                raw_line = log_node.get('properties', {}).get('raw_line', '')
                template_id = log_node.get('properties', {}).get('template_id', '')
                if raw_line:
                    log_template_texts[log_id] = raw_line
                elif template_id:
                    log_template_texts[log_id] = template_id
        
        entity_contents = {}
        for entity_id in entity_ids:
            entity_node = self.nodes_dict.get(entity_id)
            if entity_node:
                if entity_node.get('type') == 'GeneralEntity':
                    token = entity_node.get('properties', {}).get('token', '')
                    if token:
                        entity_contents[entity_id] = token
                    else:
                        entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
                elif entity_node.get('type') == 'Thread':
                    thread_name = entity_node.get('properties', {}).get('thread_name', '')
                    if thread_name:
                        entity_contents[entity_id] = thread_name
                    else:
                        entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
                else:
                    entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
        
        data.log_template_texts = log_template_texts
        data.entity_contents = entity_contents
        
        return data, label
    
    def get_all_windows(self) -> List[str]:
        """获取所有窗口ID"""
        return list(self.window_idx_map.keys())
    
    def get_window_node(self, window_id: str) -> Dict:
        """获取窗口节点"""
        return self.nodes_dict.get(window_id)
    
    def get_log_node(self, log_id: str) -> Dict:
        """获取日志节点"""
        return self.nodes_dict.get(log_id)
    
    def get_entity_node(self, entity_id: str) -> Dict:
        """获取实体节点"""
        return self.nodes_dict.get(entity_id)

