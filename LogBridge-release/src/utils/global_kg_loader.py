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
    
    # 异常类型到标签的映射
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
        
        # 全局节点索引映射（所有窗口共享）
        self.window_idx_map = {}  # window_id -> global_index
        self.log_idx_map = {}     # log_id -> global_index
        self.entity_idx_map = {}  # entity_id -> global_index
        
        # 反向映射
        self.idx_to_window = {}
        self.idx_to_log = {}
        self.idx_to_entity = {}
        
        # 窗口到日志的映射
        self.window_to_logs = defaultdict(list)  # window_id -> [log_ids]
        self.log_to_window = {}  # log_id -> window_id
        
        # 实体到日志的反向索引（优化性能）
        self.entity_to_logs = defaultdict(set)  # entity_id -> {log_ids}
        
    def load_kg(self):
        """加载知识图谱并构建全局索引"""
        import sys
        print(f"加载全局知识图谱: {self.kg_file}")
        sys.stdout.flush()
        with open(self.kg_file, 'r', encoding='utf-8') as f:
            self.kg_data = json.load(f)
        
        # 第一遍：建立所有节点的全局索引
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
                # 支持Log和LogInstance两种命名
                idx = len(self.log_idx_map)
                self.log_idx_map[node_id] = idx
                self.idx_to_log[idx] = node_id
            elif node_type in ['GeneralEntity', 'Thread', 'DataRegion', 'Node', 'ConsensusGroup', 'Anomaly']:
                # Entity节点全局共享（包括所有实体类型）
                if node_id not in self.entity_idx_map:
                    idx = len(self.entity_idx_map)
                    self.entity_idx_map[node_id] = idx
                    self.idx_to_entity[idx] = node_id
        print(f"    节点索引建立完成: {total_nodes:,} 个节点")
        sys.stdout.flush()
        
        # 第二遍：建立边索引和窗口-日志映射
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
            
            # 记录窗口-日志关系
            if relation == 'HAS_LOG':
                # HAS_LOG: Window -> Log（旧版KG可能使用）
                window_id = source
                log_id = target
                self.window_to_logs[window_id].append(log_id)
                self.log_to_window[log_id] = window_id
            elif relation == 'CONTAINS':
                # CONTAINS: Window -> LogInstance/Log 或 Window -> Entity
                # 当前KG结构（build_knowledge_graph.py）使用 CONTAINS 表示 Window->LogInstance
                window_id = source
                target_node = self.nodes_dict.get(target)
                target_type = target_node.get('type', '') if target_node else ''
                if target_type in ['LogInstance', 'Log']:
                    # Window --[CONTAINS]--> LogInstance
                    log_id = target
                    self.window_to_logs[window_id].append(log_id)
                    self.log_to_window[log_id] = window_id
                else:
                    # Window --[CONTAINS]--> Entity
                    entity_id = target
                    if window_id in self.window_to_logs:
                        for log_id in self.window_to_logs[window_id]:
                            self.entity_to_logs[entity_id].add(log_id)
            
            # 建立实体到日志的反向索引（优化性能）
            # 支持ASSOCIATED_WITH边（如果存在）
            if relation == 'ASSOCIATED_WITH' and (source.startswith('LogInstance:') or source.startswith('Log:')):
                log_id = source
                entity_id = target
                self.entity_to_logs[entity_id].add(log_id)
        print(f"    边索引建立完成: {total_edges:,} 条边")
        sys.stdout.flush()
        
        import sys
        # 统计 window_to_logs 来源（便于排查 CONTAINS vs HAS_LOG）
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
        
        # 收集所有相关的日志和实体
        log_ids_set = set()
        entity_ids_set = set()
        
        for window_id in window_ids:
            log_ids = self.window_to_logs.get(window_id, [])
            log_ids_set.update(log_ids)
            
            # 获取这些日志关联的实体
            for log_id in log_ids:
                edges = self.edges_dict.get(log_id, [])
                for target, relation, props in edges:
                    if relation == 'ASSOCIATED_WITH':
                        entity_ids_set.add(target)
        
        log_ids = list(log_ids_set)
        entity_ids = list(entity_ids_set)
        
        print(f"  包含日志: {len(log_ids)}, 实体: {len(entity_ids)}")
        
        # 构建边索引（使用全局索引）
        window_to_log_edges = []
        log_to_entity_edges = []
        entity_to_log_edges = []
        
        for window_id in window_ids:
            window_idx = self.window_idx_map[window_id]
            log_ids_in_window = self.window_to_logs.get(window_id, [])
            
            for log_id in log_ids_in_window:
                log_idx = self.log_idx_map[log_id]
                # Window -> Log
                window_to_log_edges.append([window_idx, log_idx])
                
                # Log -> Entity
                edges = self.edges_dict.get(log_id, [])
                for target, relation, props in edges:
                    if relation == 'ASSOCIATED_WITH' and target in entity_ids_set:
                        entity_idx = self.entity_idx_map[target]
                        log_to_entity_edges.append([log_idx, entity_idx])
                        # Entity -> Log (反向)
                        entity_to_log_edges.append([entity_idx, log_idx])
        
        # 构建HeteroData
        data = HeteroData()
        
        # 设置节点数量
        data['window'].num_nodes = len(window_ids)
        data['log'].num_nodes = len(log_ids)
        data['entity'].num_nodes = len(entity_ids)
        
        # 边数据
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
        
        # 存储元数据
        data.window_ids = window_ids
        data.log_ids = log_ids
        data.entity_ids = entity_ids
        
        # 存储窗口标签
        window_labels = [self.get_window_label(wid) for wid in window_ids]
        data.window_labels = torch.tensor(window_labels, dtype=torch.long)
        
        # 存储日志模板文本和实体内容
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
        # 获取目标窗口的日志
        target_log_ids = set(self.window_to_logs.get(window_id, []))
        if not target_log_ids:
            return None, -1
        
        # 获取标签
        label = self.get_window_label(window_id)
        
        # 通过Entity扩展，找到相关的其他窗口的日志
        expanded_log_ids = set(target_log_ids)
        expanded_entity_ids = set()
        expanded_window_ids = {window_id}
        
        # 第一跳：获取目标窗口关联的实体
        # 方法1：通过CONTAINS边（Window -> Entity）
        window_edges = self.edges_dict.get(window_id, [])
        for target, relation, props in window_edges:
            if relation == 'CONTAINS':
                # CONTAINS: Window -> Entity
                entity_id = target
                # 检查target是否是Entity节点（DataRegion, Node, ConsensusGroup等）
                entity_node = self.nodes_dict.get(entity_id)
                if entity_node and entity_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity']:
                    expanded_entity_ids.add(entity_id)
        
        # 方法2：通过ASSOCIATED_WITH边（Log -> Entity，如果存在）
        for log_id in target_log_ids:
            edges = self.edges_dict.get(log_id, [])
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH':
                    expanded_entity_ids.add(target)
        
        # 第二跳：通过实体找到其他窗口的日志（优化：先限制窗口数，再扩展日志）
        # 优化策略：先收集候选窗口，限制数量后再扩展日志，避免处理过多数据
        candidate_windows = {window_id}  # 候选窗口集合
        
        # 先通过实体找到候选窗口（不扩展所有日志）
        for entity_id in list(expanded_entity_ids):
            associated_logs = self.entity_to_logs.get(entity_id, set())
            # 只收集窗口ID，不扩展所有日志
            for log_id in associated_logs:
                if log_id in self.log_to_window:
                    window_id_for_log = self.log_to_window[log_id]
                    # 如果指定了allowed_windows，只考虑允许的窗口
                    if allowed_windows is None or window_id_for_log in allowed_windows:
                        candidate_windows.add(window_id_for_log)
                        # 如果已经达到限制，可以提前停止（但为了公平性，先收集所有候选）
        
        # 限制候选窗口数（避免图过大）
        if len(candidate_windows) > max_neighbors:
            # 优先保留目标窗口，然后按关联度选择其他窗口
            # 关联度 = 与目标窗口共享的实体数量
            other_windows = list(candidate_windows - {window_id})
            
            # 计算每个候选窗口与目标窗口的关联度（共享实体数量）
            window_scores = []
            target_entity_set = expanded_entity_ids  # 目标窗口关联的实体集合
            
            for other_window_id in other_windows:
                # 获取该窗口关联的实体（通过CONTAINS边）
                other_entity_set = set()
                other_window_edges = self.edges_dict.get(other_window_id, [])
                for target, relation, props in other_window_edges:
                    if relation == 'CONTAINS':
                        entity_id = target
                        entity_node = self.nodes_dict.get(entity_id)
                        if entity_node and entity_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity']:
                            other_entity_set.add(entity_id)
                
                # 也检查通过ASSOCIATED_WITH边关联的实体（如果存在）
                other_log_ids = set(self.window_to_logs.get(other_window_id, []))
                for log_id in other_log_ids:
                    edges = self.edges_dict.get(log_id, [])
                    for target, relation, props in edges:
                        if relation == 'ASSOCIATED_WITH':
                            other_entity_set.add(target)
                
                # 计算共享实体数量（关联度分数）
                shared_entities = target_entity_set & other_entity_set
                score = len(shared_entities)
                window_scores.append((other_window_id, score))
            
            # 按分数降序排序，选择top-k个窗口
            window_scores.sort(key=lambda x: x[1], reverse=True)
            selected_windows = [wid for wid, _ in window_scores[:max_neighbors-1]]
            expanded_window_ids = {window_id} | set(selected_windows)
        else:
            expanded_window_ids = candidate_windows
        
        # 现在只扩展选定窗口的日志（大幅减少处理量）
        expanded_log_ids = set(target_log_ids)  # 先包含目标窗口的日志
        for entity_id in list(expanded_entity_ids):
            associated_logs = self.entity_to_logs.get(entity_id, set())
            for log_id in associated_logs:
                if log_id in self.log_to_window:
                    window_id_for_log = self.log_to_window[log_id]
                    if window_id_for_log in expanded_window_ids:
                        expanded_log_ids.add(log_id)
        
        # 更新实体集合（只保留与扩展日志相关的实体）- 使用反向索引优化
        expanded_entity_ids = set()
        # 使用反向索引快速获取所有实体
        # 对超大日志集合进行限制，避免内存溢出
        MAX_EXPANDED_LOGS = 100000  # 扩展日志最大数量
        logs_to_process = list(expanded_log_ids)
        if len(logs_to_process) > MAX_EXPANDED_LOGS:
            import random
            # 优先保留目标窗口的日志
            target_logs_in_expanded = [lid for lid in logs_to_process if lid in target_log_ids]
            other_logs = [lid for lid in logs_to_process if lid not in target_log_ids]
            # 保留所有目标窗口日志，随机采样其他日志
            if len(other_logs) > MAX_EXPANDED_LOGS - len(target_logs_in_expanded):
                other_logs = random.sample(other_logs, MAX_EXPANDED_LOGS - len(target_logs_in_expanded))
            logs_to_process = target_logs_in_expanded + other_logs
            expanded_log_ids = set(logs_to_process)
        
        # 更新实体集合：通过Window的CONTAINS边和Log的ASSOCIATED_WITH边
        # 方法1：通过Window的CONTAINS边
        for wid in expanded_window_ids:
            window_edges = self.edges_dict.get(wid, [])
            for target, relation, props in window_edges:
                if relation == 'CONTAINS':
                    entity_id = target
                    entity_node = self.nodes_dict.get(entity_id)
                    if entity_node and entity_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity']:
                        expanded_entity_ids.add(entity_id)
        
        # 方法2：通过Log的ASSOCIATED_WITH边（如果存在）
        for log_id in logs_to_process:
            edges = self.edges_dict.get(log_id, [])
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH':
                    expanded_entity_ids.add(target)
        
        log_ids = list(expanded_log_ids)
        entity_ids = list(expanded_entity_ids)
        window_ids = list(expanded_window_ids)
        
        # 检查子图规模，如果过大则提前返回None（避免内存溢出）
        MAX_TOTAL_NODES = 200000  # 最大节点总数
        if len(log_ids) + len(entity_ids) + len(window_ids) > MAX_TOTAL_NODES:
            return None, -1
        
        # 构建边索引（优化：先建立映射，再构建边）
        window_to_log_edges = []
        log_to_entity_edges = []
        entity_to_log_edges = []
        
        # 窗口到日志的映射（局部索引）
        window_local_idx = {wid: i for i, wid in enumerate(window_ids)}
        log_local_idx = {lid: i for i, lid in enumerate(log_ids)}
        entity_local_idx = {eid: i for i, eid in enumerate(entity_ids)}
        
        # 预先建立log->entity的映射（避免重复查找）
        # 方法1：通过Window的CONTAINS边建立映射
        log_to_entities = {}
        for wid in window_ids:
            window_edges = self.edges_dict.get(wid, [])
            window_logs = self.window_to_logs.get(wid, [])
            window_entities = []
            for target, relation, props in window_edges:
                if relation == 'CONTAINS' and target in entity_local_idx:
                    window_entities.append(target)
            # 将该窗口的所有实体关联到该窗口的所有日志
            for log_id in window_logs:
                if log_id in log_local_idx:
                    if log_id not in log_to_entities:
                        log_to_entities[log_id] = []
                    log_to_entities[log_id].extend(window_entities)
        
        # 方法2：通过Log的ASSOCIATED_WITH边（如果存在）
        for log_id in log_ids:
            edges = self.edges_dict.get(log_id, [])
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH' and target in entity_local_idx:
                    if log_id not in log_to_entities:
                        log_to_entities[log_id] = []
                    if target not in log_to_entities[log_id]:
                        log_to_entities[log_id].append(target)
        
        # 构建边（优化后的版本）
        for wid in window_ids:
            w_local_idx = window_local_idx[wid]
            window_logs = self.window_to_logs.get(wid, [])
            for log_id in window_logs:
                if log_id in log_local_idx:
                    l_local_idx = log_local_idx[log_id]
                    window_to_log_edges.append([w_local_idx, l_local_idx])
                    
                    # Log -> Entity（使用预建立的映射）
                    if log_id in log_to_entities:
                        for entity_id in log_to_entities[log_id]:
                            e_local_idx = entity_local_idx[entity_id]
                            log_to_entity_edges.append([l_local_idx, e_local_idx])
                            entity_to_log_edges.append([e_local_idx, l_local_idx])
        
        # 构建HeteroData
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
        
        # 存储元数据
        data.target_window_id = window_id
        data.target_window_idx = window_local_idx.get(window_id, 0)
        data.window_ids = window_ids
        data.log_ids = log_ids
        data.entity_ids = entity_ids
        data.label = label
        
        # 存储模板文本和实体内容
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

