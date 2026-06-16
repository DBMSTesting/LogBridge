#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器：从细粒度知识图谱加载数据，构建PyG的HeteroData对象
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData


class FineGrainedKGDataLoader:
    """细粒度知识图谱数据加载器"""
    
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
        初始化数据加载器
        
        Args:
            kg_file: 细粒度知识图谱JSON文件路径
        """
        self.kg_file = kg_file
        self.kg_data = None
        self.nodes_dict = {}
        self.edges_dict = defaultdict(list)
        
        # 节点索引映射
        self.window_idx_map = {}  # window_id -> index
        self.log_idx_map = {}     # log_id -> index
        self.entity_idx_map = {}  # entity_id -> index
        
        # 反向映射
        self.idx_to_window = {}
        self.idx_to_log = {}
        self.idx_to_entity = {}
        
    def load_kg(self):
        """加载知识图谱"""
        print(f"加载知识图谱: {self.kg_file}")
        with open(self.kg_file, 'r', encoding='utf-8') as f:
            self.kg_data = json.load(f)
        
        # 建立节点索引
        for node in self.kg_data['nodes']:
            node_id = node['id']
            node_type = node['type']
            self.nodes_dict[node_id] = node
            
            if node_type == 'Window':
                idx = len(self.window_idx_map)
                self.window_idx_map[node_id] = idx
                self.idx_to_window[idx] = node_id
            elif node_type == 'LogInstance':
                idx = len(self.log_idx_map)
                self.log_idx_map[node_id] = idx
                self.idx_to_log[idx] = node_id
            elif node_type in ['GeneralEntity', 'Thread']:
                idx = len(self.entity_idx_map)
                self.entity_idx_map[node_id] = idx
                self.idx_to_entity[idx] = node_id
        
        # 建立边索引
        for edge in self.kg_data['edges']:
            source = edge['source']
            target = edge['target']
            relation = edge['relation']
            self.edges_dict[source].append((target, relation, edge.get('properties', {})))
        
        print(f"  节点数: Window={len(self.window_idx_map)}, "
              f"LogInstance={len(self.log_idx_map)}, "
              f"Entity={len(self.entity_idx_map)}")
        print(f"  边数: {len(self.kg_data['edges'])}")
    
    def get_window_label(self, window_id: str) -> int:
        """
        获取窗口的标签
        
        Args:
            window_id: 窗口ID
            
        Returns:
            int: 标签（0-6）
        """
        edges = self.edges_dict.get(window_id, [])
        for target, relation, props in edges:
            if relation == 'HAS_ANOMALY':
                anomaly_node = self.nodes_dict.get(target)
                if anomaly_node:
                    anomaly_type = anomaly_node.get('properties', {}).get('anomaly_type', 'normal')
                    if anomaly_type in self.ANOMALY_TYPE_TO_LABEL:
                        return self.ANOMALY_TYPE_TO_LABEL[anomaly_type]
        
        return self.ANOMALY_TYPE_TO_LABEL['normal']
    
    def get_window_logs(self, window_id: str) -> List[str]:
        """
        获取窗口内的所有日志实例ID
        
        Args:
            window_id: 窗口ID
            
        Returns:
            List[str]: 日志实例ID列表（按sequence_idx排序）
        """
        edges = self.edges_dict.get(window_id, [])
        log_edges = []
        for target, relation, props in edges:
            if relation == 'CONTAINS':
                sequence_idx = props.get('sequence_idx', 0)
                log_edges.append((sequence_idx, target))
        
        # 按sequence_idx排序
        log_edges.sort(key=lambda x: x[0])
        return [log_id for _, log_id in log_edges]
    
    def get_log_entities(self, log_id: str) -> List[str]:
        """
        获取日志实例关联的实体ID列表
        
        Args:
            log_id: 日志实例ID
            
        Returns:
            List[str]: 实体ID列表
        """
        edges = self.edges_dict.get(log_id, [])
        entity_ids = []
        for target, relation, props in edges:
            if relation == 'ASSOCIATED_WITH':
                entity_ids.append(target)
        return entity_ids
    
    def build_hetero_data_for_window(self, window_id: str) -> Tuple[HeteroData, int]:
        """
        为单个窗口构建HeteroData对象
        
        Args:
            window_id: 窗口ID
            
        Returns:
            Tuple[HeteroData, int]: (HeteroData对象, 标签)
        """
        # 获取窗口内的日志
        log_ids = self.get_window_logs(window_id)
        if not log_ids:
            return None, -1
        
        # 获取窗口标签
        label = self.get_window_label(window_id)
        
        # 获取所有相关的实体（通过日志关联）
        entity_ids_set = set()
        for log_id in log_ids:
            entities = self.get_log_entities(log_id)
            entity_ids_set.update(entities)
        
        entity_ids = list(entity_ids_set)
        
        # 构建节点索引（局部索引，仅针对当前窗口）
        log_local_idx = {log_id: i for i, log_id in enumerate(log_ids)}
        entity_local_idx = {entity_id: i for i, entity_id in enumerate(entity_ids)}
        
        # 构建边：Window -> LogInstance (CONTAINS)
        window_to_log = []
        for log_id in log_ids:
            log_idx = log_local_idx[log_id]
            window_to_log.append([0, log_idx])  # window只有一个，索引为0
        
        # 构建边：LogInstance -> Entity (ASSOCIATED_WITH)
        log_to_entity = []
        for log_id in log_ids:
            log_idx = log_local_idx[log_id]
            entities = self.get_log_entities(log_id)
            for entity_id in entities:
                if entity_id in entity_local_idx:
                    entity_idx = entity_local_idx[entity_id]
                    log_to_entity.append([log_idx, entity_idx])
        
        # 构建边：Entity -> LogInstance (反向，用于消息传递)
        entity_to_log = [[e[1], e[0]] for e in log_to_entity]
        
        # 构建HeteroData
        data = HeteroData()
        
        # 设置节点数量
        data['window'].num_nodes = 1
        data['log'].num_nodes = len(log_ids)
        data['entity'].num_nodes = len(entity_ids)
        
        # 边数据
        if window_to_log:
            data['window', 'CONTAINS', 'log'].edge_index = torch.tensor(
                window_to_log, dtype=torch.long
            ).t().contiguous()
        
        if log_to_entity:
            data['log', 'ASSOCIATED_WITH', 'entity'].edge_index = torch.tensor(
                log_to_entity, dtype=torch.long
            ).t().contiguous()
        
        if entity_to_log:
            data['entity', 'REVERSE_ASSOCIATED_WITH', 'log'].edge_index = torch.tensor(
                entity_to_log, dtype=torch.long
            ).t().contiguous()
        
        # 存储元数据（用于后续编码）
        data.window_id = window_id
        data.log_ids = log_ids
        data.entity_ids = entity_ids
        data.label = label
        
        # 存储日志模板文本和实体内容（用于编码）
        log_template_texts = {}
        for log_id in log_ids:
            log_node = self.get_log_node(log_id)
            if log_node:
                # 优先使用raw_line，如果没有则使用template_id
                raw_line = log_node.get('properties', {}).get('raw_line', '')
                template_id = log_node.get('properties', {}).get('template_id', '')
                if raw_line:
                    log_template_texts[log_id] = raw_line
                elif template_id:
                    log_template_texts[log_id] = template_id
        
        entity_contents = {}
        for entity_id in entity_ids:
            entity_node = self.get_entity_node(entity_id)
            if entity_node:
                # 获取实体的实际内容
                if entity_node.get('type') == 'GeneralEntity':
                    # GeneralEntity有token属性
                    token = entity_node.get('properties', {}).get('token', '')
                    if token:
                        entity_contents[entity_id] = token
                    else:
                        # 如果没有token，使用entity_id
                        entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
                elif entity_node.get('type') == 'Thread':
                    # Thread使用线程名称
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

