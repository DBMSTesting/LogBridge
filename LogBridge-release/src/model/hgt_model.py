#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构图Transformer (HGT) 模型
使用PyTorch Geometric的HGTConv进行跨窗口的实体关联建模
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, List, Tuple, Optional


class HGTModel(nn.Module):
    """异构图Transformer模型"""
    
    def __init__(
        self,
        log_dim: int,
        entity_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化HGT模型
        
        Args:
            log_dim: 日志节点特征维度
            entity_dim: 实体节点特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: HGT层数
            dropout: Dropout率
        """
        super(HGTModel, self).__init__()
        
        self.log_dim = log_dim
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        
        # 节点类型
        self.node_types = ['log', 'entity']
        
        # 边类型（元组格式：(source_type, relation, target_type)）
        self.edge_types = [
            ('log', 'ASSOCIATED_WITH', 'entity'),
            ('entity', 'REVERSE_ASSOCIATED_WITH', 'log')
        ]
        
        # 输入投影层（将不同节点类型的特征投影到统一维度）
        self.log_projection = Linear(log_dim, hidden_dim)
        self.entity_projection = Linear(entity_dim, hidden_dim)
        
        # HGT层
        self.hgt_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hgt_layers.append(
                HGTConv(
                    hidden_dim,
                    hidden_dim,
                    metadata=(self.node_types, self.edge_types),
                    heads=num_heads
                )
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典 {'log': tensor, 'entity': tensor}
            edge_index_dict: 边索引字典
            
        Returns:
            Dict[str, torch.Tensor]: 更新后的节点特征字典
        """
        # 投影到统一维度
        x_dict['log'] = self.log_projection(x_dict['log'])
        x_dict['entity'] = self.entity_projection(x_dict['entity'])
        
        # 通过HGT层
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, edge_index_dict)
            # 应用dropout（除了最后一层）
            if hgt_layer != self.hgt_layers[-1]:
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        
        return x_dict


class AttentionPooling(nn.Module):
    """注意力池化层：将窗口内所有LogInstance的特征聚合成窗口表示"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        初始化注意力池化层
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 注意力隐藏层维度
        """
        super(AttentionPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, log_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        注意力池化
        
        Args:
            log_features: 日志特征矩阵 (num_logs, feature_dim)
            mask: 掩码张量 (num_logs,)，True表示有效位置
            
        Returns:
            torch.Tensor: 池化后的窗口表示 (feature_dim,)
        """
        # 计算注意力权重
        attention_scores = self.attention(log_features)  # (num_logs, 1)
        
        # 应用掩码（如果有）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=0)  # (num_logs, 1)
        
        # 加权求和
        window_embedding = torch.sum(attention_weights * log_features, dim=0)  # (feature_dim,)
        
        return window_embedding

