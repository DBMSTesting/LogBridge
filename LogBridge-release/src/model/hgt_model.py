#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构图Transformer (HGT) 模型
使用PyTorch Geometric的HGTConv进行跨窗口的实体关联建模
"""

import torch
import torch.nn as nn
import math
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import softmax
from typing import Dict, List, Tuple, Optional

class AuditableHGTConv(HGTConv):
    """Retain the actual normalized per-head attention used in propagation."""

    def message(self, k_j, q_i, v_j, edge_attr, index, ptr, size_i):
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = softmax(alpha / math.sqrt(q_i.size(-1)), index, ptr, size_i)
        self.last_alpha = alpha.detach()
        return (v_j * alpha.view(-1, self.heads, 1)).view(-1, self.out_channels)

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

        self.node_types = ['log', 'entity']

        self.edge_types = [
            ('log', 'ASSOCIATED_WITH', 'entity'),
            ('entity', 'REVERSE_ASSOCIATED_WITH', 'log')
        ]

        self.log_projection = Linear(log_dim, hidden_dim)
        self.entity_projection = Linear(entity_dim, hidden_dim)

        self.hgt_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hgt_layers.append(
                AuditableHGTConv(
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

        x_dict['log'] = self.log_projection(x_dict['log'])
        x_dict['entity'] = self.entity_projection(x_dict['entity'])

        self.last_attention = []
        for layer_index, hgt_layer in enumerate(self.hgt_layers):
            x_dict = hgt_layer(x_dict, edge_index_dict)
            offset = 0
            for relation, edges in edge_index_dict.items():
                n = edges.size(1)
                alpha = hgt_layer.last_alpha[offset:offset + n].cpu().tolist()
                self.last_attention.append({'layer': layer_index, 'relation': list(relation),
                                            'edges': edges.detach().cpu().t().tolist(),
                                            'weights_per_head': alpha})
                offset += n

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

        attention_scores = self.attention(log_features)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=0)
        self.last_weights = attention_weights.detach().squeeze(-1)

        window_embedding = torch.sum(attention_weights * log_features, dim=0)
        
        return window_embedding

