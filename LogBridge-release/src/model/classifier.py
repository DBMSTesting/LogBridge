#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类器和损失函数
"""

import torch
import torch.nn as nn
from typing import List

class WindowClassifier(nn.Module):
    """窗口分类器（两层MLP + Softmax）"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        hidden_dims: List[int] = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        初始化分类器
        
        Args:
            input_dim: 输入维度（窗口嵌入维度）
            num_classes: 故障类别数量
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout率
        """
        super(WindowClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, window_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            window_embedding: 窗口嵌入向量 (batch_size, input_dim)
            
        Returns:
            torch.Tensor: 分类logits (batch_size, num_classes)
        """
        return self.classifier(window_embedding)


