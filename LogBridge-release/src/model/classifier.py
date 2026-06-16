#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类器和损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class WindowClassifier(nn.Module):
    """窗口分类器（两层MLP + Softmax）"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dims: List[int] = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        初始化分类器
        
        Args:
            input_dim: 输入维度（窗口嵌入维度）
            num_classes: 类别数（7类：6种异常 + 1种正常）
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
        
        # 输出层
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


class FocalLoss(nn.Module):
    """Focal Loss：用于处理类别不平衡问题"""
    
    def __init__(self, alpha: List[float] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重列表（用于平衡类别），如果为None则使用均匀权重
            gamma: 聚焦参数，越大越关注难分类样本
            reduction: 损失归约方式（'mean', 'sum', 'none'）
        """
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 预测logits (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
            
        Returns:
            torch.Tensor: Focal Loss值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




