#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
局部时序特征提取：使用Transformer Encoder处理窗口内的LogInstance序列
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class LocalTransformerEncoder(nn.Module):
    """局部Transformer编码器：处理窗口内的日志序列"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        """
        初始化局部Transformer编码器
        
        Args:
            input_dim: 输入维度（模板嵌入维度）
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            max_seq_len: 最大序列长度
        """
        super(LocalTransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, log_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            log_embeddings: 日志嵌入矩阵 (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: 编码后的特征 (batch_size, seq_len, d_model)
        """
        # 投影到d_model维度
        x = self.input_projection(log_embeddings)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        return x




