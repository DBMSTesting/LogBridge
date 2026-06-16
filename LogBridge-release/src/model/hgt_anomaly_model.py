#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的HGT异常分类模型
整合局部Transformer、HGT和分类器
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional

try:
    from .local_transformer import LocalTransformerEncoder
    from .hgt_model import HGTModel, AttentionPooling
    from .classifier import WindowClassifier
    from .encoder import LogInstanceEncoder
except ImportError:
    from local_transformer import LocalTransformerEncoder
    from hgt_model import HGTModel, AttentionPooling
    from classifier import WindowClassifier
    from encoder import LogInstanceEncoder


class HGTAnomalyModel(nn.Module):
    """完整的HGT异常分类模型"""
    
    def __init__(
        self,
        template_encoder,
        entity_encoder,
        log_embedding_dim: int = 384,
        entity_embedding_dim: int = 128,
        local_transformer_dim: int = 256,
        hgt_hidden_dim: int = 256,
        hgt_num_heads: int = 8,
        hgt_num_layers: int = 2,
        num_classes: int = 7,
        classifier_hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        """
        初始化完整模型
        
        Args:
            template_encoder: 模板编码器
            entity_encoder: 实体编码器
            log_embedding_dim: 日志嵌入维度（模板嵌入维度）
            entity_embedding_dim: 实体嵌入维度
            local_transformer_dim: 局部Transformer维度
            hgt_hidden_dim: HGT隐藏层维度
            hgt_num_heads: HGT注意力头数
            hgt_num_layers: HGT层数
            num_classes: 类别数
            classifier_hidden_dims: 分类器隐藏层维度
            dropout: Dropout率
            device: 设备
        """
        super(HGTAnomalyModel, self).__init__()
        
        self.device = device
        self.num_classes = num_classes
        self.log_embedding_dim = log_embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.local_transformer_dim = local_transformer_dim
        
        # 日志实例编码器
        self.log_encoder = LogInstanceEncoder(template_encoder, entity_encoder, log_embedding_dim)
        
        # 实体编码器（需要在forward中使用）
        self.entity_encoder = entity_encoder
        
        # 局部Transformer（处理窗口内日志序列）
        self.local_transformer = LocalTransformerEncoder(
            input_dim=log_embedding_dim,
            d_model=local_transformer_dim,
            num_layers=2,
            dropout=dropout
        ).to(device)
        
        # HGT模型
        self.hgt = HGTModel(
            log_dim=local_transformer_dim,
            entity_dim=entity_embedding_dim,
            hidden_dim=hgt_hidden_dim,
            num_heads=hgt_num_heads,
            num_layers=hgt_num_layers,
            dropout=dropout
        ).to(device)
        
        # 注意力池化
        self.attention_pooling = AttentionPooling(
            input_dim=hgt_hidden_dim,
            hidden_dim=128
        ).to(device)
        
        # 分类器
        self.classifier = WindowClassifier(
            input_dim=hgt_hidden_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout_rate=dropout
        ).to(device)
    
    def forward(
        self,
        hetero_data: HeteroData,
        log_template_texts: Optional[Dict[str, str]] = None,
        target_window_idx: Optional[int] = None,
        profile_timing: bool = False
    ) -> Optional[torch.Tensor]:
        """
        前向传播（支持跨窗口信息传递）
        
        Args:
            hetero_data: HeteroData对象，包含窗口的图结构（可能包含多个窗口）
            log_template_texts: 日志模板文本字典 {log_id: template_text}（可选）
            target_window_idx: 目标窗口的索引（如果hetero_data包含多个窗口，用于选择要分类的窗口）
            profile_timing: 是否记录各步骤耗时（用于性能分析）
            
        Returns:
            torch.Tensor: 分类logits (num_classes,)
        """
        import time
        timings = {}
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_total = time.time()
        
        # 1. 编码日志实例（使用实际的模板文本内容，批量编码以提高性能）
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_step = time.time()
        
        log_ids = hetero_data.log_ids
        
        # 边界情况：无日志时返回 None，由训练循环跳过（避免 torch.stack 空列表及 backward 梯度错误）
        if len(log_ids) == 0:
            return None
        
        # 优先使用hetero_data中存储的模板文本
        if hasattr(hetero_data, 'log_template_texts'):
            template_texts = hetero_data.log_template_texts
        else:
            template_texts = log_template_texts or {}
        
        # 收集所有模板文本（批量编码），空字符串用 log_id 兜底（避免 BERT tokenizer 报错）
        template_text_list = []
        for log_id in log_ids:
            template_text = template_texts.get(log_id, log_id)
            if not template_text or not str(template_text).strip():
                template_text = log_id if log_id else "unknown"
            template_text_list.append(str(template_text))
        
        # 空列表时 encoder.encode_batch 会返回 (0, dim)，下游需能处理
        # （正常子图应有日志，空列表通常不会出现）
        
        # 批量编码所有模板（比逐个编码快得多）
        log_embeddings = self.log_encoder.template_encoder.encode_batch(template_text_list)  # (num_logs, log_embedding_dim)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['bert_encoding'] = time.time() - start_step
            start_step = time.time()
        
        # 2. 局部Transformer编码（按窗口分组处理）
        # count-aware 展开：若 hetero_data.log_counts 提供（压缩日志的 count 字段），
        # 在 LT 输入侧把每条聚合 log 按 count 复制成多个 embedding，
        # 让 LT 看到与未压缩日志等价的频率信号；LT 输出后按段平均 pool 回原长度，
        # 保证下游 HGT 节点数不变。
        log_counts_map = getattr(hetero_data, 'log_counts', None)

        def _lt_with_counts(embs_in_window: torch.Tensor, log_indices_in_window: List[int]) -> torch.Tensor:
            """对单个窗口的 log embeddings 做 count-aware LT，返回 (num_logs_in_window, local_transformer_dim)。"""
            if log_counts_map is not None:
                counts = [max(int(log_counts_map.get(log_ids[idx], 1)), 1) for idx in log_indices_in_window]
            else:
                counts = [1] * embs_in_window.size(0)
            counts_t = torch.tensor(counts, dtype=torch.long, device=embs_in_window.device)
            expanded = embs_in_window.repeat_interleave(counts_t, dim=0).unsqueeze(0)
            out = self.local_transformer(expanded).squeeze(0)  # (sum_counts, dim)
            if int(counts_t.sum().item()) == embs_in_window.size(0):
                # 无展开，直接返回
                return out
            seg_idx = torch.arange(embs_in_window.size(0), device=embs_in_window.device).repeat_interleave(counts_t)
            pooled = torch.zeros(embs_in_window.size(0), out.size(1), device=out.device, dtype=out.dtype)
            pooled.index_add_(0, seg_idx, out)
            pooled = pooled / counts_t.unsqueeze(1).to(pooled.dtype)
            return pooled

        if hasattr(hetero_data, 'window_ids') and len(hetero_data.window_ids) > 1:
            # 多窗口模式：需要按窗口分组处理
            window_ids = hetero_data.window_ids
            log_features_dict = {}  # {log_idx: feature}

            # 获取每个窗口的日志索引
            window_log_indices = {}
            if ('window', 'CONTAINS', 'log') in hetero_data.edge_types:
                window_log_edges = hetero_data[('window', 'CONTAINS', 'log')].edge_index
                for i in range(window_log_edges.size(1)):
                    w_idx = window_log_edges[0, i].item()
                    l_idx = window_log_edges[1, i].item()
                    if w_idx not in window_log_indices:
                        window_log_indices[w_idx] = []
                    window_log_indices[w_idx].append(l_idx)

            # 对每个窗口的日志序列进行Transformer编码
            for w_idx in range(len(window_ids)):
                if w_idx in window_log_indices:
                    log_indices = sorted(window_log_indices[w_idx])
                    window_log_embs = log_embeddings[log_indices]
                    if len(window_log_embs) > 0:
                        window_log_features = _lt_with_counts(window_log_embs, log_indices)
                        for i, orig_idx in enumerate(log_indices):
                            log_features_dict[orig_idx] = window_log_features[i]

            # 构建完整的log_features（按log_ids的顺序）
            # 防御性检查：确保列表非空（避免 torch.stack 空列表报错）
            log_features_list = [log_features_dict.get(i, torch.zeros(self.local_transformer_dim, device=self.device))
                                 for i in range(len(log_ids))]
            if len(log_features_list) == 0:
                return None
            log_features = torch.stack(log_features_list)
        else:
            # 单窗口模式：直接处理所有日志（带 count-aware 展开）
            log_features = _lt_with_counts(log_embeddings, list(range(len(log_ids))))
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['local_transformer'] = time.time() - start_step
            start_step = time.time()
        
        # 3. 编码实体（使用实际的实体内容）
        entity_ids = hetero_data.entity_ids
        # 获取实体内容（从hetero_data中）
        entity_contents = getattr(hetero_data, 'entity_contents', None)
        
        # 处理空实体列表的情况
        if len(entity_ids) == 0:
            # 如果没有实体，创建一个零向量
            entity_embeddings = torch.zeros(0, self.entity_embedding_dim, device=self.device)
        else:
            entity_embeddings = self.entity_encoder.encode_batch(entity_ids, entity_contents)  # (num_entities, entity_embedding_dim)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['entity_encoding'] = time.time() - start_step
            start_step = time.time()
        
        # 4. HGT消息传递（跨窗口信息传递在这里发生）
        x_dict = {
            'log': log_features,
            'entity': entity_embeddings
        }
        
        # 构建边索引字典，但只包含有对应节点的边
        edge_index_dict = {}
        
        # 只有当log和entity节点都存在时，才添加相关的边
        if len(log_features) > 0 and len(entity_embeddings) > 0:
            if ('log', 'ASSOCIATED_WITH', 'entity') in hetero_data.edge_types:
                edge_idx = hetero_data[('log', 'ASSOCIATED_WITH', 'entity')].edge_index
                # 检查边的节点索引是否在有效范围内
                if edge_idx.size(1) > 0:
                    valid_mask = (edge_idx[0] < len(log_features)) & (edge_idx[1] < len(entity_embeddings))
                    if valid_mask.any():
                        edge_index_dict[('log', 'ASSOCIATED_WITH', 'entity')] = edge_idx[:, valid_mask]
            
            if ('entity', 'REVERSE_ASSOCIATED_WITH', 'log') in hetero_data.edge_types:
                edge_idx = hetero_data[('entity', 'REVERSE_ASSOCIATED_WITH', 'log')].edge_index
                # 检查边的节点索引是否在有效范围内
                if edge_idx.size(1) > 0:
                    valid_mask = (edge_idx[0] < len(entity_embeddings)) & (edge_idx[1] < len(log_features))
                    if valid_mask.any():
                        edge_index_dict[('entity', 'REVERSE_ASSOCIATED_WITH', 'log')] = edge_idx[:, valid_mask]
        
        # 如果没有任何边，或者某个节点类型为空，直接返回原始特征（不进行消息传递）
        if len(edge_index_dict) == 0 or len(log_features) == 0 or len(entity_embeddings) == 0:
            # 如果没有边或节点为空，直接返回原始特征（不进行消息传递）
            updated_x_dict = x_dict
        else:
            # 正常情况：有节点和边，进行HGT消息传递
            updated_x_dict = self.hgt(x_dict, edge_index_dict)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['hgt_message_passing'] = time.time() - start_step
            start_step = time.time()
        
        # 5. 注意力池化（聚合目标窗口的日志特征）
        if target_window_idx is not None and hasattr(hetero_data, 'window_ids'):
            # 多窗口模式：只聚合目标窗口的日志
            window_ids = hetero_data.window_ids
            if target_window_idx < len(window_ids):
                # 获取目标窗口的日志索引
                if ('window', 'CONTAINS', 'log') in hetero_data.edge_types:
                    window_log_edges = hetero_data[('window', 'CONTAINS', 'log')].edge_index
                    target_log_indices = []
                    for i in range(window_log_edges.size(1)):
                        if window_log_edges[0, i].item() == target_window_idx:
                            target_log_indices.append(window_log_edges[1, i].item())
                    
                    if target_log_indices:
                        target_log_features = updated_x_dict['log'][target_log_indices]
                        window_embedding = self.attention_pooling(target_log_features)
                    else:
                        # 如果没有日志，使用零向量
                        window_embedding = torch.zeros(self.hgt.hidden_dim, device=self.device)
                else:
                    window_embedding = self.attention_pooling(updated_x_dict['log'])
            else:
                window_embedding = self.attention_pooling(updated_x_dict['log'])
        else:
            # 单窗口模式：聚合所有日志
            window_embedding = self.attention_pooling(updated_x_dict['log'])  # (hgt_hidden_dim,)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['attention_pooling'] = time.time() - start_step
            start_step = time.time()
        
        # 6. 分类
        window_embedding = window_embedding.unsqueeze(0)  # (1, hgt_hidden_dim)
        logits = self.classifier(window_embedding)  # (1, num_classes)
        logits = logits.squeeze(0)  # (num_classes,)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['classification'] = time.time() - start_step
            timings['total'] = time.time() - start_total
            # 将timings存储到hetero_data中，以便在训练循环中访问
            if not hasattr(hetero_data, '_timings'):
                hetero_data._timings = timings
        
        return logits

