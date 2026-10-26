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
        num_classes: int = 6,
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

        self.log_encoder = LogInstanceEncoder(template_encoder, entity_encoder, log_embedding_dim)

        self.entity_encoder = entity_encoder

        self.local_transformer = LocalTransformerEncoder(
            input_dim=log_embedding_dim,
            d_model=local_transformer_dim,
            num_layers=2,
            dropout=dropout
        ).to(device)

        self.hgt = HGTModel(
            log_dim=local_transformer_dim,
            entity_dim=entity_embedding_dim,
            hidden_dim=hgt_hidden_dim,
            num_heads=hgt_num_heads,
            num_layers=hgt_num_layers,
            dropout=dropout
        ).to(device)

        self.attention_pooling = AttentionPooling(
            input_dim=hgt_hidden_dim,
            hidden_dim=128
        ).to(device)

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

        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_step = time.time()
        
        log_ids = hetero_data.log_ids
        if target_window_idx is None:
            target_window_idx = getattr(hetero_data, 'target_window_idx', 0)

        if len(log_ids) == 0:
            return None

        if hasattr(hetero_data, 'log_template_texts'):
            template_texts = hetero_data.log_template_texts
        else:
            template_texts = log_template_texts or {}

        template_text_list = []
        for log_id in log_ids:
            template_text = template_texts.get(log_id, log_id)
            if not template_text or not str(template_text).strip():
                template_text = log_id if log_id else "unknown"
            template_text_list.append(str(template_text))

        log_embeddings = self.log_encoder.template_encoder.encode_batch(template_text_list)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['bert_encoding'] = time.time() - start_step
            start_step = time.time()

        log_counts_map = getattr(hetero_data, 'log_counts', None)

        def _lt_with_counts(embs_in_window: torch.Tensor, log_indices_in_window: List[int]) -> torch.Tensor:
            """对单个窗口的 log embeddings 做 count-aware LT，返回 (num_logs_in_window, local_transformer_dim)。"""
            if log_counts_map is not None:
                counts = [max(int(log_counts_map.get(log_ids[idx], 1)), 1) for idx in log_indices_in_window]
            else:
                counts = [1] * embs_in_window.size(0)
            return self.local_transformer(embs_in_window.unsqueeze(0),
                                          counts=counts if log_counts_map is not None else None).squeeze(0)

        if hasattr(hetero_data, 'window_ids') and len(hetero_data.window_ids) > 1:

            window_ids = hetero_data.window_ids
            log_features_dict = {}

            window_log_indices = {}
            if ('window', 'CONTAINS', 'log') in hetero_data.edge_types:
                window_log_edges = hetero_data[('window', 'CONTAINS', 'log')].edge_index
                for i in range(window_log_edges.size(1)):
                    w_idx = window_log_edges[0, i].item()
                    l_idx = window_log_edges[1, i].item()
                    if w_idx not in window_log_indices:
                        window_log_indices[w_idx] = []
                    window_log_indices[w_idx].append(l_idx)

            for w_idx in range(len(window_ids)):
                if w_idx in window_log_indices:
                    log_indices = sorted(window_log_indices[w_idx])
                    window_log_embs = log_embeddings[log_indices]
                    if len(window_log_embs) > 0:
                        window_log_features = _lt_with_counts(window_log_embs, log_indices)
                        for i, orig_idx in enumerate(log_indices):
                            log_features_dict[orig_idx] = window_log_features[i]

            log_features_list = [log_features_dict.get(i, torch.zeros(self.local_transformer_dim, device=self.device))
                                 for i in range(len(log_ids))]
            if len(log_features_list) == 0:
                return None
            log_features = torch.stack(log_features_list)
        else:

            log_features = _lt_with_counts(log_embeddings, list(range(len(log_ids))))
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['local_transformer'] = time.time() - start_step
            start_step = time.time()

        entity_ids = hetero_data.entity_ids

        entity_contents = getattr(hetero_data, 'entity_contents', None)

        if len(entity_ids) == 0:

            entity_embeddings = torch.zeros(0, self.entity_embedding_dim, device=self.device)
        else:
            entity_embeddings = self.entity_encoder.encode_batch(entity_ids, entity_contents)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['entity_encoding'] = time.time() - start_step
            start_step = time.time()

        x_dict = {
            'log': log_features,
            'entity': entity_embeddings
        }

        edge_index_dict = {}

        if len(log_features) > 0 and len(entity_embeddings) > 0:
            if ('log', 'ASSOCIATED_WITH', 'entity') in hetero_data.edge_types:
                edge_idx = hetero_data[('log', 'ASSOCIATED_WITH', 'entity')].edge_index

                if edge_idx.size(1) > 0:
                    valid_mask = (edge_idx[0] < len(log_features)) & (edge_idx[1] < len(entity_embeddings))
                    if valid_mask.any():
                        edge_index_dict[('log', 'ASSOCIATED_WITH', 'entity')] = edge_idx[:, valid_mask]
            
            if ('entity', 'REVERSE_ASSOCIATED_WITH', 'log') in hetero_data.edge_types:
                edge_idx = hetero_data[('entity', 'REVERSE_ASSOCIATED_WITH', 'log')].edge_index

                if edge_idx.size(1) > 0:
                    valid_mask = (edge_idx[0] < len(entity_embeddings)) & (edge_idx[1] < len(log_features))
                    if valid_mask.any():
                        edge_index_dict[('entity', 'REVERSE_ASSOCIATED_WITH', 'log')] = edge_idx[:, valid_mask]

        if len(edge_index_dict) == 0 or len(log_features) == 0 or len(entity_embeddings) == 0:
            self.hgt.last_attention = []

            updated_x_dict = x_dict
        else:

            updated_x_dict = self.hgt(x_dict, edge_index_dict)
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['hgt_message_passing'] = time.time() - start_step
            start_step = time.time()

        if target_window_idx is not None and hasattr(hetero_data, 'window_ids'):

            window_ids = hetero_data.window_ids
            if target_window_idx < len(window_ids):

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

                        window_embedding = torch.zeros(self.hgt.hidden_dim, device=self.device)
                else:
                    window_embedding = self.attention_pooling(updated_x_dict['log'])
            else:
                window_embedding = self.attention_pooling(updated_x_dict['log'])
        else:

            window_embedding = self.attention_pooling(updated_x_dict['log'])
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['attention_pooling'] = time.time() - start_step
            start_step = time.time()

        window_embedding = window_embedding.unsqueeze(0)
        logits = self.classifier(window_embedding)
        logits = logits.squeeze(0)
        self.last_diagnostics = {
            'target_window_id': getattr(hetero_data, 'target_window_id', None),
            'selected_windows': list(getattr(hetero_data, 'window_ids', [])),
            'probabilities': torch.softmax(logits.detach(), dim=-1).cpu().tolist(),
            'target_log_ids': [log_ids[i] for i in target_log_indices] if 'target_log_indices' in locals() else list(log_ids),
            'target_log_attention': self.attention_pooling.last_weights.cpu().tolist(),
            'log_entity_attention': getattr(self.hgt, 'last_attention', []),
        }
        
        if profile_timing:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['classification'] = time.time() - start_step
            timings['total'] = time.time() - start_total

            if not hasattr(hetero_data, '_timings'):
                hetero_data._timings = timings
        
        return logits

