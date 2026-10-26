#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码器：模板语义编码和实体编码
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set

class TemplateEncoder:
    """日志模板编码器（使用BERT）"""
    
    def __init__(
        self, 
        model_name: str = 'bert-base-uncased', 
        device: str = 'cuda',
        precomputed_embeddings: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        初始化模板编码器
        
        Args:
            model_name: BERT模型名称
            device: 设备
            precomputed_embeddings: 预编码的向量字典 {template_text: embedding}（可选）
        """

        import torch
        import transformers.utils.import_utils as import_utils

        import_utils._torch_available = True
        import_utils._PYTORCH_AVAILABLE = True
        
        from transformers import BertModel, BertTokenizer
        
        self.device = device
        self.model_name = model_name
        self.precomputed_embeddings = precomputed_embeddings

        if precomputed_embeddings is not None:
            print(f"模板编码器初始化: {model_name} (使用预编码向量)")
            print(f"  预编码向量数量: {len(precomputed_embeddings)}")

            import time as _t
            _t0 = _t.time()
            keys = list(precomputed_embeddings.keys())
            stacked = torch.stack([precomputed_embeddings[k] for k in keys]).to(device, non_blocking=True)
            if device == 'cuda':
                torch.cuda.synchronize()
            self.precomputed_embeddings = {k: stacked[i] for i, k in enumerate(keys)}
            print(f"  ✓ 批量移到 {device}: {_t.time()-_t0:.1f}s")
            self.embedding_dim = stacked.shape[1]
            print(f"  嵌入维度: {self.embedding_dim}")
            self.tokenizer = None
            self.model = None
            return
        
        print(f"模板编码器初始化: {model_name}")
        print("  加载BERT模型和分词器...")

        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'

        try:

            self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = BertModel.from_pretrained(model_name, local_files_only=True).to(device)
            print("  ✅ 从本地缓存加载成功")
        except Exception as e:

            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            os.environ['HF_HUB_OFFLINE'] = '0'
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print(f"  本地缓存未找到或加载失败: {str(e)[:100]}")
            print("  尝试从镜像下载...")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(device)
            print("  ✅ 从镜像下载成功")
        
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        
        print(f"  嵌入维度: {self.embedding_dim}")
    
    def encode_template(self, template_text: str) -> torch.Tensor:
        """
        编码单个模板
        
        Args:
            template_text: 模板文本
            
        Returns:
            torch.Tensor: 嵌入向量 (embedding_dim,)
        """
        with torch.no_grad():

            inputs = self.tokenizer(
                template_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        
        return embedding
    
    def encode_batch(self, template_texts: List[str]) -> torch.Tensor:
        """
        批量编码模板
        
        Args:
            template_texts: 模板文本列表
            
        Returns:
            torch.Tensor: 嵌入矩阵 (batch_size, embedding_dim)
        """

        if self.precomputed_embeddings is not None:
            embeddings = []
            missing_count = 0
            for template_text in template_texts:
                if template_text in self.precomputed_embeddings:
                    embeddings.append(self.precomputed_embeddings[template_text])
                else:

                    missing_count += 1
                    embeddings.append(torch.zeros(self.embedding_dim, device=self.device))

            if missing_count > 0:
                import warnings
                warnings.warn(f"有 {missing_count} 个模板不在预编码字典中，使用零向量", UserWarning)
            return torch.stack(embeddings)

        with torch.no_grad():

            inputs = self.tokenizer(
                template_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings

class EntityEncoder(nn.Module):
    """实体编码器"""
    
    def __init__(
        self,
        entity_vocab: Dict[str, int],
        embedding_dim: int = 128,
        use_bert: bool = False,
        bert_model_name: str = 'bert-base-uncased',
        device: str = 'cuda'
    ):
        """
        初始化实体编码器
        
        Args:
            entity_vocab: 实体词汇表 {entity_id: index}
            embedding_dim: 嵌入维度
            use_bert: 是否使用BERT（对于有语义的实体）
            bert_model_name: BERT模型名称
            device: 设备
        """
        super(EntityEncoder, self).__init__()
        
        self.entity_vocab = entity_vocab
        self.embedding_dim = embedding_dim
        self.use_bert = use_bert
        self.device = device

        self.id_embedding = nn.Embedding(max(1, len(entity_vocab)), embedding_dim).to(device)

        if use_bert:
            from transformers import BertModel, BertTokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertModel.from_pretrained(bert_model_name).to(device)
            self.bert_model.eval()

            bert_dim = self.bert_model.config.hidden_size
            self.bert_projection = nn.Linear(bert_dim, embedding_dim).to(device)
        else:
            self.bert_model = None
            self.bert_tokenizer = None
            self.bert_projection = None
    
    def _is_semantic_entity(self, entity_id: str) -> bool:
        """
        判断实体是否有语义（需要BERT编码）
        
        规则：
        - Thread ID: 无语义，使用Embedding
        - GeneralEntity中的ID类（如id:1, code:200）: 无语义，使用Embedding
        - GeneralEntity中的其他实体: 可能有语义，使用BERT
        
        Args:
            entity_id: 实体ID
            
        Returns:
            bool: 是否有语义
        """

        if entity_id.startswith('Thread:'):
            return False

        if entity_id.startswith('GeneralEntity:'):
            entity_name = entity_id.split(':', 1)[-1]

            if ':' in entity_name or '=' in entity_name:

                return False

            return True
        
        return False
    
    def encode_entity(self, entity_id: str, entity_content: Optional[str] = None) -> torch.Tensor:
        """
        编码单个实体
        
        Args:
            entity_id: 实体ID
            entity_content: 实体的实际内容（如token、名称等），如果提供则使用此内容编码
            
        Returns:
            torch.Tensor: 嵌入向量 (embedding_dim,)
        """
        if entity_id not in self.entity_vocab:

            return torch.zeros(self.embedding_dim, device=self.device)
        
        entity_idx = self.entity_vocab[entity_id]

        if entity_content:
            entity_text = entity_content
        else:

            entity_text = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
        
        if self.use_bert and self._is_semantic_entity(entity_id):

            with torch.no_grad():
                inputs = self.bert_tokenizer(
                    entity_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=32
                ).to(self.device)
                outputs = self.bert_model(**inputs)

                bert_emb = outputs.last_hidden_state[:, 0, :]

                emb = self.bert_projection(bert_emb).squeeze(0)
        else:

            emb = self.id_embedding(torch.tensor(entity_idx, device=self.device))
        
        return emb
    
    def encode_batch(self, entity_ids: List[str], entity_contents: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """Encode batch."""
        if not entity_ids:
            return torch.zeros(0, self.embedding_dim, device=self.device)

        if not self.use_bert:
            indices = []
            unknown_positions = []
            for i, eid in enumerate(entity_ids):
                if eid in self.entity_vocab:
                    indices.append(self.entity_vocab[eid])
                else:
                    indices.append(0)
                    unknown_positions.append(i)
            idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
            embs = self.id_embedding(idx_tensor)
            if unknown_positions:

                embs = embs.clone()
                embs[unknown_positions] = 0.0
            return embs

        embeddings = []
        for entity_id in entity_ids:
            entity_content = entity_contents.get(entity_id) if entity_contents else None
            emb = self.encode_entity(entity_id, entity_content)
            embeddings.append(emb)
        return torch.stack(embeddings)

class LogInstanceEncoder:
    """日志实例编码器（组合模板编码和实体编码）"""
    
    def __init__(
        self,
        template_encoder: TemplateEncoder,
        entity_encoder: EntityEncoder,
        log_embedding_dim: int = 384
    ):
        """
        初始化日志实例编码器
        
        Args:
            template_encoder: 模板编码器
            entity_encoder: 实体编码器
            log_embedding_dim: 日志嵌入维度（模板嵌入维度）
        """
        self.template_encoder = template_encoder
        self.entity_encoder = entity_encoder
        self.log_embedding_dim = log_embedding_dim
    
    def encode_log_instance(
        self,
        template_id: str,
        template_text: Optional[str] = None,
        associated_entities: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        编码单个日志实例
        
        Args:
            template_id: 模板ID
            template_text: 模板文本（如果提供，使用此文本；否则使用template_id）
            associated_entities: 关联的实体ID列表
            
        Returns:
            torch.Tensor: 日志嵌入向量 (log_embedding_dim,)
        """

        if template_text:
            template_emb = self.template_encoder.encode_template(template_text)
        else:

            template_emb = self.template_encoder.encode_template(template_id)

        return template_emb

