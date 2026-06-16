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
        # 修复transformers检测PyTorch的问题（必须在导入transformers之前）
        import torch
        import transformers.utils.import_utils as import_utils
        
        # 强制设置PyTorch可用（transformers 4.x使用_torch_available）
        import_utils._torch_available = True
        import_utils._PYTORCH_AVAILABLE = True
        
        from transformers import BertModel, BertTokenizer
        
        self.device = device
        self.model_name = model_name
        self.precomputed_embeddings = precomputed_embeddings
        
        # 如果提供了预编码向量，就不需要加载BERT模型
        if precomputed_embeddings is not None:
            print(f"模板编码器初始化: {model_name} (使用预编码向量)")
            print(f"  预编码向量数量: {len(precomputed_embeddings)}")
            # 性能优化：批量 .to(device)，避免对 877k+ 向量逐个同步 CUDA
            # （逐个 .to() 每个 ~0.5ms，百万级向量要 8-15 分钟；批量 ~ 几秒）
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
        
        # 设置环境变量，强制使用本地文件，避免网络连接
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 加载BERT模型和分词器（优先使用本地缓存）
        try:
            # 先尝试使用本地文件
            self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = BertModel.from_pretrained(model_name, local_files_only=True).to(device)
            print("  ✅ 从本地缓存加载成功")
        except Exception as e:
            # 如果本地没有，尝试下载（使用镜像）
            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            os.environ['HF_HUB_OFFLINE'] = '0'
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print(f"  本地缓存未找到或加载失败: {str(e)[:100]}")
            print("  尝试从镜像下载...")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(device)
            print("  ✅ 从镜像下载成功")
        
        self.model.eval()
        
        # BERT的嵌入维度
        self.embedding_dim = self.model.config.hidden_size  # 768 for bert-base-uncased
        
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
            # 使用BERT编码
            inputs = self.tokenizer(
                template_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # 使用[CLS] token的嵌入
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (embedding_dim,)
        
        return embedding
    
    def encode_batch(self, template_texts: List[str]) -> torch.Tensor:
        """
        批量编码模板
        
        Args:
            template_texts: 模板文本列表
            
        Returns:
            torch.Tensor: 嵌入矩阵 (batch_size, embedding_dim)
        """
        # 如果使用预编码向量，直接从字典中查找
        if self.precomputed_embeddings is not None:
            embeddings = []
            missing_count = 0
            for template_text in template_texts:
                if template_text in self.precomputed_embeddings:
                    embeddings.append(self.precomputed_embeddings[template_text])
                else:
                    # 如果模板不在预编码字典中，使用零向量
                    missing_count += 1
                    embeddings.append(torch.zeros(self.embedding_dim, device=self.device))
            # 只在有缺失时警告一次
            if missing_count > 0:
                import warnings
                warnings.warn(f"有 {missing_count} 个模板不在预编码字典中，使用零向量", UserWarning)
            return torch.stack(embeddings)
        
        # 否则使用BERT模型实时编码
        with torch.no_grad():
            # 批量编码
            inputs = self.tokenizer(
                template_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # 使用[CLS] token的嵌入
            embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, embedding_dim)
        
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
        
        # ID类实体使用可学习的嵌入
        self.id_embedding = nn.Embedding(len(entity_vocab), embedding_dim).to(device)
        
        # 如果有BERT，用于语义实体
        if use_bert:
            from transformers import BertModel, BertTokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertModel.from_pretrained(bert_model_name).to(device)
            self.bert_model.eval()
            # 投影层：BERT输出维度 -> embedding_dim
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
        # Thread实体使用Embedding
        if entity_id.startswith('Thread:'):
            return False
        
        # GeneralEntity中的ID类使用Embedding
        if entity_id.startswith('GeneralEntity:'):
            entity_name = entity_id.split(':', 1)[-1]
            # 简单的启发式规则：如果包含数字和符号，可能是ID类
            if ':' in entity_name or '=' in entity_name:
                # 如 id:1, code:200, size=1
                return False
            # 其他可能有语义
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
            # 未知实体，返回零向量
            return torch.zeros(self.embedding_dim, device=self.device)
        
        entity_idx = self.entity_vocab[entity_id]
        
        # 确定要编码的文本内容
        if entity_content:
            entity_text = entity_content
        else:
            # 如果没有提供内容，从entity_id中提取
            entity_text = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
        
        if self.use_bert and self._is_semantic_entity(entity_id):
            # 使用BERT编码实体的实际内容
            with torch.no_grad():
                inputs = self.bert_tokenizer(
                    entity_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=32
                ).to(self.device)
                outputs = self.bert_model(**inputs)
                # 使用[CLS] token的嵌入
                bert_emb = outputs.last_hidden_state[:, 0, :]  # (1, bert_dim)
                # 投影到目标维度
                emb = self.bert_projection(bert_emb).squeeze(0)  # (embedding_dim,)
        else:
            # 使用可学习的嵌入（基于entity_id的索引）
            emb = self.id_embedding(torch.tensor(entity_idx, device=self.device))
        
        return emb
    
    def encode_batch(self, entity_ids: List[str], entity_contents: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """
        批量编码实体（性能优化：单次 embedding lookup 替代 Python 循环）

        Args:
            entity_ids: 实体ID列表
            entity_contents: 实体内容字典 {entity_id: entity_content}（仅 BERT 路径用）

        Returns:
            torch.Tensor: 嵌入矩阵 (batch_size, embedding_dim)
        """
        if not entity_ids:
            return torch.zeros(0, self.embedding_dim, device=self.device)

        # 主路径（非 BERT）：单次 embedding lookup
        if not self.use_bert:
            indices = []
            unknown_positions = []
            for i, eid in enumerate(entity_ids):
                if eid in self.entity_vocab:
                    indices.append(self.entity_vocab[eid])
                else:
                    indices.append(0)  # placeholder, 后面清零
                    unknown_positions.append(i)
            idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
            embs = self.id_embedding(idx_tensor)
            if unknown_positions:
                # 未知实体置零
                embs = embs.clone()
                embs[unknown_positions] = 0.0
            return embs

        # BERT 路径：分流 semantic / non-semantic（不常用，保留原逻辑）
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
        # 编码模板
        if template_text:
            template_emb = self.template_encoder.encode_template(template_text)
        else:
            # 使用template_id作为文本
            template_emb = self.template_encoder.encode_template(template_id)
        
        # 目前只使用模板嵌入，实体信息通过图结构传递
        # 后续可以通过拼接或其他方式融合实体信息
        return template_emb

