#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的实体提取器
在原有实体提取基础上，集成统计实体挖掘（Token Complexity和Recurrence Frequency）
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import sys
from pathlib import Path

# 导入原有的实体提取器
sys.path.insert(0, str(Path(__file__).parent))
from entity_extractor import EntityExtractor, LogEntities as BaseLogEntities


@dataclass
class EnhancedLogEntities(BaseLogEntities):
    """增强的日志实体信息，包含统计实体"""
    # 原有实体（继承自BaseLogEntities）
    # data_regions, nodes, node_ips, thread, consensus_groups
    
    # 新增：统计实体（包含字母和数字的字符串）
    statistical_entities: Set[str] = field(default_factory=set)  # 统计实体集合
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        base_dict = super().to_dict()
        base_dict.update({
            "statistical_entities": sorted(list(self.statistical_entities)),
        })
        return base_dict
    
    def merge(self, other: 'EnhancedLogEntities'):
        """合并另一个EnhancedLogEntities的信息"""
        super().merge(other)
        self.statistical_entities.update(other.statistical_entities)


class StatisticalEntityMiner:
    """统计实体挖掘器（基于Token Complexity和Recurrence Frequency）"""
    
    def __init__(self, tc_threshold: int = 2, min_rf: int = 5):
        """
        初始化挖掘器
        
        Args:
            tc_threshold: Token Complexity阈值（分段数量），默认2
            min_rf: 最小Recurrence Frequency，默认5
        """
        self.tc_threshold = tc_threshold
        self.min_rf = min_rf
        
        # Token提取模式：提取非自然语言单词
        # 匹配：包含数字、下划线、连字符等的token
        self.token_patterns = [
            # 包含数字和字母/下划线的token
            r'\b[a-zA-Z_]+[0-9]+[a-zA-Z0-9_]*\b',  # blk_123, file123
            r'\b[0-9]+[a-zA-Z_]+[a-zA-Z0-9_]*\b',  # 123blk, 2024file
            # 纯数字ID（长度>=10，可能是时间戳或ID）
            r'\b\d{10,}\b',  # 1705029356052
            # 包含特殊字符的token
            r'\b[a-zA-Z0-9_]+\[[0-9]+\]\b',  # DataRegion[1]
            r'\b[a-zA-Z0-9_]+=[0-9]+\b',  # nodeId=1
            r'\b[a-zA-Z0-9_]+:[0-9]+\b',  # id:123
            # 文件路径中的文件名部分
            r'\b[a-zA-Z0-9_-]+\.(?:tsfile|wal|log|txt|conf)\b',  # file.tsfile
            # UUID格式
            r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.token_patterns]
        
        # 全局统计：用于计算RF
        self.token_frequency = Counter()  # token -> 出现次数
        self.valid_tokens_cache = None  # 缓存有效的token集合
    
    def calculate_token_complexity(self, token: str) -> int:
        """
        计算Token Complexity (TC)
        
        TC定义为令牌中连续字母、数字或符号段的数量。
        例如："blk_3587508140" 可分割为 "blk"、"_"、"3587508140" 三个段，因此TC值为3。
        
        Args:
            token: token字符串
            
        Returns:
            int: TC值（分段数量）
        """
        if not token:
            return 0
        
        # 使用正则表达式分割token为连续的同类型字符段
        # 匹配连续的字母、数字或符号
        segments = re.findall(r'[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9]+', token)
        
        # TC值等于分段数量
        return len(segments)
    
    def extract_tokens_from_message(self, message: str) -> Set[str]:
        """
        从日志消息中提取token
        
        Args:
            message: 日志消息文本
            
        Returns:
            Set[str]: 提取的token集合
        """
        tokens = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(message)
            tokens.update(matches)
        
        # 过滤掉太短的token（<3个字符）
        tokens = {t for t in tokens if len(t) >= 3}
        
        # 过滤掉纯数字且长度<10的token（可能是普通数字）
        tokens = {t for t in tokens if not (t.isdigit() and len(t) < 10)}
        
        # 过滤掉时间戳格式的ID（如 20240112_031433_11379_1）
        # 模式：YYYYMMDD_HHMMSS_数字_数字
        timestamp_id_pattern = re.compile(r'^\d{8}_\d{6}_\d+_\d+$')
        tokens = {t for t in tokens if not timestamp_id_pattern.match(t)}
        
        # 过滤掉纯时间戳格式（如 20240112031433）
        pure_timestamp_pattern = re.compile(r'^\d{14,}$')
        tokens = {t for t in tokens if not pure_timestamp_pattern.match(t)}
        
        return tokens
    
    def update_token_frequency(self, tokens: Set[str]):
        """更新token频率统计"""
        for token in tokens:
            self.token_frequency[token] += 1
    
    def compute_valid_tokens(self) -> Set[str]:
        """
        计算有效的token集合（TC >= threshold 且 RF >= min_rf）
        
        Returns:
            Set[str]: 有效的token集合
        """
        if self.valid_tokens_cache is not None:
            return self.valid_tokens_cache
        
        valid_tokens = set()
        for token, frequency in self.token_frequency.items():
            tc = self.calculate_token_complexity(token)
            if tc >= self.tc_threshold and frequency >= self.min_rf:
                valid_tokens.add(token)
        
        self.valid_tokens_cache = valid_tokens
        return valid_tokens
    
    def extract_statistical_entities(self, message: str, valid_tokens: Optional[Set[str]] = None) -> Set[str]:
        """
        从日志消息中提取统计实体
        
        Args:
            message: 日志消息文本
            valid_tokens: 有效的token集合（如果为None，则使用compute_valid_tokens()计算）
            
        Returns:
            Set[str]: 提取的统计实体集合
        """
        if valid_tokens is None:
            valid_tokens = self.compute_valid_tokens()
        
        tokens = self.extract_tokens_from_message(message)
        # 只返回有效的token（满足TC和RF条件）
        return tokens & valid_tokens


class EnhancedEntityExtractor(EntityExtractor):
    """增强的实体提取器（集成统计实体挖掘）"""
    
    def __init__(self, enable_statistical_mining: bool = True, tc_threshold: int = 2, min_rf: int = 5):
        """
        初始化增强实体提取器
        
        Args:
            enable_statistical_mining: 是否启用统计实体挖掘
            tc_threshold: Token Complexity阈值
            min_rf: 最小Recurrence Frequency
        """
        super().__init__()
        self.enable_statistical_mining = enable_statistical_mining
        self.statistical_miner = StatisticalEntityMiner(tc_threshold=tc_threshold, min_rf=min_rf) if enable_statistical_mining else None
        
        # 用于批量处理时的token频率统计
        self._batch_mode = False
        self._batch_tokens = []
    
    def start_batch_mode(self):
        """开始批量模式（用于先统计token频率，再提取实体）"""
        self._batch_mode = True
        self._batch_tokens = []
        if self.statistical_miner:
            self.statistical_miner.token_frequency.clear()
            self.statistical_miner.valid_tokens_cache = None
    
    def process_batch_message(self, message: str):
        """批量处理消息（仅统计token频率，不提取实体）"""
        if self._batch_mode and self.statistical_miner:
            tokens = self.statistical_miner.extract_tokens_from_message(message)
            self.statistical_miner.update_token_frequency(tokens)
            self._batch_tokens.append((message, tokens))
    
    def finish_batch_mode(self):
        """完成批量模式（计算有效token集合）"""
        if self._batch_mode and self.statistical_miner:
            self.statistical_miner.compute_valid_tokens()
        self._batch_mode = False
    
    def extract_from_log_line(self, log_line: str) -> EnhancedLogEntities:
        """
        从日志行中提取所有实体信息（包括统计实体）
        
        Args:
            log_line: 日志行文本
            
        Returns:
            EnhancedLogEntities: 包含所有实体信息的对象
        """
        # 先调用父类方法提取原有实体
        base_entities = super().extract_from_log_line(log_line)
        
        # 创建增强实体对象
        entities = EnhancedLogEntities(
            data_regions=base_entities.data_regions,
            nodes=base_entities.nodes,
            node_ips=base_entities.node_ips,
            thread=base_entities.thread,
            consensus_groups=base_entities.consensus_groups
        )
        
        # 提取统计实体（如果启用）
        if self.enable_statistical_mining and self.statistical_miner:
            # 提取消息部分（去除时间戳和线程名）
            message_match = re.search(r'\]\s+[A-Z]+\s+(.+)$', log_line)
            if message_match:
                message = message_match.group(1)
            else:
                message = log_line
            
            # 如果不在批量模式，需要先更新频率
            if not self._batch_mode:
                tokens = self.statistical_miner.extract_tokens_from_message(message)
                self.statistical_miner.update_token_frequency(tokens)
                # 计算有效token（可能需要重新计算）
                valid_tokens = self.statistical_miner.compute_valid_tokens()
                entities.statistical_entities = self.statistical_miner.extract_statistical_entities(message, valid_tokens)
            else:
                # 批量模式：只提取token，不更新频率（频率已在process_batch_message中更新）
                valid_tokens = self.statistical_miner.compute_valid_tokens()
                entities.statistical_entities = self.statistical_miner.extract_statistical_entities(message, valid_tokens)
        
        return entities
    
    def extract_from_message(self, message: str, thread: Optional[str] = None) -> EnhancedLogEntities:
        """
        从日志消息（不包含时间戳和线程信息的纯消息）中提取实体
        
        Args:
            message: 日志消息内容
            thread: 可选的线程名（如果已知）
            
        Returns:
            EnhancedLogEntities: 提取到的实体信息
        """
        # 先调用父类方法
        base_entities = super().extract_from_message(message, thread)
        
        # 创建增强实体对象
        entities = EnhancedLogEntities(
            data_regions=base_entities.data_regions,
            nodes=base_entities.nodes,
            node_ips=base_entities.node_ips,
            thread=base_entities.thread,
            consensus_groups=base_entities.consensus_groups
        )
        
        # 提取统计实体（如果启用）
        if self.enable_statistical_mining and self.statistical_miner:
            if not self._batch_mode:
                tokens = self.statistical_miner.extract_tokens_from_message(message)
                self.statistical_miner.update_token_frequency(tokens)
                valid_tokens = self.statistical_miner.compute_valid_tokens()
                entities.statistical_entities = self.statistical_miner.extract_statistical_entities(message, valid_tokens)
            else:
                valid_tokens = self.statistical_miner.compute_valid_tokens()
                entities.statistical_entities = self.statistical_miner.extract_statistical_entities(message, valid_tokens)
        
        return entities


# 便捷函数
def extract_enhanced_entities_from_log_line(log_line: str, enable_statistical_mining: bool = True) -> EnhancedLogEntities:
    """
    便捷函数：从日志行提取增强实体
    
    Args:
        log_line: 完整的日志行
        enable_statistical_mining: 是否启用统计实体挖掘
        
    Returns:
        EnhancedLogEntities: 提取到的实体信息
    """
    extractor = EnhancedEntityExtractor(enable_statistical_mining=enable_statistical_mining)
    return extractor.extract_from_log_line(log_line)


if __name__ == "__main__":
    # 测试代码
    test_logs = [
        "- 2024-01-12 03:09:34,967 [pool-31-IoTDB-LogDispatcher-DataRegion[1]-1] INFO org.apache.iotdb.consensus.iot.logdispatcher.LogDispatcher$LogDispatcherThread:360 - DataRegion[1]: startIndex: 1, maxIndex: 1, pendingEntries size: 0, bufferedEntries size: 0",
        "2024-01-12 03:09:34,965 [pool-31-IoTDB-LogDispatcher-DataRegion[1]-2] INFO org.apache.iotdb.consensus.iot.logdispatcher.LogDispatcher$LogDispatcherThread:308 - Peer{groupId=DataRegion[1], endpoint=TEndPoint(ip:172.20.0.11, port:10760), nodeId=1}: Dispatcher starts",
    ]
    
    print("=== 增强实体提取测试 ===\n")
    
    # 测试1：批量模式（先统计频率，再提取）
    print("测试1：批量模式")
    extractor = EnhancedEntityExtractor(enable_statistical_mining=True, tc_threshold=2, min_rf=1)
    extractor.start_batch_mode()
    
    for log_line in test_logs:
        extractor.process_batch_message(log_line)
    
    extractor.finish_batch_mode()
    
    for i, log_line in enumerate(test_logs, 1):
        print(f"\n日志 {i}:")
        print(f"  {log_line[:100]}...")
        entities = extractor.extract_from_log_line(log_line)
        print(f"  提取结果:")
        print(f"    DataRegions: {entities.data_regions}")
        print(f"    Nodes: {entities.nodes}")
        print(f"    Thread: {entities.thread}")
        print(f"    ConsensusGroups: {entities.consensus_groups}")
        print(f"    统计实体: {entities.statistical_entities}")
