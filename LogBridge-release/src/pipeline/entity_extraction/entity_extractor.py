#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志实体提取模块
从IoTDB日志中提取DataRegion、Node、Thread、ConsensusGroup等实体信息
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class LogEntities:
    """日志中提取的实体信息"""
    data_regions: Set[str] = field(default_factory=set)  # DataRegion ID集合，如 {"1", "2"}
    nodes: Set[str] = field(default_factory=set)  # Node ID集合，如 {"1", "2", "3"}
    node_ips: Dict[str, str] = field(default_factory=dict)  # Node ID -> IP地址映射
    thread: Optional[str] = None  # 线程名，如 "pool-34-IoTDB-LogDispatcher-DataRegion[2]-1"
    consensus_groups: Set[str] = field(default_factory=set)  # ConsensusGroup ID集合，如 {"DataRegion:1", "DataRegion:2"}
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "data_regions": sorted(list(self.data_regions)),
            "nodes": sorted(list(self.nodes)),
            "node_ips": self.node_ips,
            "thread": self.thread,
            "consensus_groups": sorted(list(self.consensus_groups)),
        }
    
    def merge(self, other: 'LogEntities'):
        """合并另一个LogEntities的信息"""
        self.data_regions.update(other.data_regions)
        self.nodes.update(other.nodes)
        self.node_ips.update(other.node_ips)
        self.consensus_groups.update(other.consensus_groups)
        # thread只保留第一个（通常同一行日志只有一个线程）


class EntityExtractor:
    """实体提取器"""
    
    def __init__(self):
        # DataRegion匹配模式
        # 匹配: DataRegion[1], DataRegion[2], root.sg1[1], root.sg1[2] 等
        self.data_region_pattern = re.compile(
            r'(?:DataRegion|root\.sg\d+)\[(\d+)\]',
            re.IGNORECASE
        )
        
        # Node ID匹配模式
        # 匹配: nodeId=1, nodeId=2 等
        self.node_id_pattern = re.compile(
            r'nodeId\s*=\s*(\d+)',
            re.IGNORECASE
        )
        
        # IP地址匹配模式
        # 匹配: 172.20.0.11, 172.20.0.12 等
        self.ip_pattern = re.compile(
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        )
        
        # TEndPoint匹配模式
        # 匹配: TEndPoint(ip:172.20.0.11, port:10760)
        self.endpoint_pattern = re.compile(
            r'TEndPoint\s*\(\s*ip\s*:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s*,\s*port\s*:\s*(\d+)\s*\)',
            re.IGNORECASE
        )
        
        # Peer匹配模式（包含nodeId和endpoint）
        # 匹配: Peer{groupId=DataRegion[1], endpoint=TEndPoint(ip:172.20.0.11, port:10760), nodeId=1}
        self.peer_pattern = re.compile(
            r'Peer\s*\{\s*groupId\s*=\s*[^,]+,\s*endpoint\s*=\s*TEndPoint\s*\(\s*ip\s*:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s*,\s*port\s*:\s*(\d+)\s*\)\s*,\s*nodeId\s*=\s*(\d+)\s*\}',
            re.IGNORECASE
        )
        
        # 线程名匹配模式（从日志行格式中提取）
        # 日志格式: TIMESTAMP [thread_name] LEVEL ... 或 - TIMESTAMP [thread_name] LEVEL ...
        # 注意：线程名可能包含内部的[]，如 pool-31-IoTDB-LogDispatcher-DataRegion[1]-1
        # 策略：找到时间戳后的第一个[，然后找到对应的]，这个]后面应该跟着空白+日志级别
        # 使用 [^\]]+ 来匹配线程名，避免贪婪匹配导致的问题
        self.thread_pattern = re.compile(
            r'^\s*-?\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+\[([^\]]+)\]\s+[A-Z]+\s+'
        )
        
        # ConsensusGroup匹配模式
        # 匹配: TConsensusGroupId(type:DataRegion, id:1)
        self.consensus_group_pattern = re.compile(
            r'TConsensusGroupId\s*\(\s*type\s*:\s*(\w+)\s*,\s*id\s*:\s*(\d+)\s*\)',
            re.IGNORECASE
        )
        
        # 通信模式：DataRegion[X]->IP
        # 匹配: DataRegion[2]->172.20.0.12
        self.communication_pattern = re.compile(
            r'DataRegion\[(\d+)\]\s*->\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        )
    
    def extract_from_log_line(self, log_line: str) -> LogEntities:
        """
        从单行日志中提取实体信息
        
        Args:
            log_line: 完整的日志行（包含时间戳、线程等信息）
        
        Returns:
            LogEntities: 提取到的实体信息
        """
        entities = LogEntities()
        
        # 1. 提取线程名（从日志行格式中提取第一个[]内的内容）
        thread_match = self.thread_pattern.search(log_line)
        if thread_match:
            entities.thread = thread_match.group(1)
        
        # 2. 提取DataRegion
        data_region_matches = self.data_region_pattern.findall(log_line)
        for region_id in data_region_matches:
            entities.data_regions.add(region_id)
        
        # 3. 提取Node信息（优先从Peer模式中提取，包含nodeId、IP和port的完整信息）
        # 先匹配Peer模式（包含完整信息）
        peer_matches = self.peer_pattern.finditer(log_line)
        for peer_match in peer_matches:
            ip = peer_match.group(1)
            port = peer_match.group(2)
            node_id = peer_match.group(3)
            entities.nodes.add(node_id)
            entities.node_ips[node_id] = ip
        
        # 再匹配单独的nodeId（如果没有在Peer中找到）
        node_id_matches = self.node_id_pattern.finditer(log_line)
        for node_match in node_id_matches:
            node_id = node_match.group(1)
            entities.nodes.add(node_id)
            
            # 尝试从同一行中找到对应的IP地址
            if node_id not in entities.node_ips:
                # 查找附近的TEndPoint
                endpoint_match = self.endpoint_pattern.search(log_line)
                if endpoint_match:
                    ip = endpoint_match.group(1)
                    entities.node_ips[node_id] = ip
        
        # 如果没有找到nodeId但找到了TEndPoint，尝试从IP推断或单独记录
        if not entities.nodes:
            endpoint_match = self.endpoint_pattern.search(log_line)
            if endpoint_match:
                ip = endpoint_match.group(1)
                port = endpoint_match.group(2)
                # 如果日志中有通信模式，可以从那里推断
                comm_match = self.communication_pattern.search(log_line)
                if comm_match:
                    # 这里只记录IP，不创建Node ID（因为没有明确的nodeId）
                    pass
        
        # 4. 提取ConsensusGroup
        consensus_matches = self.consensus_group_pattern.finditer(log_line)
        for consensus_match in consensus_matches:
            group_type = consensus_match.group(1)  # 如 "DataRegion"
            group_id = consensus_match.group(2)  # 如 "1"
            # 组合成统一格式：type:id
            consensus_group_key = f"{group_type}:{group_id}"
            entities.consensus_groups.add(consensus_group_key)
        
        # 5. 从通信模式中提取DataRegion和IP关联
        comm_matches = self.communication_pattern.finditer(log_line)
        for comm_match in comm_matches:
            region_id = comm_match.group(1)
            target_ip = comm_match.group(2)
            entities.data_regions.add(region_id)
            # 这里可以记录通信关系，但暂时只提取实体
        
        return entities
    
    def extract_from_message(self, message: str, thread: Optional[str] = None) -> LogEntities:
        """
        从日志消息（不包含时间戳和线程信息的纯消息）中提取实体
        
        Args:
            message: 日志消息内容
            thread: 可选的线程名（如果已知）
        
        Returns:
            LogEntities: 提取到的实体信息
        """
        entities = LogEntities()
        
        # 如果提供了线程名，直接设置
        if thread:
            entities.thread = thread
        
        # 提取DataRegion
        data_region_matches = self.data_region_pattern.findall(message)
        for region_id in data_region_matches:
            entities.data_regions.add(region_id)
        
        # 提取Node信息
        peer_matches = self.peer_pattern.finditer(message)
        for peer_match in peer_matches:
            ip = peer_match.group(1)
            node_id = peer_match.group(3)
            entities.nodes.add(node_id)
            entities.node_ips[node_id] = ip
        
        node_id_matches = self.node_id_pattern.finditer(message)
        for node_match in node_id_matches:
            node_id = node_match.group(1)
            entities.nodes.add(node_id)
            if node_id not in entities.node_ips:
                endpoint_match = self.endpoint_pattern.search(message)
                if endpoint_match:
                    ip = endpoint_match.group(1)
                    entities.node_ips[node_id] = ip
        
        # 提取ConsensusGroup
        consensus_matches = self.consensus_group_pattern.finditer(message)
        for consensus_match in consensus_matches:
            group_type = consensus_match.group(1)
            group_id = consensus_match.group(2)
            consensus_group_key = f"{group_type}:{group_id}"
            entities.consensus_groups.add(consensus_group_key)
        
        # 提取通信模式
        comm_matches = self.communication_pattern.finditer(message)
        for comm_match in comm_matches:
            region_id = comm_match.group(1)
            entities.data_regions.add(region_id)
        
        return entities


# 全局提取器实例
_extractor = None


def get_entity_extractor() -> EntityExtractor:
    """获取全局实体提取器实例（单例模式）"""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def extract_entities_from_log_line(log_line: str) -> LogEntities:
    """
    便捷函数：从日志行中提取实体
    
    Args:
        log_line: 完整的日志行
    
    Returns:
        LogEntities: 提取到的实体信息
    """
    extractor = get_entity_extractor()
    return extractor.extract_from_log_line(log_line)


def extract_entities_from_message(message: str, thread: Optional[str] = None) -> LogEntities:
    """
    便捷函数：从日志消息中提取实体
    
    Args:
        message: 日志消息内容
        thread: 可选的线程名
    
    Returns:
        LogEntities: 提取到的实体信息
    """
    extractor = get_entity_extractor()
    return extractor.extract_from_message(message, thread)


# 测试代码
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        # 测试1: 标准格式，包含DataRegion和线程
        "- 2024-01-12 03:09:34,967 [pool-31-IoTDB-LogDispatcher-DataRegion[1]-1] INFO org.apache.iotdb.consensus.iot.logdispatcher.LogDispatcher$LogDispatcherThread:360 - DataRegion[1]: startIndex: 1, maxIndex: 1, pendingEntries size: 0, bufferedEntries size: 0",
        
        # 测试2: 包含Peer信息（nodeId和IP）
        "- 2024-01-12 03:09:34,965 [pool-31-IoTDB-LogDispatcher-DataRegion[1]-2] INFO org.apache.iotdb.consensus.iot.logdispatcher.LogDispatcher$LogDispatcherThread:308 - Peer{groupId=DataRegion[1], endpoint=TEndPoint(ip:172.20.0.11, port:10760), nodeId=1}: Dispatcher for Peer{groupId=DataRegion[1], endpoint=TEndPoint(ip:172.20.0.13, port:10760), nodeId=3} starts",
        
        # 测试3: 包含ConsensusGroup
        "- 2024-01-12 03:09:37,619 [pool-34-IoTDB-LogDispatcher-DataRegion[2]-1] INFO org.apache.iotdb.consensus.iot.logdispatcher.LogDispatcher$LogDispatcherThread:456 - Send Batch[startIndex:1, endIndex:1] to ConsensusGroup:TConsensusGroupId(type:DataRegion, id:2)",
        
        # 测试4: 包含通信模式
        "- 2024-01-12 03:09:37,569 [pool-34-IoTDB-LogDispatcher-DataRegion[2]-1] INFO org.apache.iotdb.consensus.iot.logdispatcher.LogDispatcher$LogDispatcherThread:473 - DataRegion[2]->172.20.0.12: currentIndex: 1, maxIndex: 1",
        
        # 测试5: root.sg1格式
        "- 2024-01-12 03:09:34,918 [pool-27-IoTDB-DataNodeInternalRPC-Processor-2] INFO org.apache.iotdb.db.storageengine.dataregion.DataRegion:551 - The data region root.sg1[1] is created successfully",
    ]
    
    extractor = EntityExtractor()
    
    print("=" * 80)
    print("实体提取测试")
    print("=" * 80)
    
    for i, test_line in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"日志行: {test_line[:100]}...")
        
        entities = extractor.extract_from_log_line(test_line)
        print(f"提取结果:")
        print(f"  DataRegions: {sorted(list(entities.data_regions))}")
        print(f"  Nodes: {sorted(list(entities.nodes))}")
        print(f"  Node IPs: {entities.node_ips}")
        print(f"  Thread: {entities.thread}")
        print(f"  ConsensusGroups: {sorted(list(entities.consensus_groups))}")

