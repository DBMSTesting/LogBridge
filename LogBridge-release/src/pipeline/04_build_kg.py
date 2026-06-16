#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于窗口数据构建知识图谱
从已经处理好的窗口数据中提取实体信息，构建知识图谱
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import sys

# 修复导入路径
import sys
from pathlib import Path
import importlib.util

# 新目录结构
script_dir = Path(__file__).resolve().parent  # log_anomaly_diagnosis/src/pipeline
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "entity_extraction"))
project_root = script_dir  # for downstream code that refers to it

# 导入entity_extractor（优先使用enhanced版本）
try:
    from entity_extraction.enhanced_entity_extractor import extract_enhanced_entities_from_log_line, EnhancedLogEntities
    USE_ENHANCED_EXTRACTOR = True
except ImportError:
    try:
        from entity_extraction.entity_extractor import extract_entities_from_log_line
        USE_ENHANCED_EXTRACTOR = False
    except ImportError:
        entity_file = project_root / "entity_extraction" / "entity_extractor.py"
        if entity_file.exists():
            spec = importlib.util.spec_from_file_location("entity_extractor", entity_file)
            entity_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entity_module)
            extract_entities_from_log_line = entity_module.extract_entities_from_log_line
            USE_ENHANCED_EXTRACTOR = False
        else:
            raise ImportError(f"Cannot find entity_extractor module at {entity_file}")

# RealtimeDrainParser不需要（因为窗口数据中已经包含template_id）
# from realtime_drain_parser import RealtimeDrainParser


# ---------------------------------------------------------------------------
# 知识图谱数据结构
# ---------------------------------------------------------------------------

@dataclass
class KGNode:
    """知识图谱节点"""
    node_id: str  # 节点唯一ID
    node_type: str  # 节点类型：DataRegion, Node, Thread, ConsensusGroup, Template, Window
    properties: Dict = field(default_factory=dict)  # 节点属性
    
    def to_dict(self):
        return {
            "id": self.node_id,
            "type": self.node_type,
            "properties": self.properties
        }


@dataclass
class KGEdge:
    """知识图谱边（关系）"""
    source_id: str  # 源节点ID
    target_id: str  # 目标节点ID
    relation_type: str  # 关系类型
    properties: Dict = field(default_factory=dict)  # 边属性（如权重、时间戳等）
    
    def to_dict(self):
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation_type,
            "properties": self.properties
        }


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    nodes: Dict[str, KGNode] = field(default_factory=dict)  # node_id -> Node
    edges: List[KGEdge] = field(default_factory=list)  # 边列表
    
    def add_node(self, node: KGNode):
        """添加节点（如果已存在则合并属性）"""
        if node.node_id in self.nodes:
            # 合并属性
            existing_node = self.nodes[node.node_id]
            existing_node.properties.update(node.properties)
        else:
            self.nodes[node.node_id] = node
    
    def add_edge(self, edge: KGEdge):
        """添加边"""
        self.edges.append(edge)
    
    def to_dict(self):
        """转换为字典格式（便于JSON序列化）"""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "nodes_by_type": self._count_nodes_by_type(),
                "edges_by_type": self._count_edges_by_type()
            }
        }
    
    def _count_nodes_by_type(self):
        """统计各类型节点数量"""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.node_type] += 1
        return dict(counts)
    
    def _count_edges_by_type(self):
        """统计各类型边数量"""
        counts = defaultdict(int)
        for edge in self.edges:
            counts[edge.relation_type] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# 节点ID生成函数
# ---------------------------------------------------------------------------

def get_node_id(node_type: str, value: str) -> str:
    """生成节点唯一ID"""
    return f"{node_type}:{value}"


# ---------------------------------------------------------------------------
# 知识图谱构建函数
# ---------------------------------------------------------------------------

def build_knowledge_graph_from_windows(windows_data: Dict, parser: Optional[object] = None) -> KnowledgeGraph:
    """
    从窗口数据构建知识图谱
    
    节点类型：
    1. Window: 时间窗口节点，ID格式为 "Window:label1:count:1:xxx"
    2. LogInstance: 日志实例节点，ID格式为 "LogInstance:Window:label1:count:1:xxx:0"（每条日志一个节点）
    3. Anomaly: 异常类型节点，ID格式为 "Anomaly:full_memory"
    4. DataRegion: DataRegion节点，ID格式为 "DataRegion:1"
    5. Node: 节点（物理节点/IP），ID格式为 "Node:1"
    6. Thread: 线程节点，ID格式为 "Thread:pool-xxx"
    7. ConsensusGroup: 共识组节点，ID格式为 "ConsensusGroup:DataRegion:1"
    8. GeneralEntity: 通用实体节点（统计挖掘得到的实体）
    
    关系类型：
    1. CONTAINS: Window --[CONTAINS]--> LogInstance（窗口包含日志实例）
    2. ASSOCIATED_WITH: LogInstance --[ASSOCIATED_WITH]--> Entity（日志实例关联实体）
    3. HAS_ANOMALY: Window --[HAS_ANOMALY]--> Anomaly（窗口有异常）
    4. CONTAINS: Window --[CONTAINS]--> Entity（窗口包含实体，带频率属性，用于统计）
    5. BELONGS_TO: DataRegion --[BELONGS_TO]--> ConsensusGroup
    """
    kg = KnowledgeGraph()
    
    print("构建知识图谱...")
    print(f"  处理 {len(windows_data)} 个窗口...")
    
    # 统计信息
    window_count = 0
    log_count = 0
    template_count = 0
    
    for window_key, window_info in windows_data.items():
        window_count += 1
        
        # 1. 创建Window节点
        window_node_id = get_node_id("Window", window_key)
        window_node = KGNode(
            node_id=window_node_id,
            node_type="Window",
            properties={
                "window_key": window_key,
                "label_file": window_info.get("label_file", ""),
                "window_start": window_info.get("window_start", ""),
                "window_end": window_info.get("window_end", ""),
                "log_count": window_info.get("log_count", 0)
            }
        )
        kg.add_node(window_node)
        
        # 2. 为每条日志创建LogInstance节点（与iotbench格式一致）
        logs = window_info.get("logs", [])
        log_count += len(logs)
        total_logs = len(logs)
        
        # 为每条日志创建LogInstance节点，并记录每个LogInstance包含的实体
        log_instance_ids = []
        log_instance_entities = {}  # log_instance_id -> {entity_type: [entity_ids]}
        
        for log_idx, log_entry in enumerate(logs):
            # 生成唯一的LogInstance ID（格式：LogInstance:{window_key}_{log_idx}，与iotbench一致）
            log_instance_id = f"LogInstance:{window_key}_{log_idx}"
            log_instance_ids.append(log_instance_id)
            
            # 获取模板ID
            template_id = log_entry.get("template_id")
            if not template_id and parser is not None:
                # 如果没有template_id且提供了parser，实时解析
                raw_line = log_entry.get("raw_line", "")
                if raw_line:
                    result = parser.parse_log(raw_line)
                    cluster_id = result.get('cluster_id', -1)
                    if cluster_id >= 0:
                        template_id = f"cluster_{cluster_id}"
            
            # 创建LogInstance节点
            log_instance_node = KGNode(
                node_id=log_instance_id,
                node_type="LogInstance",
                properties={
                    "window_key": window_key,
                    "template_id": template_id,
                    "timestamp": log_entry.get("timestamp", ""),
                    "raw_line": log_entry.get("raw_line", "")[:200] if log_entry.get("raw_line") else ""  # 只保存前200字符
                }
            )
            kg.add_node(log_instance_node)
            
            # Window --[CONTAINS]--> LogInstance
            kg.add_edge(KGEdge(
                source_id=window_node_id,
                target_id=log_instance_id,
                relation_type="CONTAINS",
                properties={}
            ))
            
            # 记录该LogInstance包含的实体（用于后续创建ASSOCIATED_WITH边）
            log_instance_entities[log_instance_id] = {
                'DataRegion': [],
                'Node': [],
                'Thread': None,
                'ConsensusGroup': [],
                'GeneralEntity': []
            }
        
        # 3. 创建Anomaly节点
        anomaly_types = window_info.get("anomaly_types", [])
        for anomaly_type in anomaly_types:
            anomaly_node_id = get_node_id("Anomaly", anomaly_type)
            anomaly_node = KGNode(
                node_id=anomaly_node_id,
                node_type="Anomaly",
                properties={"anomaly_type": anomaly_type}
            )
            kg.add_node(anomaly_node)
            
            # Window --[HAS_ANOMALY]--> Anomaly
            kg.add_edge(KGEdge(
                source_id=window_node_id,
                target_id=anomaly_node_id,
                relation_type="HAS_ANOMALY",
                properties={}
            ))
        
        # 4. 处理窗口内的日志，统计实体出现频率
        
        # 统计实体在窗口中出现的日志数量（频率统计）
        # 使用dict记录每个实体出现在多少条日志中
        entity_log_counts = {
            'DataRegion': defaultdict(int),  # dr_id -> 出现的日志数量
            'Node': defaultdict(int),        # node_id -> 出现的日志数量
            'Thread': defaultdict(int),      # thread_name -> 出现的日志数量
            'ConsensusGroup': defaultdict(int),  # cg_str -> 出现的日志数量
            'StatisticalEntity': defaultdict(int),  # statistical_entity -> 出现的日志数量
        }
        
        # 用于记录Node的IP信息
        node_ips = {}
        
        # 遍历所有日志，统计实体出现的日志数量，并创建ASSOCIATED_WITH边
        for log_idx, log_entry in enumerate(logs):
            log_instance_id = log_instance_ids[log_idx]
            
            # 优先使用已提取的实体信息（如果存在）
            entities_dict = log_entry.get("entities")
            if entities_dict:
                # 从已提取的实体信息中恢复
                if USE_ENHANCED_EXTRACTOR:
                    entities = EnhancedLogEntities(
                        data_regions=set(entities_dict.get("data_regions", [])),
                        nodes=set(entities_dict.get("nodes", [])),
                        node_ips=entities_dict.get("node_ips", {}),
                        thread=entities_dict.get("thread"),
                        consensus_groups=set(entities_dict.get("consensus_groups", [])),
                        statistical_entities=set(entities_dict.get("statistical_entities", []))
                    )
                else:
                    # 使用基础实体类型
                    from entity_extraction.entity_extractor import LogEntities
                    entities = LogEntities(
                        data_regions=set(entities_dict.get("data_regions", [])),
                        nodes=set(entities_dict.get("nodes", [])),
                        node_ips=entities_dict.get("node_ips", {}),
                        thread=entities_dict.get("thread"),
                        consensus_groups=set(entities_dict.get("consensus_groups", []))
                    )
            else:
                # 如果没有已提取的实体信息，实时提取
                raw_line = log_entry.get("raw_line", "")
                if not raw_line:
                    continue
                
                if USE_ENHANCED_EXTRACTOR:
                    entities = extract_enhanced_entities_from_log_line(raw_line, enable_statistical_mining=True)
                else:
                    entities = extract_entities_from_log_line(raw_line)
            
            # 统计实体出现的日志数量，并记录该LogInstance包含的实体
            for dr_id in entities.data_regions:
                entity_log_counts['DataRegion'][dr_id] += 1
                log_instance_entities[log_instance_id]['DataRegion'].append(dr_id)
            
            for node_id in entities.nodes:
                entity_log_counts['Node'][node_id] += 1
                if node_id not in node_ips and entities.node_ips:
                    node_ips[node_id] = entities.node_ips.get(node_id, "")
                log_instance_entities[log_instance_id]['Node'].append(node_id)
            
            if entities.thread:
                entity_log_counts['Thread'][entities.thread] += 1
                log_instance_entities[log_instance_id]['Thread'] = entities.thread
            
            for cg_str in entities.consensus_groups:
                entity_log_counts['ConsensusGroup'][cg_str] += 1
                log_instance_entities[log_instance_id]['ConsensusGroup'].append(cg_str)
            
            if USE_ENHANCED_EXTRACTOR and hasattr(entities, 'statistical_entities'):
                for stat_entity in entities.statistical_entities:
                    entity_log_counts['StatisticalEntity'][stat_entity] += 1
                    log_instance_entities[log_instance_id]['GeneralEntity'].append(stat_entity)
        
        # 3. 创建DataRegion节点并建立带频率的边和ASSOCIATED_WITH边
        for dr_id, count in entity_log_counts['DataRegion'].items():
            dr_node_id = get_node_id("DataRegion", dr_id)
            dr_node = KGNode(
                node_id=dr_node_id,
                node_type="DataRegion",
                properties={"id": dr_id}
            )
            kg.add_node(dr_node)
            
            # 计算频率
            percentage = (count / total_logs * 100) if total_logs > 0 else 0.0
            frequency_level = "high" if percentage > 50 else "medium" if percentage > 10 else "low"
            
            # Window包含DataRegion（带频率属性）
            kg.add_edge(KGEdge(
                source_id=window_node_id,
                target_id=dr_node_id,
                relation_type="CONTAINS",
                properties={
                    "entity_type": "DataRegion",
                    "count": count,  # 出现在多少条日志中
                    "percentage": round(percentage, 2),  # 占比
                    "frequency_level": frequency_level
                }
            ))
            
            # 创建ASSOCIATED_WITH边：只有包含该DataRegion的LogInstance才连接
            for log_instance_id, entities_dict in log_instance_entities.items():
                if dr_id in entities_dict['DataRegion']:
                    kg.add_edge(KGEdge(
                        source_id=log_instance_id,
                        target_id=dr_node_id,
                        relation_type="ASSOCIATED_WITH",
                        properties={"entity_type": "DataRegion"}
                    ))
        
        # 4. 创建Node节点并建立带频率的边和ASSOCIATED_WITH边
        for node_id, count in entity_log_counts['Node'].items():
            node_node_id = get_node_id("Node", node_id)
            node_node = KGNode(
                node_id=node_node_id,
                node_type="Node",
                properties={
                    "id": node_id,
                    "ip": node_ips.get(node_id, "")
                }
            )
            kg.add_node(node_node)
            
            # 计算频率
            percentage = (count / total_logs * 100) if total_logs > 0 else 0.0
            frequency_level = "high" if percentage > 50 else "medium" if percentage > 10 else "low"
            
            # Window包含Node（带频率属性）
            kg.add_edge(KGEdge(
                source_id=window_node_id,
                target_id=node_node_id,
                relation_type="CONTAINS",
                properties={
                    "entity_type": "Node",
                    "count": count,
                    "percentage": round(percentage, 2),
                    "frequency_level": frequency_level
                }
            ))
            
            # 创建ASSOCIATED_WITH边：只有包含该Node的LogInstance才连接
            for log_instance_id, entities_dict in log_instance_entities.items():
                if node_id in entities_dict['Node']:
                    kg.add_edge(KGEdge(
                        source_id=log_instance_id,
                        target_id=node_node_id,
                        relation_type="ASSOCIATED_WITH",
                        properties={"entity_type": "Node"}
                    ))
        
        # 5. 创建Thread节点并建立带频率的边和ASSOCIATED_WITH边
        for thread_name, count in entity_log_counts['Thread'].items():
            thread_node_id = get_node_id("Thread", thread_name)
            thread_node = KGNode(
                node_id=thread_node_id,
                node_type="Thread",
                properties={"name": thread_name}
            )
            kg.add_node(thread_node)
            
            # 计算频率
            percentage = (count / total_logs * 100) if total_logs > 0 else 0.0
            frequency_level = "high" if percentage > 50 else "medium" if percentage > 10 else "low"
            
            # Window包含Thread（带频率属性）
            kg.add_edge(KGEdge(
                source_id=window_node_id,
                target_id=thread_node_id,
                relation_type="CONTAINS",
                properties={
                    "entity_type": "Thread",
                    "count": count,
                    "percentage": round(percentage, 2),
                    "frequency_level": frequency_level
                }
            ))
            
            # 创建ASSOCIATED_WITH边：只有包含该Thread的LogInstance才连接
            for log_instance_id, entities_dict in log_instance_entities.items():
                if entities_dict['Thread'] == thread_name:
                    kg.add_edge(KGEdge(
                        source_id=log_instance_id,
                        target_id=thread_node_id,
                        relation_type="ASSOCIATED_WITH",
                        properties={"entity_type": "Thread"}
                    ))
        
        # 6. 创建ConsensusGroup节点并建立带频率的边和ASSOCIATED_WITH边
        for cg_str, count in entity_log_counts['ConsensusGroup'].items():
            cg_node_id = get_node_id("ConsensusGroup", cg_str)
            cg_node = KGNode(
                node_id=cg_node_id,
                node_type="ConsensusGroup",
                properties={"group": cg_str}
            )
            kg.add_node(cg_node)
            
            # 计算频率
            percentage = (count / total_logs * 100) if total_logs > 0 else 0.0
            frequency_level = "high" if percentage > 50 else "medium" if percentage > 10 else "low"
            
            # Window包含ConsensusGroup（带频率属性）
            kg.add_edge(KGEdge(
                source_id=window_node_id,
                target_id=cg_node_id,
                relation_type="CONTAINS",
                properties={
                    "entity_type": "ConsensusGroup",
                    "count": count,
                    "percentage": round(percentage, 2),
                    "frequency_level": frequency_level
                }
            ))
            
            # 创建ASSOCIATED_WITH边：只有包含该ConsensusGroup的LogInstance才连接
            for log_instance_id, entities_dict in log_instance_entities.items():
                if cg_str in entities_dict['ConsensusGroup']:
                    kg.add_edge(KGEdge(
                        source_id=log_instance_id,
                        target_id=cg_node_id,
                        relation_type="ASSOCIATED_WITH",
                        properties={"entity_type": "ConsensusGroup"}
                    ))
        
        # 7. 创建统计实体节点并建立带频率的边和ASSOCIATED_WITH边（如果使用增强提取器）
        if USE_ENHANCED_EXTRACTOR:
            for stat_entity, count in entity_log_counts['StatisticalEntity'].items():
                # 统计实体使用GeneralEntity类型
                stat_node_id = get_node_id("GeneralEntity", stat_entity)
                stat_node = KGNode(
                    node_id=stat_node_id,
                    node_type="GeneralEntity",
                    properties={
                        "token": stat_entity,
                        "entity_source": "statistical_mining"
                    }
                )
                kg.add_node(stat_node)
                
                # 计算频率
                percentage = (count / total_logs * 100) if total_logs > 0 else 0.0
                frequency_level = "high" if percentage > 50 else "medium" if percentage > 10 else "low"
                
                # Window包含统计实体（带频率属性）
                kg.add_edge(KGEdge(
                    source_id=window_node_id,
                    target_id=stat_node_id,
                    relation_type="CONTAINS",
                    properties={
                        "entity_type": "GeneralEntity",
                        "count": count,
                        "percentage": round(percentage, 2),
                        "frequency_level": frequency_level
                    }
                ))
                
                # 创建ASSOCIATED_WITH边：只有包含该GeneralEntity的LogInstance才连接
                for log_instance_id, entities_dict in log_instance_entities.items():
                    if stat_entity in entities_dict['GeneralEntity']:
                        kg.add_edge(KGEdge(
                            source_id=log_instance_id,
                            target_id=stat_node_id,
                            relation_type="ASSOCIATED_WITH",
                            properties={"entity_type": "GeneralEntity", "entity_source": "statistical_mining"}
                        ))
            
            # DataRegion属于ConsensusGroup (遍历每个 CG，避免依赖循环变量泄露)
            for cg_str_local in entity_log_counts['ConsensusGroup']:
                if ":" not in cg_str_local:
                    continue
                cg_type, cg_id = cg_str_local.split(":", 1)
                if cg_type != "DataRegion":
                    continue
                dr_node_id = get_node_id("DataRegion", cg_id)
                if dr_node_id not in kg.nodes:
                    continue
                cg_node_id_local = get_node_id("ConsensusGroup", cg_str_local)
                kg.add_edge(KGEdge(
                    source_id=dr_node_id,
                    target_id=cg_node_id_local,
                    relation_type="BELONGS_TO",
                    properties={}
                ))
        
        # 7. Template节点不需要单独创建
        # Template信息存储在Log节点的template_sequence中
        # 模板详情存储在单独的模板文件中
        
        if window_count % 100 == 0:
            print(f"  已处理 {window_count}/{len(windows_data)} 个窗口...")
    
    print(f"  完成：处理 {window_count} 个窗口，{log_count} 条日志")
    
    return kg


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="从窗口数据构建知识图谱",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-file", type=Path, required=True,
                       help="输入的窗口数据JSON文件路径")
    parser.add_argument("--output-file", type=Path, required=True,
                       help="输出的知识图谱JSON文件路径")
    parser.add_argument("--sample", type=int, default=None,
                       help="只处理前N个窗口（用于测试，默认处理全部）")
    parser.add_argument("--realtime-parse", action="store_true",
                       help="如果窗口数据中没有template_id，实时解析日志生成（默认False，推荐先重新生成窗口数据）")
    parser.add_argument("--sim-threshold", type=float, default=0.5,
                       help="实时解析时的Drain相似度阈值（仅在--realtime-parse时使用）")
    parser.add_argument("--depth", type=int, default=4,
                       help="实时解析时的Drain树深度（仅在--realtime-parse时使用）")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("构建知识图谱")
    print("=" * 80)
    print(f"\n输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    
    # 读取窗口数据
    print("\n1. 读取窗口数据...")
    if not args.input_file.exists():
        print(f"错误: 输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    with args.input_file.open('r', encoding='utf-8') as f:
        windows_data = json.load(f)
    
    print(f"   读取了 {len(windows_data)} 个窗口")
    
    # 如果指定了采样，只处理前N个窗口
    if args.sample:
        windows_data = dict(list(windows_data.items())[:args.sample])
        print(f"   采样处理: {len(windows_data)} 个窗口")
    
    # 如果需要实时解析，初始化parser
    parser = None
    if args.realtime_parse:
        print("\n1.5. 初始化Drain解析器（实时解析模式）...")
        parser = RealtimeDrainParser(
            depth=args.depth,
            sim_th=args.sim_threshold
        )
        print(f"   相似度阈值: {args.sim_threshold}, 树深度: {args.depth}")
        print("   ⚠️  注意：实时解析模式效率较低，推荐先重新生成包含template_id的窗口数据")
    
    # 构建知识图谱
    print("\n2. 构建知识图谱...")
    kg = build_knowledge_graph_from_windows(windows_data, parser=parser)
    
    # 保存结果
    print(f"\n3. 保存知识图谱到 {args.output_file}...")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    kg_dict = kg.to_dict()
    with args.output_file.open('w', encoding='utf-8') as f:
        json.dump(kg_dict, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ 保存成功")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("知识图谱统计")
    print("=" * 80)
    stats = kg_dict["statistics"]
    print(f"\n节点统计:")
    print(f"  总节点数: {stats['total_nodes']}")
    for node_type, count in sorted(stats['nodes_by_type'].items()):
        print(f"  {node_type}: {count}")
    
    print(f"\n边统计:")
    print(f"  总边数: {stats['total_edges']}")
    for edge_type, count in sorted(stats['edges_by_type'].items()):
        print(f"  {edge_type}: {count}")
    
    # 打印示例节点和边
    print(f"\n示例节点（每种类型1个）:")
    node_types_seen = set()
    for node in kg_dict["nodes"][:20]:  # 最多显示20个
        if node["type"] not in node_types_seen:
            print(f"  {node['id']} ({node['type']}): {node['properties']}")
            node_types_seen.add(node["type"])
    
    print(f"\n示例边（每种类型1个）:")
    edge_types_seen = set()
    for edge in kg_dict["edges"][:20]:  # 最多显示20个
        if edge["relation"] not in edge_types_seen:
            print(f"  {edge['source']} --[{edge['relation']}]--> {edge['target']}")
            edge_types_seen.add(edge["relation"])
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

