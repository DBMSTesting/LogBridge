#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级索引构建脚本
从完整KG中提取子图构建所需的最小数据，预计算每个窗口的top-k相似窗口。
相比完整KG，内存占用减少约85%+。
"""

import json
import sys
import time
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# 新目录结构
SRC_ROOT = Path(__file__).resolve().parent.parent  # log_anomaly_diagnosis/src
sys.path.insert(0, str(SRC_ROOT / "utils"))
from global_kg_loader import GlobalKGDataLoader

# 异常类型到标签的映射（与GlobalKGDataLoader保持一致）
ANOMALY_TYPE_TO_LABEL = {
    'compaction': 0,
    'export': 1,
    'flush': 2,
    'full_cpu': 3,
    'full_memory': 4,
    'network_bandwidth2': 5,
    'normal': 6
}

ENTITY_TYPES = frozenset(['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity', 'Anomaly'])


def build_lightweight_index(kg_file: Path, output_file: Path, top_k_candidates: int = 200,
                             max_entity_cov: int = None) -> Dict:
    """
    从KG构建轻量级索引
    
    Args:
        kg_file: 知识图谱JSON文件路径
        output_file: 输出索引文件路径（.pkl）
        top_k_candidates: 每个窗口预计算的候选相似窗口数量（用于后续按allowed_windows过滤）
        
    Returns:
        轻量级索引字典
    """
    print(f"{'='*60}")
    print(f"构建轻量级索引")
    print(f"{'='*60}")
    print(f"输入KG: {kg_file}")
    print(f"输出文件: {output_file}")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # 1. 加载KG
    print(f"\n1. 加载知识图谱...")
    sys.stdout.flush()
    data_loader = GlobalKGDataLoader(kg_file)
    data_loader.load_kg()
    
    nodes_dict = data_loader.nodes_dict
    edges_dict = data_loader.edges_dict
    window_to_logs = dict(data_loader.window_to_logs)
    log_to_window = data_loader.log_to_window
    
    # 2. 提取轻量级数据
    print(f"\n2. 提取轻量级数据...")
    sys.stdout.flush()
    
    # 2.1 窗口-实体映射
    window_entity_sets = {}
    for window_id in data_loader.window_idx_map.keys():
        entity_set = set()
        for target, relation, props in edges_dict.get(window_id, []):
            if relation == 'CONTAINS':
                target_node = nodes_dict.get(target)
                if target_node and target_node.get('type') in ENTITY_TYPES:
                    entity_set.add(target)
        if entity_set:
            window_entity_sets[window_id] = frozenset(entity_set)  # 使用frozenset节省内存
    
    print(f"    window_entity_sets: {len(window_entity_sets)} 个窗口")
    sys.stdout.flush()
    
    # 2.2 实体-窗口反向映射
    entity_to_windows = defaultdict(set)
    for window_id, entity_set in window_entity_sets.items():
        for entity_id in entity_set:
            entity_to_windows[entity_id].add(window_id)
    entity_to_windows = dict(entity_to_windows)
    print(f"    entity_to_windows: {len(entity_to_windows)} 个实体")
    sys.stdout.flush()
    
    # 2.3 窗口到日志（已是list格式）
    window_to_logs = {k: list(v) for k, v in window_to_logs.items()}
    print(f"    window_to_logs: {len(window_to_logs)} 个窗口")
    sys.stdout.flush()
    
    # 2.4 窗口到标签（与build_subgraphs_optimized逻辑一致：优先用Window.properties.anomaly_types，否则get_window_label）
    window_to_label = {}
    for window_id in data_loader.window_idx_map.keys():
        node = nodes_dict.get(window_id)
        if node:
            anomaly_types = node.get('properties', {}).get('anomaly_types', [])
            if anomaly_types:
                label = ANOMALY_TYPE_TO_LABEL.get(anomaly_types[0], -1)
            else:
                label = data_loader.get_window_label(window_id)
        else:
            label = ANOMALY_TYPE_TO_LABEL['normal']
        window_to_label[window_id] = label
    print(f"    window_to_label: {len(window_to_label)} 个窗口")
    sys.stdout.flush()
    
    # 2.5 窗口到实体（用于构建log->entity边，与window_entity_sets等价但保留为dict便于查找）
    window_to_entities = {wid: set(es) for wid, es in window_entity_sets.items()}
    
    # 2.6 日志到实体（仅来自 KG 真实 ASSOCIATED_WITH 边）
    # 注：旧版本会把"log 所在窗口的所有实体"一并塞进来，导致 log->entity 边数虚膨胀 10×+
    #     这里只保留真实关联边
    log_to_entities = defaultdict(set)
    for log_id in data_loader.log_idx_map.keys():
        for target, relation, props in edges_dict.get(log_id, []):
            if relation == 'ASSOCIATED_WITH':
                target_node = nodes_dict.get(target)
                if target_node and target_node.get('type') in ENTITY_TYPES:
                    log_to_entities[log_id].add(target)
    log_to_entities = {k: list(v) for k, v in log_to_entities.items()}
    print(f"    log_to_entities: {len(log_to_entities)} 条日志有实体关联")
    sys.stdout.flush()
    
    # 2.7 日志文本（raw_line 或 template_id）
    log_to_text = {}
    for log_id in data_loader.log_idx_map.keys():
        node = nodes_dict.get(log_id)
        if node:
            raw_line = node.get('properties', {}).get('raw_line', '')
            template_id = node.get('properties', {}).get('template_id', '')
            if raw_line:
                log_to_text[log_id] = raw_line
            elif template_id:
                log_to_text[log_id] = template_id
    print(f"    log_to_text: {len(log_to_text)} 条日志")
    sys.stdout.flush()
    
    # 2.8 实体内容
    entity_to_content = {}
    for entity_id in data_loader.entity_idx_map.keys():
        node = nodes_dict.get(entity_id)
        if node:
            if node.get('type') == 'GeneralEntity':
                token = node.get('properties', {}).get('token', '')
                entity_to_content[entity_id] = token if token else (entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id)
            elif node.get('type') == 'Thread':
                thread_name = node.get('properties', {}).get('thread_name', '')
                entity_to_content[entity_id] = thread_name if thread_name else (entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id)
            else:
                entity_to_content[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
    print(f"    entity_to_content: {len(entity_to_content)} 个实体")
    sys.stdout.flush()
    
    # 3. 预计算每个窗口的top-k相似窗口
    # 优化：跳过过高 cov 的实体（这些是"全局热点"，作为桥梁无意义且导致候选数爆炸）
    print(f"\n3. 预计算窗口相似度（top-{top_k_candidates} 候选）...")
    if max_entity_cov is not None:
        # 找出高 cov 实体
        skipped_entities = {e for e, ws in entity_to_windows.items() if len(ws) > max_entity_cov}
        print(f"    优化：忽略 cov > {max_entity_cov} 的 {len(skipped_entities)} 个全局热点实体")
        print(f"        例: {sorted(skipped_entities, key=lambda e: -len(entity_to_windows[e]))[:5]}")
    else:
        skipped_entities = set()
    sys.stdout.flush()

    window_top_neighbors = {}
    all_windows = list(window_entity_sets.keys())

    import time as _time
    t_loop_start = _time.time()
    for i, window_id in enumerate(all_windows):
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = _time.time() - t_loop_start
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(all_windows) - i - 1) / max(rate, 1e-3)
            print(f"    已处理 {i+1}/{len(all_windows)} 个窗口  elapsed={elapsed:.0f}s  rate={rate:.0f} win/s  ETA={eta/60:.1f}min")
            sys.stdout.flush()

        entities = window_entity_sets[window_id]
        # 过滤掉高 cov 实体，剩下的才用来找候选窗口
        effective_entities = entities - skipped_entities if skipped_entities else entities
        candidates = set()
        for e in effective_entities:
            candidates.update(entity_to_windows.get(e, []))
        candidates.discard(window_id)

        if not candidates:
            window_top_neighbors[window_id] = []
            continue

        scores = []
        for other_wid in candidates:
            other_entities = window_entity_sets.get(other_wid, frozenset())
            # 算交集时仍然包括所有实体（让高 cov 实体也加分，但不让它们"招揽"候选）
            score = len(entities & other_entities)
            scores.append((other_wid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        window_top_neighbors[window_id] = scores[:top_k_candidates]
    
    print(f"    window_top_neighbors: {len(window_top_neighbors)} 个窗口")
    sys.stdout.flush()
    
    # 4. 组装索引
    index = {
        'window_entity_sets': window_entity_sets,
        'entity_to_windows': entity_to_windows,
        'window_to_logs': window_to_logs,
        'window_to_label': window_to_label,
        'window_to_entities': window_to_entities,
        'log_to_entities': log_to_entities,
        'log_to_text': log_to_text,
        'entity_to_content': entity_to_content,
        'window_top_neighbors': window_top_neighbors,
        'all_windows': all_windows,
        'log_to_window': log_to_window,
    }
    
    # 5. 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n4. 保存索引到 {output_file}...")
    sys.stdout.flush()
    
    with open(output_file, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"索引构建完成")
    print(f"  文件大小: {file_size_mb:.1f} MB")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    return index


def main():
    parser = argparse.ArgumentParser(description='从KG构建轻量级索引')
    parser.add_argument('--kg-file', type=str, required=True, help='知识图谱JSON文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出索引文件路径（默认：kg同目录/lightweight_index.pkl）')
    parser.add_argument('--top-k-candidates', type=int, default=200, help='每个窗口预计算的候选相似窗口数')
    parser.add_argument('--max-entity-cov', type=int, default=None,
                        help='跳过 cov > 此值的全局热点实体（不作为候选展开锚点）')

    args = parser.parse_args()
    
    kg_file = Path(args.kg_file)
    if not kg_file.exists():
        print(f"错误: KG文件不存在 {kg_file}")
        sys.exit(1)
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = kg_file.parent / "lightweight_index.pkl"
    
    build_lightweight_index(kg_file, output_file, args.top_k_candidates,
                             max_entity_cov=args.max_entity_cov)


if __name__ == '__main__':
    main()
