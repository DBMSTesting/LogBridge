#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于轻量级索引的子图构建脚本（优化版）
- 使用轻量级索引替代完整KG，内存占用减少约85%
- 使用预计算的window_top_neighbors，构建时直接查表，不再循环计算相似度
- 支持多进程，每个进程仅加载轻量索引（~1GB级），可安全增加进程数
"""

import sys
import time
import argparse
import pickle
import gc
import shutil
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Any
import torch
from concurrent.futures import ProcessPoolExecutor, wait, TimeoutError as FutureTimeoutError
import multiprocessing as mp

# 添加父目录到路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "04_subgraph_building"))
sys.path.insert(0, str(project_root / "05_model" / "utils"))

from build_subgraphs_optimized import IncrementalSaver
from torch_geometric.data import HeteroData

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

MAX_TOTAL_NODES = 200000

# 全局变量用于worker进程
_index = None


def _init_worker(index_path: str):
    """初始化worker进程，加载轻量级索引"""
    global _index
    with open(index_path, 'rb') as f:
        _index = pickle.load(f)
    print(f"  子进程加载轻量索引完成: {len(_index.get('all_windows', []))} 个窗口")
    sys.stdout.flush()


def _get_expanded_windows_from_precomputed(
    window_id: str,
    max_neighbors: int,
    allowed_windows: Optional[Set[str]],
    index: Dict
) -> Set[str]:
    """
    从预计算的window_top_neighbors中获取expanded_window_ids
    直接查表，O(k)复杂度，k为预计算的候选数
    """
    top_neighbors = index.get('window_top_neighbors', {}).get(window_id, [])
    
    expanded = {window_id}
    count = 0
    for other_wid, _ in top_neighbors:
        if count >= max_neighbors - 1:
            break
        if allowed_windows is None or other_wid in allowed_windows:
            expanded.add(other_wid)
            count += 1
    
    return expanded


def build_single_subgraph_lightweight(args_tuple: Tuple) -> Tuple[str, Any, int, float]:
    """
    基于轻量级索引构建单个子图
    
    Args:
        args_tuple: (window_id, num_hops, max_neighbors, allowed_windows)
        
    Returns:
        Tuple[window_id, hetero_data, label, execution_time]
    """
    global _index
    window_id, num_hops, max_neighbors, allowed_windows = args_tuple
    
    task_start = time.time()
    
    try:
        # 若窗口无实体，跳过
        if window_id not in _index.get('window_entity_sets', {}):
            return (window_id, None, -1, time.time() - task_start)
        
        # 1. 直接查表获取expanded_window_ids（核心优化：不再循环计算相似度）
        expanded_window_ids = _get_expanded_windows_from_precomputed(
            window_id, max_neighbors, allowed_windows, _index
        )
        
        # 2. 获取扩展的日志和实体
        window_to_logs = _index['window_to_logs']
        window_to_entities = _index['window_to_entities']
        log_to_entities = _index['log_to_entities']
        log_to_text = _index['log_to_text']
        entity_to_content = _index['entity_to_content']
        window_to_label = _index['window_to_label']
        
        expanded_log_ids = set()
        for wid in expanded_window_ids:
            expanded_log_ids.update(window_to_logs.get(wid, []))
        
        expanded_entity_ids = set()
        for wid in expanded_window_ids:
            expanded_entity_ids.update(window_to_entities.get(wid, set()))
        for log_id in expanded_log_ids:
            expanded_entity_ids.update(log_to_entities.get(log_id, []))
        
        # 3. 规模检查
        if len(expanded_log_ids) + len(expanded_entity_ids) + len(expanded_window_ids) > MAX_TOTAL_NODES:
            return (window_id, None, -1, time.time() - task_start)
        
        # 4. 获取标签
        label = window_to_label.get(window_id, ANOMALY_TYPE_TO_LABEL['normal'])
        
        # 5. 构建局部索引和边
        window_local_idx = {wid: i for i, wid in enumerate(expanded_window_ids)}
        log_local_idx = {lid: i for i, lid in enumerate(expanded_log_ids)}
        entity_local_idx = {eid: i for i, eid in enumerate(expanded_entity_ids)}
        
        # log->entity映射：仅使用每条 log 真实关联的实体（来自 KG 的 ASSOCIATED_WITH 边）
        # 移除 method 1（"窗口实体复制到所有日志"），避免 log->entity 边数虚膨胀
        log_to_entities_local = {}
        for log_id in expanded_log_ids:
            kept = [e for e in log_to_entities.get(log_id, []) if e in entity_local_idx]
            if kept:
                log_to_entities_local[log_id] = kept
        
        # 构建边
        window_to_log_edges = []
        log_to_entity_edges = []
        entity_to_log_edges = []
        
        for wid in expanded_window_ids:
            w_idx = window_local_idx[wid]
            for log_id in window_to_logs.get(wid, []):
                if log_id in log_local_idx:
                    l_idx = log_local_idx[log_id]
                    window_to_log_edges.append([w_idx, l_idx])
                    for entity_id in log_to_entities_local.get(log_id, []):
                        e_idx = entity_local_idx[entity_id]
                        log_to_entity_edges.append([l_idx, e_idx])
                        entity_to_log_edges.append([e_idx, l_idx])
        
        # 6. 构建HeteroData（与原始格式完全一致）
        data = HeteroData()
        data['window'].num_nodes = len(expanded_window_ids)
        data['log'].num_nodes = len(expanded_log_ids)
        data['entity'].num_nodes = len(expanded_entity_ids)
        
        if window_to_log_edges:
            data['window', 'CONTAINS', 'log'].edge_index = torch.tensor(
                window_to_log_edges, dtype=torch.long
            ).t().contiguous()
        if log_to_entity_edges:
            data['log', 'ASSOCIATED_WITH', 'entity'].edge_index = torch.tensor(
                log_to_entity_edges, dtype=torch.long
            ).t().contiguous()
        if entity_to_log_edges:
            data['entity', 'REVERSE_ASSOCIATED_WITH', 'log'].edge_index = torch.tensor(
                entity_to_log_edges, dtype=torch.long
            ).t().contiguous()
        
        data.target_window_id = window_id
        data.target_window_idx = window_local_idx.get(window_id, 0)
        data.window_ids = list(expanded_window_ids)
        data.log_ids = list(expanded_log_ids)
        data.entity_ids = list(expanded_entity_ids)
        data.label = label
        
        # 7. 存储log_template_texts和entity_contents（与原始一致）
        log_template_texts = {}
        for log_id in expanded_log_ids:
            text = log_to_text.get(log_id)
            if text:
                log_template_texts[log_id] = text
        
        entity_contents = {}
        for entity_id in expanded_entity_ids:
            content = entity_to_content.get(entity_id)
            if content:
                entity_contents[entity_id] = content
            else:
                entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
        
        data.log_template_texts = log_template_texts
        data.entity_contents = entity_contents

        # 7b. 若索引提供 log_to_count（压缩日志的 count 字段），附到 hetero_data
        # 供 model forward 做 count-aware LT 展开
        log_to_count = _index.get('log_to_count') if isinstance(_index, dict) else None
        if log_to_count:
            data.log_counts = {lid: log_to_count.get(lid, 1) for lid in expanded_log_ids}

        data = data.cpu()
        execution_time = time.time() - task_start
        return (window_id, data, label, execution_time)
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return (window_id, None, -1, time.time() - task_start)


def build_and_save_subgraphs_lightweight(
    index_path: Path,
    window_ids: List[str],
    labels: List[int],
    output_dir: Path,
    num_hops: int = 1,
    max_neighbors: int = 10,
    allowed_windows: Optional[Set[str]] = None,
    split_name: str = "train",
    num_workers: int = 4,
    chunk_size: int = 1000,
    max_subgraphs_per_run: Optional[int] = None
) -> Tuple[Optional[Path], Dict]:
    """
    使用轻量级索引构建并保存子图
    
    接口与 build_and_save_subgraphs_optimized 保持一致
    """
    print(f"\n{'='*80}")
    print(f"构建 {split_name} 集的子图（轻量级索引版）")
    print(f"{'='*80}")
    print(f"窗口数: {len(window_ids)}")
    print(f"参数: num_hops={num_hops}, max_neighbors={max_neighbors}, num_workers={num_workers}")
    sys.stdout.flush()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载索引以获取all_windows（用于断点续传检查）
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    # 断点续传
    output_file = output_dir / f"subgraphs_{split_name}.pt"
    existing_windows = set()
    existing_subgraphs = {}
    existing_subgraphs_file_path = None
    
    if output_file.exists():
        try:
            saved = torch.load(output_file, map_location='cpu')
            if isinstance(saved, dict):
                if 'subgraphs' in saved:
                    existing_subgraphs = saved['subgraphs']
                    existing_windows = set(existing_subgraphs.keys())
                    existing_subgraphs_file_path = str(output_file)
                elif 'chunk_files' in saved:
                    existing_subgraphs_file_path = str(output_file)
                    for cf in saved.get('chunk_files', []):
                        cf_path = Path(cf)
                        if not cf_path.is_absolute() and saved.get('chunk_dir'):
                            cf_path = Path(saved['chunk_dir']) / cf_path.name
                        try:
                            chunk_data = torch.load(str(cf_path), map_location='cpu')
                            if isinstance(chunk_data, dict):
                                existing_windows.update(chunk_data.keys())
                        except Exception:
                            pass
        except Exception as e:
            print(f"  警告: 加载已存在文件失败: {e}")
            existing_subgraphs_file_path = str(output_file)
    
    if not existing_windows:
        chunk_dir = output_dir / f"chunks_{split_name}"
        if chunk_dir.exists():
            for cf in sorted(chunk_dir.glob('chunk_*.pt')):
                try:
                    chunk_data = torch.load(str(cf), map_location='cpu')
                    if isinstance(chunk_data, dict):
                        existing_windows.update(chunk_data.keys())
                except Exception:
                    pass
    
    remaining_window_ids = [wid for wid in window_ids if wid not in existing_windows]
    remaining_labels = [labels[i] for i, wid in enumerate(window_ids) if wid not in existing_windows]
    
    if existing_windows:
        print(f"  断点续传: 跳过 {len(existing_windows)} 个已构建窗口，剩余 {len(remaining_window_ids)} 个")
        sys.stdout.flush()
    
    if not remaining_window_ids:
        print("  所有窗口已构建完成！")
        sys.stdout.flush()
        return output_file, {}
    
    incremental_saver = IncrementalSaver(output_dir, split_name, chunk_size=chunk_size)
    initial_chunk_count = len(incremental_saver.chunk_files)
    
    # 轻量索引下可安全增加进程数（每个进程约1GB级）
    MAX_SAFE_WORKERS = 16
    if num_workers > MAX_SAFE_WORKERS:
        num_workers = MAX_SAFE_WORKERS
    
    use_parallel = len(remaining_window_ids) >= 100 and num_workers > 1
    
    start_time = time.time()
    
    if use_parallel:
        print(f"\n2. 使用 {num_workers} 个进程并行构建（轻量索引）...")
        sys.stdout.flush()
        
        args_list = [
            (wid, num_hops, max_neighbors, allowed_windows)
            for wid in remaining_window_ids
        ]
        
        completed = 0
        total_to_build = len(remaining_window_ids)
        failed_count = 0
        should_stop = False
        
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=[str(index_path)]
        ) as executor:
            future_to_window = {
                executor.submit(build_single_subgraph_lightweight, args): args[0]
                for args in args_list
            }
            pending = set(future_to_window.keys())
            
            while pending and not should_stop:
                done, _ = wait(pending, timeout=10, return_when='FIRST_COMPLETED')
                
                for future in done:
                    if should_stop:
                        break
                    pending.discard(future)
                    
                    try:
                        result = future.result(timeout=1)
                        window_id, hetero_data, label, _ = result
                        completed += 1
                        
                        if completed % 100 == 0 or completed == total_to_build:
                            elapsed = time.time() - start_time
                            avg = elapsed / completed
                            remaining = avg * (total_to_build - completed)
                            print(f"  进度: {completed}/{total_to_build} ({100*completed/total_to_build:.1f}%) - "
                                  f"已用: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟, 失败: {failed_count}")
                            sys.stdout.flush()
                        
                        if hetero_data is not None:
                            incremental_saver.add_subgraph(window_id, hetero_data)
                            
                            if max_subgraphs_per_run is not None:
                                new_count = len(incremental_saver.current_chunk_data)
                                new_count += (len(incremental_saver.chunk_files) - initial_chunk_count) * chunk_size
                                if new_count >= max_subgraphs_per_run:
                                    print(f"\n  达到最大子图数限制 {max_subgraphs_per_run}，保存并退出...")
                                    should_stop = True
                                    for f in pending:
                                        if not f.done():
                                            f.cancel()
                        else:
                            failed_count += 1
                    except Exception as e:
                        window_id = future_to_window.get(future, "unknown")
                        print(f"  ⚠️ 窗口 {window_id} 处理失败: {e}")
                        completed += 1
                        failed_count += 1
        
        print(f"\n  处理完成: 成功 {completed - failed_count}, 失败 {failed_count}")
        sys.stdout.flush()
    else:
        print(f"\n2. 串行构建...")
        sys.stdout.flush()
        
        global _index
        with open(index_path, 'rb') as f:
            _index = pickle.load(f)
        
        completed = 0
        total_to_build = len(remaining_window_ids)
        for i, window_id in enumerate(remaining_window_ids):
            if (i + 1) % 10 == 0 or (i + 1) == total_to_build:
                elapsed = time.time() - start_time
                if i > 0:
                    avg = elapsed / (i + 1)
                    remaining = avg * (total_to_build - i - 1)
                    print(f"  进度: {i+1}/{total_to_build} - 已用: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟")
                sys.stdout.flush()
            
            result = build_single_subgraph_lightweight(
                (window_id, num_hops, max_neighbors, allowed_windows)
            )
            window_id, hetero_data, label, _ = result
            completed += 1
            
            if hetero_data is not None:
                incremental_saver.add_subgraph(window_id, hetero_data)
                if max_subgraphs_per_run is not None:
                    new_count = len(incremental_saver.current_chunk_data)
                    new_count += (len(incremental_saver.chunk_files) - initial_chunk_count) * chunk_size
                    if new_count >= max_subgraphs_per_run:
                        print(f"\n  达到最大子图数限制 {max_subgraphs_per_run}，保存并退出...")
                        break
    
    # 最终保存
    print(f"\n3. 最终保存...")
    sys.stdout.flush()
    
    metadata = {
        'num_hops': num_hops,
        'max_neighbors': max_neighbors,
        'num_windows': len(window_ids),
        'num_success': len(existing_subgraphs) + len(incremental_saver.current_chunk_data) +
                       (len(incremental_saver.chunk_files) - initial_chunk_count) * chunk_size,
        'window_to_label': {wid: labels[i] for i, wid in enumerate(window_ids)}
    }
    
    final_file = incremental_saver.finalize(
        metadata,
        merge_chunks=False,
        existing_subgraphs=existing_subgraphs,
        existing_subgraphs_file_path=existing_subgraphs_file_path
    )
    
    elapsed = time.time() - start_time
    print(f"\n构建完成！总耗时: {elapsed/60:.1f}分钟")
    sys.stdout.flush()
    
    return final_file or output_file, metadata
