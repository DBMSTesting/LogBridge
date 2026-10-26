#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练HGT异常分类模型（支持跨窗口信息传递）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter
import numpy as np

SRC_ROOT = Path(__file__).resolve().parent.parent
for sub in ('model', 'utils', 'pipeline'):
    sys.path.insert(0, str(SRC_ROOT / sub))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

from global_kg_loader import GlobalKGDataLoader
from encoder import TemplateEncoder, EntityEncoder
from hgt_anomaly_model import HGTAnomalyModel

class WindowDataset(Dataset):
    """窗口数据集（支持跨窗口信息传递）"""
    
    def __init__(
        self,
        window_ids: List[str],
        labels: List[int],
        data_loader: Optional[GlobalKGDataLoader] = None,
        use_subgraph: bool = True,
        num_hops: int = 2,
        max_neighbors: int = 50,
        allowed_windows: Optional[Set[str]] = None,
        prebuild_subgraphs: bool = True,
        prebuilt_subgraph_dir: Optional[Path] = None,
        split_name: str = "train",
        strip_count_aware: bool = False,
    ):
        """Init."""
        self.window_ids = window_ids
        self.labels = labels
        self.data_loader = data_loader
        self.use_subgraph = use_subgraph
        self.num_hops = num_hops
        self.max_neighbors = max_neighbors
        self.allowed_windows = allowed_windows
        self.prebuild_subgraphs = prebuild_subgraphs
        self.prebuilt_subgraph_dir = prebuilt_subgraph_dir
        self.strip_count_aware = strip_count_aware

        self.subgraph_cache = {}
        if prebuilt_subgraph_dir is not None:
            import sys
            import importlib.util

            load_subgraphs_path = Path(__file__).parent / "load_subgraphs.py"
            
            if load_subgraphs_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "load_subgraphs_module", 
                    load_subgraphs_path
                )
                load_subgraphs_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(load_subgraphs_module)
                load_prebuilt_subgraphs = load_subgraphs_module.load_prebuilt_subgraphs
                check_subgraph_compatibility = load_subgraphs_module.check_subgraph_compatibility
            else:

                def load_prebuilt_subgraphs(*args, **kwargs):
                    return {}, {}
                def check_subgraph_compatibility(*args, **kwargs):
                    return False
            
            try:
                subgraphs, metadata = load_prebuilt_subgraphs(prebuilt_subgraph_dir, split_name, load_all=False)

                if not check_subgraph_compatibility(metadata, num_hops, max_neighbors):
                    raise ValueError('Context parameters do not match the requested configuration')
                else:

                    if '_chunk_files' in metadata:

                        self.chunk_metadata = metadata
                        self.chunk_cache = {}
                        print(f"  使用chunks格式，将按需从 {len(metadata['_chunk_files'])} 个chunks加载子图")
                        sys.stdout.flush()
                    else:

                        for window_id in window_ids:
                            if window_id in subgraphs:
                                target_window_idx = getattr(subgraphs[window_id], 'target_window_idx', 0)
                                self.subgraph_cache[window_id] = (subgraphs[window_id], target_window_idx)
                        print(f"  从磁盘加载了 {len(self.subgraph_cache)} 个子图")
                        sys.stdout.flush()
            except FileNotFoundError as e:
                raise FileNotFoundError('Precomputed contexts are required') from e

        if self.prebuild_subgraphs and len(self.subgraph_cache) == 0 and not getattr(self, 'chunk_metadata', None):
            import sys
            print(f"  预构建 {len(window_ids)} 个子图...")
            sys.stdout.flush()
            for i, window_id in enumerate(window_ids):
                if i % 100 == 0 and i > 0:
                    print(f"    已预构建: {i}/{len(window_ids)} ({100*i/len(window_ids):.1f}%)")
                    sys.stdout.flush()
                
                if self.use_subgraph:
                    hetero_data, _ = self.data_loader.build_window_subgraph(
                        window_id,
                        num_hops=self.num_hops,
                        max_neighbors=self.max_neighbors,
                        allowed_windows=self.allowed_windows
                    )
                else:
                    hetero_data, _ = self.data_loader.build_window_subgraph(
                        window_id,
                        num_hops=0,
                        max_neighbors=1
                    )
                
                if hetero_data is not None:
                    target_window_idx = getattr(hetero_data, 'target_window_idx', 0)
                    self.subgraph_cache[window_id] = (hetero_data, target_window_idx)
            
            print(f"  子图预构建完成: {len(self.subgraph_cache)}/{len(window_ids)}")
            sys.stdout.flush()
    
    def __len__(self):
        return len(self.window_ids)
    
    def __getitem__(self, idx):
        window_id = self.window_ids[idx]
        label = self.labels[idx]

        if self.prebuilt_subgraph_dir is not None:

            if hasattr(self, 'chunk_metadata') and self.chunk_metadata:

                from pathlib import Path
                chunk_files = self.chunk_metadata.get('_chunk_files', [])
                if not hasattr(self, 'chunk_cache'):
                    self.chunk_cache = {}

                for chunk_file in chunk_files:
                    chunk_path = Path(chunk_file)
                    if not chunk_path.is_absolute():

                        chunk_dir = Path(self.chunk_metadata.get('_chunk_dir', ''))
                        if chunk_dir:
                            chunk_path = Path(chunk_dir) / chunk_file
                        else:
                            chunk_path = self.prebuilt_subgraph_dir.parent / chunk_file

                    chunk_key = str(chunk_path)
                    if chunk_key not in self.chunk_cache:
                        try:
                            self.chunk_cache[chunk_key] = torch.load(chunk_path, map_location='cpu', weights_only=False)
                        except Exception as e:
                            continue

                    chunk_data = self.chunk_cache[chunk_key]
                    if window_id in chunk_data:
                        hetero_data = chunk_data[window_id]
                        target_window_idx = getattr(hetero_data, 'target_window_idx', 0)
                        return hetero_data, label, target_window_idx

                return None, label, 0

            elif len(self.subgraph_cache) > 0:
                if window_id in self.subgraph_cache:
                    hetero_data, target_window_idx = self.subgraph_cache[window_id]
                    return hetero_data, label, target_window_idx
                else:

                    return None, label, 0

        if self.prebuild_subgraphs:
            if window_id in self.subgraph_cache:
                hetero_data, target_window_idx = self.subgraph_cache[window_id]
                return hetero_data, label, target_window_idx
            else:

                return None, label, 0

        if self.data_loader is None:

            return None, label, 0
            
        if self.use_subgraph:
            hetero_data, _ = self.data_loader.build_window_subgraph(
                window_id,
                num_hops=self.num_hops,
                max_neighbors=self.max_neighbors,
                allowed_windows=self.allowed_windows
            )
            target_window_idx = getattr(hetero_data, 'target_window_idx', 0) if hetero_data else 0
        else:
            hetero_data, _ = self.data_loader.build_window_subgraph(
                window_id,
                num_hops=0,
                max_neighbors=1
            )
            target_window_idx = 0
            
            return hetero_data, label, target_window_idx

def collate_fn(batch):
    """自定义collate函数"""
    hetero_data_list, labels, target_window_indices = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    target_window_indices = list(target_window_indices)
    return list(hetero_data_list), labels, target_window_indices

def train_epoch(
    model: HGTAnomalyModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    profile_timing: bool = False,
    strip_count_aware: bool = False,
):
    """Train epoch."""
    import sys
    import time
    from torch.cuda.amp import autocast, GradScaler
    from collections import defaultdict
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(dataloader)

    scaler = GradScaler() if use_amp else None
    
    start_time = time.time()

    timing_stats = defaultdict(list)
    data_loading_times = []
    
    optimizer.zero_grad()
    
    for batch_idx, (hetero_data_list, labels, target_window_indices) in enumerate(dataloader):
        data_load_start = time.time()
        labels = labels.to(device)
        data_loading_times.append(time.time() - data_load_start)
        
        batch_loss = None
        batch_correct = 0
        batch_size = len(hetero_data_list)
        valid_samples = 0

        with autocast(enabled=use_amp):
            for hetero_data, label, target_window_idx in zip(hetero_data_list, labels, target_window_indices):
                if hetero_data is None:
                    continue

                if strip_count_aware and hasattr(hetero_data, 'log_counts'):
                    hetero_data.log_counts = None

                data_to_device_start = time.time()
                hetero_data = hetero_data.to(device)
                data_to_device_time = time.time() - data_to_device_start

                logits = model(hetero_data, target_window_idx=target_window_idx, profile_timing=profile_timing)

                if profile_timing and hasattr(hetero_data, '_timings'):
                    for key, value in hetero_data._timings.items():
                        timing_stats[key].append(value)
                    timing_stats['data_to_device'].append(data_to_device_time)

                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0)) / accumulation_steps
                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss += loss
                valid_samples += 1

                pred = logits.argmax(dim=0)
                if pred.item() == label.item():
                    batch_correct += 1
        
        if batch_size > 0 and batch_loss is not None and valid_samples > 0:

            if use_amp:
                scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:

                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += batch_loss.item() * accumulation_steps
            correct += batch_correct
            total += batch_size

        if batch_idx < 10 or batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / max(1, batch_idx + 1)
            remaining = avg_time * (total_batches - batch_idx - 1)
            print(f"    训练进度: {batch_idx+1}/{total_batches} ({100*(batch_idx+1)/total_batches:.1f}%) - "
                  f"Loss: {total_loss/max(1, batch_idx+1):.4f}, Acc: {correct/max(1, total):.4f} - "
                  f"已用时间: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟")

            if profile_timing and timing_stats:
                print(f"    【性能分析】平均耗时（秒）:")
                for key in ['bert_encoding', 'local_transformer', 'entity_encoding', 
                           'hgt_message_passing', 'attention_pooling', 'classification', 
                           'data_to_device', 'total']:
                    if key in timing_stats and len(timing_stats[key]) > 0:
                        avg = sum(timing_stats[key]) / len(timing_stats[key])
                        pct = (avg / sum(timing_stats.get('total', [avg]))) * 100 if 'total' in timing_stats else 0
                        print(f"      {key}: {avg:.4f}s ({pct:.1f}%)")
                if data_loading_times:
                    avg_data_load = sum(data_loading_times) / len(data_loading_times)
                    print(f"      data_loading: {avg_data_load:.4f}s")
            
            sys.stdout.flush()

    if total_batches % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    if profile_timing and timing_stats:
        print(f"\n    【Epoch性能统计】平均耗时（秒）:")
        for key in ['bert_encoding', 'local_transformer', 'entity_encoding', 
                   'hgt_message_passing', 'attention_pooling', 'classification', 
                   'data_to_device', 'total']:
            if key in timing_stats and len(timing_stats[key]) > 0:
                avg = sum(timing_stats[key]) / len(timing_stats[key])
                total_avg = sum(timing_stats.get('total', [avg])) / len(timing_stats.get('total', [1]))
                pct = (avg / total_avg) * 100 if total_avg > 0 else 0
                print(f"      {key}: {avg:.4f}s ({pct:.1f}%)")
        if data_loading_times:
            avg_data_load = sum(data_loading_times) / len(data_loading_times)
            print(f"      data_loading: {avg_data_load:.4f}s")
        print()
    
    return avg_loss, accuracy

def subgraph_to_single_window(sg):
    """构建仅包含目标窗口日志及关联实体的子图，用于单窗口评估。"""
    from torch_geometric.data import HeteroData
    target_idx = sg.target_window_idx
    target_wid = sg.target_window_id

    if ('window', 'CONTAINS', 'log') not in sg.edge_types:
        return None
    wl_edges = sg['window', 'CONTAINS', 'log'].edge_index
    target_log_local_idx = [wl_edges[1, i].item() for i in range(wl_edges.size(1))
                             if wl_edges[0, i].item() == target_idx]
    if not target_log_local_idx:
        return None

    target_log_set = set(target_log_local_idx)
    target_entity_local_idx = set()
    if ('log', 'ASSOCIATED_WITH', 'entity') in sg.edge_types:
        le_edges = sg['log', 'ASSOCIATED_WITH', 'entity'].edge_index
        for i in range(le_edges.size(1)):
            if le_edges[0, i].item() in target_log_set:
                target_entity_local_idx.add(le_edges[1, i].item())
    target_entity_local_idx = sorted(target_entity_local_idx)

    log_new_idx = {old: new for new, old in enumerate(target_log_local_idx)}
    entity_new_idx = {old: new for new, old in enumerate(target_entity_local_idx)}

    new_sg = HeteroData()
    new_sg['window'].num_nodes = 1
    new_sg['log'].num_nodes = len(target_log_local_idx)
    new_sg['entity'].num_nodes = len(target_entity_local_idx)
    new_sg['window', 'CONTAINS', 'log'].edge_index = torch.tensor(
        [[0] * len(target_log_local_idx), list(range(len(target_log_local_idx)))], dtype=torch.long)

    if ('log', 'ASSOCIATED_WITH', 'entity') in sg.edge_types and target_entity_local_idx:
        le_edges = sg['log', 'ASSOCIATED_WITH', 'entity'].edge_index
        src, dst = [], []
        for i in range(le_edges.size(1)):
            li, ei = le_edges[0, i].item(), le_edges[1, i].item()
            if li in log_new_idx and ei in entity_new_idx:
                src.append(log_new_idx[li]); dst.append(entity_new_idx[ei])
        if src:
            new_sg['log', 'ASSOCIATED_WITH', 'entity'].edge_index = torch.tensor([src, dst], dtype=torch.long)
            new_sg['entity', 'REVERSE_ASSOCIATED_WITH', 'log'].edge_index = torch.tensor([dst, src], dtype=torch.long)

    new_sg.target_window_id = target_wid
    new_sg.target_window_idx = 0
    new_sg.window_ids = [target_wid]
    new_sg.log_ids = [sg.log_ids[i] for i in target_log_local_idx]
    new_sg.entity_ids = [sg.entity_ids[i] for i in target_entity_local_idx]
    new_sg.label = sg.label
    if hasattr(sg, 'log_template_texts'):
        new_sg.log_template_texts = {lid: sg.log_template_texts[lid]
                                       for lid in new_sg.log_ids if lid in sg.log_template_texts}
    if hasattr(sg, 'entity_contents'):
        new_sg.entity_contents = {eid: sg.entity_contents[eid]
                                    for eid in new_sg.entity_ids if eid in sg.entity_contents}
    if hasattr(sg, 'log_counts'):
        new_sg.log_counts = {lid: sg.log_counts[lid]
                               for lid in new_sg.log_ids if lid in sg.log_counts}
    return new_sg

def evaluate(
    model: HGTAnomalyModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    class_names: List[str],
    inference_mode: str = 'multi_window',
    strip_count_aware: bool = False,
):
    """评估分类模型；默认使用选定历史上下文，single_window仅保留目标窗口。"""
    import sys
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total_batches = len(dataloader)
    n_dropped = 0

    with torch.no_grad():
        for batch_idx, (hetero_data_list, labels, target_window_indices) in enumerate(dataloader):
            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"    评估进度: {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.1f}%)")
                sys.stdout.flush()
            labels = labels.to(device)

            for hetero_data, label, target_window_idx in zip(hetero_data_list, labels, target_window_indices):
                if hetero_data is None:
                    continue

                if strip_count_aware and hasattr(hetero_data, 'log_counts'):
                    hetero_data.log_counts = None

                if inference_mode == 'single_window':
                    sw_sg = subgraph_to_single_window(hetero_data)
                    if sw_sg is None:
                        n_dropped += 1
                        continue
                    if strip_count_aware:
                        sw_sg.log_counts = None
                    sw_sg = sw_sg.to(device)
                    logits = model(sw_sg, target_window_idx=0)
                else:
                    hetero_data = hetero_data.to(device)
                    logits = model(hetero_data, target_window_idx=target_window_idx)

                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                total_loss += loss.item()
                pred = logits.argmax(dim=0)

                all_preds.append(pred.item())
                all_labels.append(label.item())

    if inference_mode == 'single_window' and n_dropped > 0:
        print(f"    [single_window] 跳过 {n_dropped} 个 target_window 为空的子图")
    
    avg_loss = total_loss / len(all_preds) if len(all_preds) > 0 else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds) if all_preds else 0.0

    all_possible_labels = list(range(len(class_names)))
    print("\n分类报告:")
    try:
        print(classification_report(
            all_labels, all_preds, 
            target_names=class_names, 
            labels=all_possible_labels,
            zero_division=0
        ))
    except Exception as e:
        print(f"  警告: 分类报告生成失败: {e}")
        print(f"  实际预测的类别: {set(all_preds)}")

    print("\n混淆矩阵 (行=真实标签, 列=预测标签):")
    try:
        cm = confusion_matrix(all_labels, all_preds, labels=all_possible_labels)

        print("      ", end="")
        for name in class_names:
            print(f"{name[:8]:>10}", end="")
        print()

        for i, name in enumerate(class_names):
            print(f"{name[:8]:>8}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:>10}", end="")
            print(f"  (总计: {cm[i, :].sum()})")

        if 'compaction' in class_names:
            compaction_idx = class_names.index('compaction')
            compaction_total = cm[compaction_idx, :].sum()
            compaction_correct = cm[compaction_idx, compaction_idx]
            compaction_errors = cm[compaction_idx, :].copy()
            compaction_errors[compaction_idx] = 0
            
            print(f"\n【Compaction误判分析】:")
            print(f"  总样本数: {compaction_total}")
            print(f"  正确识别: {compaction_correct} ({100*compaction_correct/compaction_total if compaction_total > 0 else 0:.1f}%)")
            print(f"  误判分布:")
            for j, pred_name in enumerate(class_names):
                if compaction_errors[j] > 0:
                    print(f"    被误判为 {pred_name}: {compaction_errors[j]} ({100*compaction_errors[j]/compaction_total if compaction_total > 0 else 0:.1f}%)")
    except Exception as e:
        print(f"  警告: 混淆矩阵生成失败: {e}")
        print(f"  实际标签的类别: {set(all_labels)}")
        print(f"  所有可能的类别: {all_possible_labels}")
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='训练HGT异常分类模型（支持跨窗口信息传递）')
    parser.add_argument('--kg-file', type=str, required=True,
                        help='细粒度知识图谱文件路径')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu/auto，auto表示自动检测)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--use-subgraph', action='store_true', default=True,
                        help='使用子图采样（包含跨窗口邻居）')
    parser.add_argument('--num-hops', type=int, default=1,
                        help='子图采样跳数')
    parser.add_argument('--max-neighbors', type=int, default=5,
                        help='最大邻居窗口数')
    parser.add_argument('--max-windows', type=int, default=None,
                       help='最大窗口数（用于小规模测试，None表示使用全部）')
    parser.add_argument('--prebuilt-subgraph-dir', type=str, default=None,
                       help='预构建子图目录（如果提供，直接从磁盘加载，避免重新构建）')
    parser.add_argument('--bert-embeddings-file', type=str, default=None,
                       help='预编码的BERT向量文件（.pt文件），如果提供，将使用预编码向量而不是实时编码')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='使用混合精度训练（Automatic Mixed Precision），可以提升速度并减少显存占用')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='梯度累积步数（模拟更大的batch_size，1表示不累积）')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='数据加载的进程数（0表示单进程，如果内存允许可以增加）')
    parser.add_argument('--profile-timing', action='store_true', default=False,
                       help='是否记录各步骤耗时（用于性能分析）')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练（提供检查点文件路径，如：./checkpoints/best_model.pt）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference-mode', type=str, default='multi_window',
                        choices=['single_window', 'multi_window'],
                        help='Use admissible earlier context with multi_window inference')
    parser.add_argument('--no-count-aware', action='store_true', default=False,
                        help='Ablation: omit the learned log-count feature')

    args = parser.parse_args()
    import torch
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device is None or args.device == 'auto':
        import torch
        if torch.cuda.is_available():
            args.device = 'cuda'
            print(f"检测到CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            args.device = 'cpu'
            print("CUDA不可用，使用CPU")
    elif args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("警告：指定使用CUDA但CUDA不可用，自动切换到CPU")
            args.device = 'cpu'
    
    import sys
    print("=" * 80)
    print("训练HGT异常分类模型（支持跨窗口信息传递）")
    print("=" * 80)
    print(f"使用设备: {args.device}")
    sys.stdout.flush()

    print("\n1. 加载全局知识图谱...")
    sys.stdout.flush()

    data_loader = GlobalKGDataLoader(Path(args.kg_file))
    data_loader.load_kg()
    print("  知识图谱加载完成！")
    sys.stdout.flush()

    print("\n2. 准备数据...")
    sys.stdout.flush()

    prebuilt_dir = Path(args.prebuilt_subgraph_dir) if args.prebuilt_subgraph_dir else None
    use_prebuilt_windows = False
    train_windows = None
    train_labels = None
    val_windows = None
    val_labels = None
    test_windows = None
    test_labels = None
    
    if prebuilt_dir is not None:
        print("  尝试从预构建子图加载窗口ID和标签...")
        sys.stdout.flush()
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from load_subgraphs import load_prebuilt_subgraphs

            def _windows_from(subgraphs, metadata):
                w2l = metadata.get('window_to_label', {}) or {}
                if subgraphs:
                    wids = list(subgraphs.keys())
                else:
                    wids = list(w2l.keys())
                labels = [w2l.get(wid, 0) for wid in wids]
                return wids, labels

            train_subgraphs, train_metadata = load_prebuilt_subgraphs(prebuilt_dir, "train")
            train_windows, train_labels = _windows_from(train_subgraphs, train_metadata)

            val_subgraphs, val_metadata = load_prebuilt_subgraphs(prebuilt_dir, "val")
            val_windows, val_labels = _windows_from(val_subgraphs, val_metadata)

            test_subgraphs, test_metadata = load_prebuilt_subgraphs(prebuilt_dir, "test")
            if any(m.get('protocol') != 'logbridge-context' for m in
                   (train_metadata, val_metadata, test_metadata)):
                raise ValueError('Context schema mismatch; run pipeline/prepare_data.py')
            test_windows, test_labels = _windows_from(test_subgraphs, test_metadata)
            
            print(f"  ✅ 成功从预构建子图加载:")
            print(f"    训练集: {len(train_windows)} 个窗口")
            print(f"    验证集: {len(val_windows)} 个窗口")
            print(f"    测试集: {len(test_windows)} 个窗口")
            use_prebuilt_windows = True
            sys.stdout.flush()
        except Exception as e:
            print(f"  ⚠️  无法从预构建子图加载窗口ID: {e}")
            print(f"  将使用知识图谱中的窗口进行采样...")
            sys.stdout.flush()
            use_prebuilt_windows = False
    
    if not use_prebuilt_windows:
        raise ValueError('Training requires contexts generated by prepare_data.py')

    else:

        print("\n3. 使用预构建子图的窗口和标签（已划分好）")
        sys.stdout.flush()
        print(f"  训练集: {len(train_windows)}")
        print(f"  验证集: {len(val_windows)}")
        print(f"  测试集: {len(test_windows)}")
        sys.stdout.flush()

        sample_windows = train_windows

    print("\n4. 构建实体词汇表...")
    sys.stdout.flush()
    print(f"  使用 {len(sample_windows)} 个训练窗口构建实体词汇表")
    sys.stdout.flush()
    all_entity_ids = set()
    import time
    start_time = time.time()
    for i, window_id in enumerate(sample_windows):
        if i == 0:
            print(f"    开始处理第一个窗口...")
            sys.stdout.flush()
        elif i % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i if i > 0 else 0
            remaining = avg_time * (len(sample_windows) - i)
            print(f"    已处理窗口: {i}/{len(sample_windows)} ({100*i/len(sample_windows):.1f}%) - 已用时间: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟")
            sys.stdout.flush()

        target_log_ids = set(data_loader.window_to_logs.get(window_id, []))
        if target_log_ids:

            for log_id in target_log_ids:
                edges = data_loader.edges_dict.get(log_id, [])
                for target, relation, props in edges:
                    if relation == 'ASSOCIATED_WITH':
                        all_entity_ids.add(target)
    
    entity_vocab = {entity_id: idx for idx, entity_id in enumerate(sorted(all_entity_ids))}
    print(f"  实体词汇表大小: {len(entity_vocab)}")
    sys.stdout.flush()

    print("\n4. 初始化编码器...")
    sys.stdout.flush()
    print("  初始化模板编码器...")
    sys.stdout.flush()

    precomputed_embeddings = None
    if args.bert_embeddings_file:
        bert_emb_file = Path(args.bert_embeddings_file)
        if bert_emb_file.exists():
            print(f"  加载预编码BERT向量: {bert_emb_file}")
            sys.stdout.flush()
            try:
                bert_data = torch.load(bert_emb_file, map_location='cpu')
                precomputed_embeddings = bert_data.get('embeddings', {})
                model_name = bert_data.get('model_name', 'bert-base-uncased')
                embedding_dim = bert_data.get('embedding_dim', 768)
                num_templates = bert_data.get('num_templates', len(precomputed_embeddings))
                print(f"    ✅ 加载成功: {num_templates} 个模板，维度: {embedding_dim}")
                print(f"    模型: {model_name}")
                sys.stdout.flush()
            except Exception as e:
                print(f"    ⚠️  加载预编码向量失败: {e}")
                print(f"    将使用实时BERT编码")
                sys.stdout.flush()
                precomputed_embeddings = None
        else:
            print(f"  ⚠️  预编码向量文件不存在: {bert_emb_file}")
            print(f"    将使用实时BERT编码")
            sys.stdout.flush()
    
    template_encoder = TemplateEncoder(
        device=args.device,
        precomputed_embeddings=precomputed_embeddings
    )
    print("  初始化实体编码器...")
    sys.stdout.flush()
    entity_encoder = EntityEncoder(
        entity_vocab=entity_vocab,
        embedding_dim=128,
        use_bert=False,
        device=args.device
    )
    print("  编码器初始化完成")
    sys.stdout.flush()

    print("\n5. 初始化模型...")
    sys.stdout.flush()
    print("  创建HGT异常分类模型...")
    sys.stdout.flush()

    num_classes = 6
    print(f"  模型类别数: {num_classes}")
    sys.stdout.flush()
    
    model = HGTAnomalyModel(
        template_encoder=template_encoder,
        entity_encoder=entity_encoder,
        log_embedding_dim=template_encoder.embedding_dim,
        entity_embedding_dim=128,
        num_classes=num_classes,
        device=args.device
    )
    model = model.to(args.device)
    print("  模型已移动到设备:", args.device)
    sys.stdout.flush()

    print("\n6. 设置损失函数和优化器...")
    sys.stdout.flush()
    print("  初始化交叉熵损失...")
    sys.stdout.flush()
    print('  Unweighted cross-entropy over six anomaly categories')
    criterion = nn.CrossEntropyLoss()
    print("  初始化Adam优化器...")
    sys.stdout.flush()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("  初始化学习率调度器...")
    sys.stdout.flush()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print("  损失函数和优化器设置完成")
    sys.stdout.flush()

    print("\n7. 创建数据加载器...")
    sys.stdout.flush()
    print(f"  使用子图采样: {args.use_subgraph}")
    print(f"  采样跳数: {args.num_hops}")
    print(f"  最大邻居窗口数: {args.max_neighbors}")
    sys.stdout.flush()

    print("  计算允许扩展的窗口集合...")
    sys.stdout.flush()
    train_allowed_windows = set(train_windows)
    val_allowed_windows = set(val_windows)
    test_allowed_windows = set(test_windows)
    
    print(f"  训练集允许扩展的窗口数: {len(train_allowed_windows)}")
    print(f"  验证集允许扩展的窗口数: {len(val_allowed_windows)}")
    print(f"  测试集允许扩展的窗口数: {len(test_allowed_windows)}")
    sys.stdout.flush()

    prebuilt_dir = Path(args.prebuilt_subgraph_dir) if args.prebuilt_subgraph_dir else None
    if prebuilt_dir is not None:
        print(f"  使用预构建子图目录: {prebuilt_dir}")
        sys.stdout.flush()

        train_subgraph_file = prebuilt_dir / 'subgraphs_train.pt'
        val_subgraph_file = prebuilt_dir / 'subgraphs_val.pt'
        test_subgraph_file = prebuilt_dir / 'subgraphs_test.pt'
        
        has_train = train_subgraph_file.exists()
        has_val = val_subgraph_file.exists()
        has_test = test_subgraph_file.exists()
        
        print(f"    训练集预构建子图: {'存在' if has_train else '不存在'}")
        print(f"    验证集预构建子图: {'存在' if has_val else '不存在'}")
        print(f"    测试集预构建子图: {'存在' if has_test else '不存在'}")
        sys.stdout.flush()

        if has_train and has_val and has_test:
            data_loader_for_dataset = None
            prebuild_subgraphs = False
        else:

            data_loader_for_dataset = data_loader
            prebuild_subgraphs = False
            print(f"  注意: 部分数据集没有预构建子图，将动态构建")
            sys.stdout.flush()
    else:
        print("  未提供预构建子图目录，将在内存中构建子图")
        sys.stdout.flush()
        data_loader_for_dataset = data_loader
        prebuild_subgraphs = True
    
    print("  创建训练集数据集...")
    sys.stdout.flush()
    train_dataset = WindowDataset(
        train_windows, train_labels, data_loader_for_dataset,
        use_subgraph=args.use_subgraph,
        num_hops=args.num_hops,
        max_neighbors=args.max_neighbors,
        allowed_windows=train_allowed_windows,
        prebuild_subgraphs=prebuild_subgraphs,
        prebuilt_subgraph_dir=prebuilt_dir,
        split_name="train"
    )
    print("  创建验证集数据集...")
    sys.stdout.flush()
    val_dataset = WindowDataset(
        val_windows, val_labels, data_loader_for_dataset,
        use_subgraph=args.use_subgraph,
        num_hops=args.num_hops,
        max_neighbors=args.max_neighbors,
        allowed_windows=val_allowed_windows,
        prebuild_subgraphs=prebuild_subgraphs,
        prebuilt_subgraph_dir=prebuilt_dir,
        split_name="val"
    )
    print("  创建测试集数据集...")
    sys.stdout.flush()
    test_dataset = WindowDataset(
        test_windows, test_labels, data_loader_for_dataset,
        use_subgraph=args.use_subgraph,
        num_hops=args.num_hops,
        max_neighbors=args.max_neighbors,
        allowed_windows=test_allowed_windows,
        prebuild_subgraphs=prebuild_subgraphs,
        prebuilt_subgraph_dir=prebuilt_dir,
        split_name="test"
    )
    
    print("  创建DataLoader...")
    sys.stdout.flush()

    num_workers = args.num_workers
    if num_workers > 0:
        print(f"  使用 {num_workers} 个数据加载进程")
        sys.stdout.flush()
    else:
        print(f"  使用单进程数据加载（num_workers=0）")
        sys.stdout.flush()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True if num_workers > 0 and args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True if num_workers > 0 and args.device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True if num_workers > 0 and args.device == 'cuda' else False
    )
    print(f"  数据加载器创建完成 - 训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}, 测试批次数: {len(test_loader)}")
    sys.stdout.flush()

    print("\n8. 开始训练...")
    sys.stdout.flush()

    num_classes = 6
    print(f"  检测到 {num_classes} 个类别")

    if hasattr(data_loader, 'ANOMALY_TYPE_TO_LABEL') and len(data_loader.ANOMALY_TYPE_TO_LABEL) == 2:

        class_names = ['normal', 'not_normal']
        print(f"  使用二分类任务，类别: {class_names}")
    elif num_classes == 2:

        class_names = ['normal', 'not_normal']
        print(f"  推断为二分类任务，类别: {class_names}")
    else:

        class_names = ['compaction', 'export', 'flush', 'full_cpu', 'full_memory', 'network_bandwidth2', 'normal']

        if num_classes < len(class_names):
            class_names = class_names[:num_classes]
        print(f"  使用多分类任务，类别: {class_names}")
    sys.stdout.flush()

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n从检查点恢复训练: {resume_path}")
            sys.stdout.flush()
            try:
                checkpoint = torch.load(resume_path, map_location=args.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"  从Epoch {start_epoch} 继续训练")
                if 'val_acc' in checkpoint:
                    best_val_acc = checkpoint['val_acc']
                    print(f"  最佳验证准确率: {best_val_acc:.4f}")
                if 'entity_vocab' in checkpoint:

                    print(f"  检查点包含entity_vocab")
                print(f"  检查点加载成功！")
                sys.stdout.flush()
            except Exception as e:
                print(f"  ⚠️  警告: 加载检查点失败: {e}")
                print(f"  将从头开始训练")
                sys.stdout.flush()
                start_epoch = 0
        else:
            print(f"  ⚠️  警告: 检查点文件不存在: {resume_path}")
            print(f"  将从头开始训练")
            sys.stdout.flush()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        sys.stdout.flush()

        print("  训练阶段...")
        sys.stdout.flush()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device,
            use_amp=args.use_amp,
            accumulation_steps=args.accumulation_steps,
            profile_timing=args.profile_timing,
            strip_count_aware=args.no_count_aware,
        )
        print(f"  训练完成 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        sys.stdout.flush()

        print("  验证阶段...")
        sys.stdout.flush()
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion, args.device, class_names,
            inference_mode=args.inference_mode,
            strip_count_aware=args.no_count_aware,
        )
        print(f"  验证完成 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        sys.stdout.flush()

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            print(f"  验证准确率提升！保存模型 (Acc: {val_acc:.4f})")
            sys.stdout.flush()
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'entity_vocab': entity_vocab,
                'protocol': 'logbridge-context',
                'seed': args.seed,
                'training_args': vars(args),
            }, output_dir / 'best_model.pt')
            print(f"  保存最佳模型 (验证准确率: {val_acc:.4f})")
            sys.stdout.flush()
        else:
            patience_counter += 1
            print(f"  验证准确率未提升，耐心值: {patience_counter}/{args.early_stopping_patience}")
            sys.stdout.flush()
            if patience_counter >= args.early_stopping_patience:
                print(f"  早停触发 (耐心值: {args.early_stopping_patience})")
                sys.stdout.flush()
                break

    print("\n9. 测试最佳模型...")
    sys.stdout.flush()
    print("  加载最佳模型...")
    sys.stdout.flush()
    checkpoint = torch.load(Path(args.output_dir) / 'best_model.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  开始测试...")
    sys.stdout.flush()
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, criterion, args.device, class_names,
        inference_mode=args.inference_mode,
        strip_count_aware=args.no_count_aware,
    )
    print(f"  测试完成 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    sys.stdout.flush()
    
    print("\n训练完成！")
    sys.stdout.flush()
    print(f"\n跨窗口信息传递已启用:")
    print(f"  - 每个窗口的子图包含通过Entity节点连接的跨窗口邻居")
    print(f"  - HGT模型可以在这些跨窗口的Entity节点上进行消息传递")
    print(f"  - 实现了真正的跨窗口信息共享和关联建模")

if __name__ == '__main__':
    main()

