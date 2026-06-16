#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预构建子图脚本（优化版）
优化策略：
1. KG共享内存优化：使用mmap避免每个进程重复加载KG
2. 预计算索引优化：预计算窗口-实体映射和窗口关联度
3. 增量保存优化：分批保存，避免重复序列化
"""

import json
import sys
import time
import argparse
import pickle
import mmap
import os
import signal
import atexit
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Any
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, wait, TimeoutError as FutureTimeoutError
from functools import partial
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from collections import defaultdict

# 添加父目录到路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "05_model" / "utils"))
from global_kg_loader import GlobalKGDataLoader


# ==================== 策略1: KG共享内存优化 ====================

class SharedKGLoader:
    """共享KG加载器，使用multiprocessing.shared_memory实现真正的进程间共享"""
    
    def __init__(self, kg_file: Path, cache_dir: Optional[Path] = None):
        """
        初始化共享KG加载器
        
        Args:
            kg_file: 原始KG JSON文件路径
            cache_dir: 缓存目录（用于存储序列化的KG）
        """
        self.kg_file = kg_file
        self.cache_dir = cache_dir or kg_file.parent / "kg_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{kg_file.stem}.pkl"
        self.shared_memory_name = None
        self.shared_memory_size = None
        self.shared_memory = None  # 保存共享内存对象引用，用于清理
        self.kg_data = None
        
    def prepare_shared_kg(self):
        """准备共享KG（主进程调用，将KG加载到共享内存）"""
        print(f"准备共享KG缓存: {self.cache_file}")
        sys.stdout.flush()
        
        # 先检查是否有缓存文件，如果没有则创建
        if not self.cache_file.exists():
            print(f"  缓存文件不存在，创建缓存文件...")
            sys.stdout.flush()
            
            # 加载原始KG
            print(f"  加载原始KG...")
            sys.stdout.flush()
            data_loader = GlobalKGDataLoader(self.kg_file)
            data_loader.load_kg()
            
            # 序列化KG数据到文件
            print(f"  序列化KG数据到文件（这可能需要几分钟）...")
            sys.stdout.flush()
            start_time = time.time()
            
            # 构建nodes_dict（确保所有节点都被包含）
            nodes_dict = {}
            for node_id, node_data in data_loader.nodes_dict.items():
                nodes_dict[node_id] = node_data
            
            kg_data = {
                'nodes_dict': nodes_dict,
                'edges_dict': dict(data_loader.edges_dict),
                'window_idx_map': data_loader.window_idx_map,
                'log_idx_map': data_loader.log_idx_map,
                'entity_idx_map': data_loader.entity_idx_map,
                'idx_to_window': data_loader.idx_to_window,
                'idx_to_log': data_loader.idx_to_log,
                'idx_to_entity': data_loader.idx_to_entity,
                'window_to_logs': dict(data_loader.window_to_logs),
                'log_to_window': data_loader.log_to_window,
                'entity_to_logs': {k: list(v) for k, v in data_loader.entity_to_logs.items()},
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(kg_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            elapsed = time.time() - start_time
            file_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
            print(f"  序列化完成: {file_size_mb:.1f} MB, 耗时 {elapsed:.1f}秒")
            sys.stdout.flush()
        else:
            print(f"  缓存文件已存在，跳过序列化")
            sys.stdout.flush()
        
        # 读取缓存文件到内存并序列化为字节
        print(f"  将KG数据加载到共享内存...")
        sys.stdout.flush()
        start_time = time.time()
        
        # ⚠️ 优化：直接从文件读取并序列化，避免在内存中保留完整的kg_data对象
        # 这样可以减少主进程的内存占用
        with open(self.cache_file, 'rb') as f:
            # 先读取文件大小
            f.seek(0, 2)  # 移动到文件末尾
            file_size = f.tell()
            f.seek(0)  # 回到文件开头
            
            # 直接读取文件内容并序列化（如果文件已经序列化，可以直接使用）
            # 但为了确保格式一致，我们还是重新序列化
            kg_data = pickle.load(f)
        
        # 序列化为字节
        kg_bytes = pickle.dumps(kg_data, protocol=pickle.HIGHEST_PROTOCOL)
        kg_size = len(kg_bytes)
        
        # ⚠️ 重要：立即释放kg_data，避免主进程保留KG副本
        # worker进程会从共享内存加载KG，主进程不需要保留
        del kg_data
        import gc
        gc.collect()
        
        # 创建共享内存
        shm_name = f"kg_{self.kg_file.stem}_{os.getpid()}"
        try:
            # 尝试清理可能存在的旧共享内存
            try:
                old_shm = SharedMemory(name=shm_name, create=False)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            
            shm = SharedMemory(name=shm_name, create=True, size=kg_size)
            shm.buf[:kg_size] = kg_bytes
            self.shared_memory_name = shm_name
            self.shared_memory_size = kg_size
            self.shared_memory = shm  # 保存引用，用于后续清理
            
            # ⚠️ 重要：立即释放kg_bytes，避免主进程保留KG数据的副本
            # 共享内存已经创建，主进程不需要保留kg_bytes
            del kg_bytes
            import gc
            gc.collect()
            
            elapsed = time.time() - start_time
            print(f"  KG已加载到共享内存: {kg_size/(1024*1024):.1f} MB, 耗时 {elapsed:.1f}秒")
            print(f"  共享内存名称: {shm_name}")
            sys.stdout.flush()
            
            # 注意：不关闭共享内存，让worker进程访问
            # 主进程会在所有worker结束后清理共享内存
            
        except Exception as e:
            print(f"  警告: 创建共享内存失败: {e}")
            print(f"  回退到文件缓存模式")
            sys.stdout.flush()
            self.shared_memory_name = None
    
    def load_shared_kg(self, shared_memory_name: Optional[str] = None, shared_memory_size: Optional[int] = None):
        """
        加载共享KG（子进程调用）
        
        从共享内存读取序列化的KG数据并反序列化。
        注意：虽然序列化数据在共享内存中，但pickle.loads()反序列化后的
        Python对象（dict, list等）仍然是进程独立的。
        
        Args:
            shared_memory_name: 共享内存名称（由主进程传入）
            shared_memory_size: 共享内存大小（由主进程传入）
        """
        if self.kg_data is not None:
            return self.kg_data
        
        print(f"  子进程从共享内存加载KG...")
        sys.stdout.flush()
        start_time = time.time()
        
        shm_name = shared_memory_name or self.shared_memory_name
        shm_size = shared_memory_size or self.shared_memory_size
        
        if shm_name and shm_size:
            try:
                # 从共享内存读取
                shm = SharedMemory(name=shm_name, create=False)
                kg_bytes = bytes(shm.buf[:shm_size])
                self.kg_data = pickle.loads(kg_bytes)
                shm.close()  # 子进程关闭自己的引用，但不unlink
                
                elapsed = time.time() - start_time
                print(f"  从共享内存加载完成: {shm_size/(1024*1024):.1f} MB, 耗时 {elapsed:.1f}秒")
                sys.stdout.flush()
                return self.kg_data
            except Exception as e:
                print(f"  警告: 从共享内存加载失败: {e}，回退到文件模式")
                sys.stdout.flush()
        
        # 回退到文件模式
        print(f"  从文件加载KG缓存...")
        sys.stdout.flush()
        with open(self.cache_file, 'rb') as f:
            self.kg_data = pickle.load(f)
        
        elapsed = time.time() - start_time
        file_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
        print(f"  从文件加载完成: {file_size_mb:.1f} MB, 耗时 {elapsed:.1f}秒")
        sys.stdout.flush()
        
        return self.kg_data


# ==================== 策略2: 预计算索引优化 ====================

class PrecomputedIndices:
    """预计算的索引，加速子图构建"""
    
    def __init__(self, kg_data: Dict):
        """
        初始化预计算索引
        
        Args:
            kg_data: 从SharedKGLoader加载的KG数据
        """
        self.kg_data = kg_data
        # ⚠️ 优化：只使用window_entity_sets，避免重复存储
        self.window_entity_sets = None
        self.entity_to_windows = None
        
    def compute_indices(self):
        """计算所有索引"""
        print("预计算索引...")
        sys.stdout.flush()
        start_time = time.time()
        
        # 1. 预计算窗口-实体映射
        print("  1. 计算窗口-实体映射...")
        sys.stdout.flush()
        # ⚠️ 优化：只使用window_entity_sets，避免重复存储（window_to_entities和window_entity_sets存储相同数据）
        self.window_entity_sets = {}
        
        edges_dict = self.kg_data['edges_dict']
        nodes_dict = self.kg_data['nodes_dict']
        window_idx_map = self.kg_data['window_idx_map']
        
        # 直接从 Window 的 CONTAINS 边获取 Entity（新版本KG结构）
        # 遍历所有Window节点
        for window_id in window_idx_map.keys():
            edges = edges_dict.get(window_id, [])
            entity_set = set()
            for target, relation, props in edges:
                if relation == 'CONTAINS':
                    # 检查target是否是Entity节点
                    target_node = nodes_dict.get(target)
                    if target_node and target_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity', 'Anomaly']:
                        entity_set.add(target)
            if entity_set:
                self.window_entity_sets[window_id] = entity_set
        
        print(f"    完成: {len(self.window_entity_sets)} 个窗口")
        sys.stdout.flush()
        
        # 2. 预计算实体-窗口反向映射
        print("  2. 计算实体-窗口反向映射...")
        sys.stdout.flush()
        self.entity_to_windows = defaultdict(set)
        
        for window_id, entity_set in self.window_entity_sets.items():
            for entity_id in entity_set:
                self.entity_to_windows[entity_id].add(window_id)
        
        print(f"    完成: {len(self.entity_to_windows)} 个实体")
        sys.stdout.flush()
        
        elapsed = time.time() - start_time
        print(f"预计算索引完成，耗时 {elapsed:.1f}秒")
        sys.stdout.flush()
    
    def get_window_entities(self, window_id: str) -> Set[str]:
        """获取窗口的实体集合（O(1)查找）"""
        return self.window_entity_sets.get(window_id, set())
    
    def get_entity_windows(self, entity_id: str) -> Set[str]:
        """获取实体关联的窗口集合（O(1)查找）"""
        return self.entity_to_windows.get(entity_id, set())
    
    def compute_window_similarity(self, window1_id: str, window2_id: str) -> int:
        """计算两个窗口的相似度（共享实体数量，O(1)）"""
        entities1 = self.window_entity_sets.get(window1_id, set())
        entities2 = self.window_entity_sets.get(window2_id, set())
        return len(entities1 & entities2)


# ==================== 策略3: 增量保存优化 ====================

class IncrementalSaver:
    """增量保存器，避免重复序列化"""
    
    def __init__(self, output_dir: Path, split_name: str, chunk_size: int = 1000):
        """
        初始化增量保存器
        
        Args:
            output_dir: 输出目录
            split_name: 数据集名称
            chunk_size: 每个chunk的子图数量
        """
        self.output_dir = output_dir
        self.split_name = split_name
        self.chunk_size = chunk_size
        self.chunk_dir = output_dir / f"chunks_{split_name}"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查已存在的chunk文件，从最大的chunk编号+1开始，避免覆盖
        existing_chunks = sorted(self.chunk_dir.glob('chunk_*.pt'))
        if existing_chunks:
            # 提取chunk编号，找到最大的
            max_chunk_num = -1
            for chunk_file in existing_chunks:
                try:
                    # chunk文件名格式: chunk_N.pt
                    chunk_num = int(chunk_file.stem.split('_')[1])
                    max_chunk_num = max(max_chunk_num, chunk_num)
                except:
                    pass
            self.current_chunk = max_chunk_num + 1
            print(f"  检测到已存在的chunk文件，从chunk_{self.current_chunk}开始（避免覆盖）")
        else:
            self.current_chunk = 0
        
        self.current_chunk_data = {}
        self.chunk_files = []
    
    def add_subgraph(self, window_id: str, hetero_data: Any):
        """添加一个子图"""
        self.current_chunk_data[window_id] = hetero_data
        
        if len(self.current_chunk_data) >= self.chunk_size:
            self.save_chunk()
            # 强制垃圾回收，释放内存
            import gc
            gc.collect()
    
    def save_chunk(self):
        """保存当前chunk"""
        if not self.current_chunk_data:
            return
        
        chunk_file = self.chunk_dir / f"chunk_{self.current_chunk}.pt"
        chunk_size = len(self.current_chunk_data)
        print(f"  保存chunk {self.current_chunk}: {chunk_size} 个子图...")
        sys.stdout.flush()
        
        try:
            torch.save(self.current_chunk_data, chunk_file)
            self.chunk_files.append(chunk_file)
            # 清空current_chunk_data并强制释放内存
            self.current_chunk_data = {}
            self.current_chunk += 1
            print(f"  ✓ chunk {self.current_chunk-1} 保存成功: {chunk_file.name}")
            sys.stdout.flush()
        except Exception as e:
            print(f"  ✗ chunk保存失败: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
    
    def finalize(self, metadata: Dict, merge_chunks: bool = False, existing_subgraphs: Optional[Dict] = None, existing_subgraphs_file_path: Optional[str] = None) -> Path:
        """
        最终保存（可选择是否合并chunks）
        
        Args:
            metadata: 元数据
            merge_chunks: 是否合并chunks（默认False，避免内存峰值）
            existing_subgraphs: 已存在的子图（用于合并时包含）
        """
        # 保存最后一个chunk
        if self.current_chunk_data:
            self.save_chunk()
        
        if not self.chunk_files:
            return None
        
        output_file = self.output_dir / f"subgraphs_{self.split_name}.pt"
        
        if merge_chunks:
            # 合并所有chunks（分批处理，避免内存峰值）
            print(f"\n合并 {len(self.chunk_files)} 个chunks到最终文件（分批处理）...")
            if existing_subgraphs:
                print(f"  同时合并 {len(existing_subgraphs)} 个已存在的子图...")
            sys.stdout.flush()
            
            # 先添加已存在的子图
            all_subgraphs = {}
            if existing_subgraphs:
                all_subgraphs.update(existing_subgraphs)
            
            # 分批合并，每次只加载2个chunks
            batch_size = 2
            total_size = len(all_subgraphs)
            
            for i in range(0, len(self.chunk_files), batch_size):
                batch_files = self.chunk_files[i:i+batch_size]
                print(f"  合并批次 {i//batch_size + 1}/{(len(self.chunk_files) + batch_size - 1)//batch_size}: "
                      f"{len(batch_files)} 个chunks...")
                sys.stdout.flush()
                
                batch_data = {}
                for chunk_file in batch_files:
                    chunk_data = torch.load(chunk_file, map_location='cpu')
                    batch_data.update(chunk_data)
                    total_size += len(chunk_data)
                
                all_subgraphs.update(batch_data)
                # 释放batch_data内存
                del batch_data
                import gc
                gc.collect()
            
            # 保存最终文件
            save_data = {
                'subgraphs': all_subgraphs,
                'metadata': metadata
            }
            
            temp_file = output_file.with_suffix('.tmp')
            torch.save(save_data, temp_file)
            import shutil
            shutil.move(str(temp_file), str(output_file))
            
            # 清理chunk文件
            for chunk_file in self.chunk_files:
                chunk_file.unlink()
            self.chunk_dir.rmdir()
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"合并完成: {total_size} 个子图, {file_size_mb:.1f} MB")
            sys.stdout.flush()
        else:
            # 不合并，只保存metadata和chunk文件列表
            print(f"\n保存 {len(self.chunk_files)} 个chunks的元数据（不合并，避免内存峰值）...")
            if existing_subgraphs:
                print(f"  同时保存 {len(existing_subgraphs)} 个已存在子图的引用...")
            sys.stdout.flush()
            
            # 计算总子图数（包括已存在的和新构建的）
            # 优化：不一次性加载所有chunks，而是从已保存的chunk中累加
            # 使用current_chunk_data和已保存的chunk文件数量来估算
            chunk_subgraphs_count = len(self.current_chunk_data)  # 当前chunk中的子图数
            # 已保存的chunks：每个chunk约1000个子图（除了最后一个）
            chunk_subgraphs_count += (len(self.chunk_files) - 1) * self.chunk_size
            # 最后一个chunk可能不满1000个，需要加载（但只有一个文件，内存占用小）
            if len(self.chunk_files) > 0:
                try:
                    last_chunk = torch.load(self.chunk_files[-1], map_location='cpu')
                    chunk_subgraphs_count += len(last_chunk)
                    del last_chunk  # 立即释放内存
                    import gc
                    gc.collect()
                except:
                    # 如果加载失败，使用估算值
                    pass
            
            total_subgraphs = chunk_subgraphs_count
            if existing_subgraphs:
                total_subgraphs += len(existing_subgraphs)
            
            # 如果有已存在的子图或原文件路径，先备份原文件（避免覆盖）
            existing_subgraphs_file = None
            # 优先使用传入的原文件路径
            if existing_subgraphs_file_path and Path(existing_subgraphs_file_path).exists():
                # 如果提供了原文件路径，备份它
                backup_file = output_file.parent / f"subgraphs_{self.split_name}_backup.pt"
                # 如果备份文件已存在，跳过（避免重复备份）
                if not backup_file.exists():
                    import shutil
                    shutil.copy2(existing_subgraphs_file_path, str(backup_file))
                    existing_subgraphs_file = str(backup_file)
                    print(f"  已备份原文件到: {backup_file.name} (从 {Path(existing_subgraphs_file_path).name})")
                else:
                    existing_subgraphs_file = str(backup_file)
                    print(f"  备份文件已存在: {backup_file.name}")
            # 如果output_file存在且还没有备份，也备份它（防止覆盖）
            elif output_file.exists() and not (output_file.parent / f"subgraphs_{self.split_name}_backup.pt").exists():
                backup_file = output_file.parent / f"subgraphs_{self.split_name}_backup.pt"
                import shutil
                shutil.copy2(str(output_file), str(backup_file))
                existing_subgraphs_file = str(backup_file)
                print(f"  已备份原文件到: {backup_file.name} (从 {output_file.name})")
            elif existing_subgraphs:
                # 如果原文件存在且包含subgraphs，备份它
                if output_file.exists():
                    # 备份原文件
                    backup_file = output_file.parent / f"subgraphs_{self.split_name}_backup.pt"
                    import shutil
                    shutil.copy2(str(output_file), str(backup_file))
                    existing_subgraphs_file = str(backup_file)
                    print(f"  已备份原文件到: {backup_file.name}")
                else:
                    # 如果原文件不存在，创建一个包含已存在子图的文件
                    existing_file = output_file.parent / f"subgraphs_{self.split_name}_existing.pt"
                    torch.save({'subgraphs': existing_subgraphs}, existing_file)
                    existing_subgraphs_file = str(existing_file)
            
            # 重要：合并所有chunk文件（包括之前运行的）
            # 检查chunks目录中是否还有其他chunk文件（之前运行留下的）
            all_chunk_files = sorted(self.chunk_dir.glob('chunk_*.pt'))
            # 确保包含所有chunk文件，而不仅仅是当前运行的
            final_chunk_files = [str(f) for f in all_chunk_files]
            
            # 重新计算总子图数（包括所有chunk文件）
            # ⚠️ 优化：逐个加载chunk文件，加载后立即释放，避免一次性加载所有chunk到内存
            total_from_all_chunks = 0
            for chunk_file in all_chunk_files:
                try:
                    chunk_data = torch.load(chunk_file, map_location='cpu')
                    total_from_all_chunks += len(chunk_data)
                    # 立即释放chunk_data内存
                    del chunk_data
                    import gc
                    gc.collect()
                except:
                    pass
            
            # 如果existing_subgraphs不为空，也要加上
            if existing_subgraphs:
                total_from_all_chunks += len(existing_subgraphs)
            
            save_data = {
                'chunk_files': final_chunk_files,  # 使用所有chunk文件
                'chunk_dir': str(self.chunk_dir),
                'metadata': metadata,
                'num_subgraphs': total_from_all_chunks,  # 使用重新计算的总数
                'existing_subgraphs_file': existing_subgraphs_file  # 已存在子图的文件路径（备份）
            }
            
            temp_file = output_file.with_suffix('.tmp')
            torch.save(save_data, temp_file)
            import shutil
            shutil.move(str(temp_file), str(output_file))
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"元数据保存完成: {save_data['num_subgraphs']} 个子图分布在 {len(self.chunk_files)} 个chunks, "
                  f"元数据文件 {file_size_mb:.1f} MB")
            print(f"  Chunk目录: {self.chunk_dir}")
            sys.stdout.flush()
        
        return output_file


# ==================== 优化的子图构建函数 ====================

# 全局变量用于进程间共享（优化后）
_shared_kg_data = None
_precomputed_indices = None

def _init_worker_optimized(kg_file_str: str, cache_dir_str: str, shared_memory_name: Optional[str] = None, shared_memory_size: Optional[int] = None):
    """初始化工作进程（优化版，使用共享内存）"""
    global _shared_kg_data, _precomputed_indices
    
    # 转换字符串路径为Path对象
    kg_file = Path(kg_file_str)
    cache_dir = Path(cache_dir_str)
    
    # 加载共享KG（优先从共享内存，失败则从文件）
    loader = SharedKGLoader(kg_file, cache_dir)
    _shared_kg_data = loader.load_shared_kg(shared_memory_name, shared_memory_size)
    
    # 初始化预计算索引
    _precomputed_indices = PrecomputedIndices(_shared_kg_data)
    _precomputed_indices.compute_indices()
    
    print(f"  子进程初始化完成（使用共享内存KG和预计算索引）")
    sys.stdout.flush()


def build_single_subgraph_optimized(args_tuple: Tuple) -> Tuple[str, Any, int]:
    """
    构建单个子图（优化版，使用预计算索引）
    
    Args:
        args_tuple: (window_id, num_hops, max_neighbors, allowed_windows) 或 (window_id, num_hops, max_neighbors, allowed_windows, label_override)
        当提供 label_override 时优先使用（主进程从KG的HAS_ANOMALY边获取的正确标签），
        因为 Window 节点的 properties 中可能不包含 anomaly_types。
        
    Returns:
        Tuple[window_id, hetero_data, label]
    """
    import time
    global _shared_kg_data, _precomputed_indices
    
    label_override = None
    if len(args_tuple) == 5:
        window_id, num_hops, max_neighbors, allowed_windows, label_override = args_tuple
    else:
        window_id, num_hops, max_neighbors, allowed_windows = args_tuple
    
    # 记录任务开始执行时间（不是提交时间）
    task_start_time = time.time()
    
    try:
        # 创建一个临时的GlobalKGDataLoader实例，使用共享数据
        # 注意：这里需要修改GlobalKGDataLoader以支持从预加载数据初始化
        # 为了简化，我们直接使用共享数据构建子图
        
        # 使用预计算索引快速获取窗口的实体
        target_entity_set = _precomputed_indices.get_window_entities(window_id)
        if not target_entity_set:
            execution_time = time.time() - task_start_time
            return (window_id, None, -1, execution_time)
        
        # 快速找到候选窗口（使用预计算索引）
        candidate_windows = {window_id}
        for entity_id in target_entity_set:
            entity_windows = _precomputed_indices.get_entity_windows(entity_id)
            for other_window_id in entity_windows:
                if allowed_windows is None or other_window_id in allowed_windows:
                    candidate_windows.add(other_window_id)
        
        # 限制候选窗口数（使用预计算的相似度）
        if len(candidate_windows) > max_neighbors:
            other_windows = list(candidate_windows - {window_id})
            
            # 使用预计算的相似度（O(1)查找）
            window_scores = []
            for other_window_id in other_windows:
                score = _precomputed_indices.compute_window_similarity(window_id, other_window_id)
                window_scores.append((other_window_id, score))
            
            window_scores.sort(key=lambda x: x[1], reverse=True)
            selected_windows = [wid for wid, _ in window_scores[:max_neighbors-1]]
            expanded_window_ids = {window_id} | set(selected_windows)
        else:
            expanded_window_ids = candidate_windows
        
        # 获取窗口的日志
        window_to_logs = _shared_kg_data['window_to_logs']
        log_to_window = _shared_kg_data['log_to_window']
        entity_to_logs_raw = _shared_kg_data['entity_to_logs']  # 可能是list，需要转换为set
        edges_dict = _shared_kg_data['edges_dict']
        nodes_dict = _shared_kg_data['nodes_dict']
        
        # 转换entity_to_logs为set（如果存储的是list）
        entity_to_logs = {}
        for entity_id, log_list in entity_to_logs_raw.items():
            entity_to_logs[entity_id] = set(log_list) if isinstance(log_list, list) else log_list
        
        # 扩展日志（只扩展选定窗口的日志）
        expanded_log_ids = set()
        for wid in expanded_window_ids:
            expanded_log_ids.update(window_to_logs.get(wid, []))
        
        # 扩展实体（从Window的CONTAINS边获取，而不是从Log的ASSOCIATED_WITH边）
        expanded_entity_ids = set()
        # 方法1：从Window的CONTAINS边获取Entity（新版本KG结构）
        for wid in expanded_window_ids:
            window_edges = edges_dict.get(wid, [])
            for target, relation, props in window_edges:
                if relation == 'CONTAINS':
                    target_node = nodes_dict.get(target)
                    if target_node and target_node.get('type') in ['DataRegion', 'Node', 'ConsensusGroup', 'Thread', 'GeneralEntity', 'Anomaly']:
                        expanded_entity_ids.add(target)
        
        # 方法2：从Log的ASSOCIATED_WITH边获取Entity（如果存在，兼容旧版本KG结构）
        for log_id in expanded_log_ids:
            edges = edges_dict.get(log_id, [])
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH':
                    expanded_entity_ids.add(target)
        
        # 检查规模限制
        MAX_TOTAL_NODES = 200000
        if len(expanded_log_ids) + len(expanded_entity_ids) + len(expanded_window_ids) > MAX_TOTAL_NODES:
            execution_time = time.time() - task_start_time
            return (window_id, None, -1, execution_time)
        
        # 获取标签（优先使用主进程传入的 label_override，因为 Window 节点 properties 中通常不包含 anomaly_types）
        ANOMALY_TYPE_TO_LABEL = {
            'compaction': 0,
            'export': 1,
            'flush': 2,
            'full_cpu': 3,
            'full_memory': 4,
            'network_bandwidth2': 5,
            'normal': 6
        }
        if label_override is not None and label_override >= 0:
            label = label_override
        else:
            # 回退：从 Window 的 HAS_ANOMALY 边获取（与 get_window_label 逻辑一致）
            window_node = nodes_dict.get(window_id)
            label = -1
            if window_node:
                anomaly_types = window_node.get('properties', {}).get('anomaly_types', [])
                if anomaly_types:
                    anomaly_type = anomaly_types[0]
                    label = ANOMALY_TYPE_TO_LABEL.get(anomaly_type, -1)
                else:
                    # 从 HAS_ANOMALY 边获取
                    for target, relation, _ in edges_dict.get(window_id, []):
                        if relation == 'HAS_ANOMALY':
                            anomaly_node = nodes_dict.get(target)
                            if anomaly_node:
                                at = anomaly_node.get('properties', {}).get('anomaly_type', 'normal')
                                label = ANOMALY_TYPE_TO_LABEL.get(at, -1)
                                break
                    if label < 0:
                        label = ANOMALY_TYPE_TO_LABEL.get('normal', 6)
        
        # 构建边索引
        window_local_idx = {wid: i for i, wid in enumerate(expanded_window_ids)}
        log_local_idx = {lid: i for i, lid in enumerate(expanded_log_ids)}
        entity_local_idx = {eid: i for i, eid in enumerate(expanded_entity_ids)}
        
        # 预建立log->entity映射
        # 方法1：通过Window的CONTAINS边建立映射（新版本KG结构）
        log_to_entities = {}
        for wid in expanded_window_ids:
            window_edges = edges_dict.get(wid, [])
            window_logs = window_to_logs.get(wid, [])
            window_entities = []
            for target, relation, props in window_edges:
                if relation == 'CONTAINS' and target in entity_local_idx:
                    window_entities.append(target)
            # 将该窗口的所有实体关联到该窗口的所有日志
            for log_id in window_logs:
                if log_id in log_local_idx:
                    if log_id not in log_to_entities:
                        log_to_entities[log_id] = []
                    log_to_entities[log_id].extend(window_entities)
        
        # 方法2：通过Log的ASSOCIATED_WITH边（如果存在，兼容旧版本KG结构）
        for log_id in expanded_log_ids:
            edges = edges_dict.get(log_id, [])
            entities = []
            for target, relation, props in edges:
                if relation == 'ASSOCIATED_WITH' and target in entity_local_idx:
                    if log_id not in log_to_entities:
                        log_to_entities[log_id] = []
                    if target not in log_to_entities[log_id]:
                        log_to_entities[log_id].append(target)
        
        # 构建边
        window_to_log_edges = []
        log_to_entity_edges = []
        entity_to_log_edges = []
        
        for wid in expanded_window_ids:
            w_local_idx = window_local_idx[wid]
            window_logs = window_to_logs.get(wid, [])
            for log_id in window_logs:
                if log_id in log_local_idx:
                    l_local_idx = log_local_idx[log_id]
                    window_to_log_edges.append([w_local_idx, l_local_idx])
                    
                    if log_id in log_to_entities:
                        for entity_id in log_to_entities[log_id]:
                            e_local_idx = entity_local_idx[entity_id]
                            log_to_entity_edges.append([l_local_idx, e_local_idx])
                            entity_to_log_edges.append([e_local_idx, l_local_idx])
        
        # 构建HeteroData
        from torch_geometric.data import HeteroData
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
        
        # 存储元数据
        data.target_window_id = window_id
        data.target_window_idx = window_local_idx.get(window_id, 0)
        data.window_ids = list(expanded_window_ids)
        data.log_ids = list(expanded_log_ids)
        data.entity_ids = list(expanded_entity_ids)
        data.label = label
        
        # 存储模板文本和实体内容
        log_template_texts = {}
        for log_id in expanded_log_ids:
            log_node = nodes_dict.get(log_id)
            if log_node:
                raw_line = log_node.get('properties', {}).get('raw_line', '')
                template_id = log_node.get('properties', {}).get('template_id', '')
                if raw_line:
                    log_template_texts[log_id] = raw_line
                elif template_id:
                    log_template_texts[log_id] = template_id
        
        entity_contents = {}
        for entity_id in expanded_entity_ids:
            entity_node = nodes_dict.get(entity_id)
            if entity_node:
                if entity_node.get('type') == 'GeneralEntity':
                    token = entity_node.get('properties', {}).get('token', '')
                    if token:
                        entity_contents[entity_id] = token
                    else:
                        entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
                elif entity_node.get('type') == 'Thread':
                    thread_name = entity_node.get('properties', {}).get('thread_name', '')
                    if thread_name:
                        entity_contents[entity_id] = thread_name
                    else:
                        entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
                else:
                    entity_contents[entity_id] = entity_id.split(':', 1)[-1] if ':' in entity_id else entity_id
        
        data.log_template_texts = log_template_texts
        data.entity_contents = entity_contents
        
        # 确保在CPU上
        data = data.cpu()
        
        # 计算执行时间（从任务开始执行到完成的时间）
        execution_time = time.time() - task_start_time
        
        return (window_id, data, label, execution_time)
        
    except MemoryError as e:
        import traceback
        execution_time = time.time() - task_start_time
        print(f"  内存错误: 窗口 {window_id} - {str(e)[:200]}", file=sys.stderr)
        print(f"  节点数: Window={len(expanded_window_ids)}, Log={len(expanded_log_ids)}, Entity={len(expanded_entity_ids)}", file=sys.stderr)
        sys.stderr.flush()
        return (window_id, None, -1, execution_time)
    except Exception as e:
        import traceback
        execution_time = time.time() - task_start_time
        print(f"  异常: 窗口 {window_id} - {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        print(f"  节点数: Window={len(expanded_window_ids) if 'expanded_window_ids' in locals() else 'N/A'}, Log={len(expanded_log_ids) if 'expanded_log_ids' in locals() else 'N/A'}, Entity={len(expanded_entity_ids) if 'expanded_entity_ids' in locals() else 'N/A'}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return (window_id, None, -1, execution_time)


def build_and_save_subgraphs_optimized(
    kg_file: Path,
    window_ids: List[str],
    labels: List[int],
    output_dir: Path,
    num_hops: int = 1,
    max_neighbors: int = 10,
    allowed_windows: Optional[Set[str]] = None,
    split_name: str = "train",
    num_workers: int = 4,
    chunk_size: int = 5000,
    max_subgraphs_per_run: Optional[int] = None
):
    """
    构建并保存子图（优化版）
    
    优化点：
    1. 使用共享KG缓存，避免每个进程重复加载
    2. 使用预计算索引，加速窗口选择
    3. 使用增量保存，避免重复序列化
    """
    print(f"\n{'='*80}")
    print(f"构建 {split_name} 集的子图（优化版）")
    print(f"{'='*80}")
    print(f"窗口数: {len(window_ids)}")
    print(f"参数: num_hops={num_hops}, max_neighbors={max_neighbors}")
    sys.stdout.flush()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备共享KG缓存
    print(f"\n1. 准备共享KG缓存...")
    sys.stdout.flush()
    shared_loader = SharedKGLoader(kg_file)
    shared_loader.prepare_shared_kg()
    
    # ⚠️ 重要：主进程在prepare_shared_kg()中已经加载了KG到内存（用于序列化）
    # 这里需要立即释放主进程中的KG数据，避免重复占用内存
    # 因为worker进程会从共享内存加载KG，主进程不需要保留KG副本
    if shared_loader.kg_data is not None:
        del shared_loader.kg_data
        shared_loader.kg_data = None
        import gc
        gc.collect()
        print(f"  主进程已释放KG数据，避免重复占用内存")
        sys.stdout.flush()
    
    # 检查已存在的子图（断点续传）
    # ⚠️ 重要：以磁盘上的 chunks 目录为准，不依赖 pt 文件的 chunk_files 列表（pt 可能过期/不完整）
    output_file = output_dir / f"subgraphs_{split_name}.pt"
    existing_subgraphs = {}
    existing_windows = set()
    existing_subgraphs_file_path = None  # 保存原文件路径，用于finalize时备份
    chunk_dir = output_dir / f"chunks_{split_name}"
    
    # 1. 优先从磁盘 chunks 目录加载（唯一可信来源）
    if chunk_dir.exists():
        chunk_files_on_disk = sorted(chunk_dir.glob('chunk_*.pt'))
        if chunk_files_on_disk:
            print(f"  从磁盘 chunks 目录统计已构建子图: {chunk_dir}（共 {len(chunk_files_on_disk)} 个chunk文件）")
            sys.stdout.flush()
            for chunk_file in chunk_files_on_disk:
                try:
                    chunk_data = torch.load(str(chunk_file), map_location='cpu')
                    if isinstance(chunk_data, dict):
                        existing_windows.update(chunk_data.keys())
                    elif hasattr(chunk_data, 'keys'):
                        existing_windows.update(chunk_data.keys())
                    del chunk_data
                    import gc
                    gc.collect()
                except Exception as e:
                    print(f"  警告: 加载chunk文件 {chunk_file.name} 失败: {str(e)[:100]}")
                    sys.stdout.flush()
            if existing_windows:
                print(f"  已从磁盘统计 {len(existing_windows)} 个已构建的子图（以磁盘为准）")
                sys.stdout.flush()
            if output_file.exists():
                existing_subgraphs_file_path = str(output_file)
    
    # 2. 若磁盘无 chunks，则从 pt 文件加载（旧格式或合并格式）
    if not existing_windows and output_file.exists():
        try:
            print(f"  检测到已存在的子图文件（无chunks目录），尝试从 pt 加载...")
            sys.stdout.flush()
            saved_data = torch.load(output_file, map_location='cpu')
            if isinstance(saved_data, dict):
                if 'subgraphs' in saved_data:
                    existing_subgraphs = saved_data['subgraphs']
                    existing_windows = set(existing_subgraphs.keys())
                    existing_subgraphs_file_path = str(output_file)
                elif 'chunk_files' in saved_data:
                    chunk_files = saved_data.get('chunk_files', [])
                    chunk_dir_from_pt = saved_data.get('chunk_dir', '')
                    existing_subgraphs_file_path = str(output_file)
                    if 'existing_subgraphs_file' in saved_data:
                        existing_subgraphs_file_path = saved_data['existing_subgraphs_file']
                    for chunk_file in chunk_files:
                        try:
                            chunk_file_path = Path(chunk_file)
                            if not chunk_file_path.is_absolute() and chunk_dir_from_pt:
                                chunk_file_path = Path(chunk_dir_from_pt) / chunk_file_path.name
                            chunk_data = torch.load(str(chunk_file_path), map_location='cpu')
                            if isinstance(chunk_data, dict):
                                existing_windows.update(chunk_data.keys())
                            elif hasattr(chunk_data, 'keys'):
                                existing_windows.update(chunk_data.keys())
                        except Exception:
                            pass
                else:
                    existing_subgraphs = saved_data
                    existing_subgraphs_file_path = str(output_file)
            else:
                existing_subgraphs = saved_data
                existing_subgraphs_file_path = str(output_file)
            if not existing_windows and existing_subgraphs:
                existing_windows = set(existing_subgraphs.keys())
            if existing_windows:
                print(f"  已从 pt 加载 {len(existing_windows)} 个已构建的子图")
                sys.stdout.flush()
        except Exception as e:
            print(f"  警告: 加载已存在文件失败: {str(e)[:100]}")
            existing_subgraphs_file_path = str(output_file)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    remaining_window_ids = [wid for wid in window_ids if wid not in existing_windows]
    remaining_labels = [labels[i] for i, wid in enumerate(window_ids) if wid not in existing_windows]
    
    if len(existing_windows) > 0:
        print(f"  断点续传: 跳过 {len(existing_windows)} 个已构建窗口，剩余 {len(remaining_window_ids)} 个待构建")
        sys.stdout.flush()
    
    if not remaining_window_ids:
        print("  所有窗口已构建完成！")
        sys.stdout.flush()
        # 即使所有窗口都已构建完成，也要备份原文件（防止后续运行时覆盖）
        if existing_subgraphs_file_path and Path(existing_subgraphs_file_path).exists():
            backup_file = output_dir / f"subgraphs_{split_name}_backup.pt"
            if not backup_file.exists():
                import shutil
                shutil.copy2(existing_subgraphs_file_path, str(backup_file))
                print(f"  已备份原文件到: {backup_file.name}")
                sys.stdout.flush()
        return output_file, {}
    
    # 初始化增量保存器
    incremental_saver = IncrementalSaver(output_dir, split_name, chunk_size=chunk_size)
    
    # 记录本次运行开始时的chunk文件数（用于计算本次运行新增的子图数）
    initial_chunk_count = len(incremental_saver.chunk_files)
    initial_chunk_data_count = len(incremental_saver.current_chunk_data)
    
    # 并行构建
    use_parallel = len(remaining_window_ids) >= 500 and num_workers > 1
    # ⚠️ 重要：每个进程都会加载完整的KG副本（约8.7GB）
    # 实际内存占用 = 进程数 × 8.7GB
    # 5个进程 = 约44GB内存
    MAX_SAFE_WORKERS = 5
    if num_workers > MAX_SAFE_WORKERS:
        num_workers = MAX_SAFE_WORKERS
    
    if use_parallel:
        print(f"\n2. 使用 {num_workers} 个并行进程构建子图...")
        print(f"  注意: 使用共享KG缓存，进程启动时间大幅减少")
        print(f"  调试: num_workers={num_workers}, MAX_SAFE_WORKERS={MAX_SAFE_WORKERS}")
        sys.stdout.flush()
        
        args_list = [
            (window_id, num_hops, max_neighbors, allowed_windows, remaining_labels[i])
            for i, window_id in enumerate(remaining_window_ids)
        ]
        
        completed = 0
        total_to_build = len(remaining_window_ids)
        start_time = time.time()
        
        # 传递共享内存信息给worker进程
        init_args = (
            str(kg_file), 
            str(shared_loader.cache_dir),
            shared_loader.shared_memory_name,
            shared_loader.shared_memory_size
        )
        
        # 全局变量，用于信号处理时清理
        executor_ref = [None]
        
        def cleanup_executor():
            """清理executor和worker进程"""
            if executor_ref[0] is not None:
                try:
                    print(f"\n  正在清理worker进程...")
                    sys.stdout.flush()
                    executor_ref[0].shutdown(wait=False, cancel_futures=True)
                    # 强制终止所有worker进程
                    import psutil
                    current_process = psutil.Process(os.getpid())
                    for child in current_process.children(recursive=True):
                        try:
                            child.terminate()
                        except:
                            pass
                    # 等待进程退出
                    gone, alive = psutil.wait_procs(current_process.children(recursive=True), timeout=5)
                    for p in alive:
                        try:
                            p.kill()
                        except:
                            pass
                    print(f"  ✓ worker进程已清理")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"  警告: 清理worker进程时出错: {e}")
                    sys.stdout.flush()
        
        def signal_handler(signum, frame):
            """信号处理函数，确保退出时清理worker进程"""
            print(f"\n  收到信号 {signum}，正在清理...")
            sys.stdout.flush()
            cleanup_executor()
            sys.exit(1)
        
        # 注册信号处理和退出处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup_executor)
        
        executor = None
        try:
            print(f"  调试: 创建ProcessPoolExecutor，max_workers={num_workers}")
            sys.stdout.flush()
            executor = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker_optimized,
                initargs=init_args
            )
            executor_ref[0] = executor
            print(f"  调试: ProcessPoolExecutor创建完成，实际worker数应该为{num_workers}")
            sys.stdout.flush()
            
            future_to_window = {
                executor.submit(build_single_subgraph_optimized, args): args[0]
                for args in args_list
            }
            
            failed_count = 0
            should_stop = False  # 标志：是否达到限制需要停止
            last_progress_time = time.time()  # 记录上次进度更新时间
            
            # 使用循环处理
            pending_futures = set(future_to_window.keys())
            
            while pending_futures and not should_stop:
                # 使用wait()检查已完成的future（非阻塞）
                if pending_futures:
                    done, not_done = wait(pending_futures, timeout=10, return_when='FIRST_COMPLETED')
                    
                    # 处理已完成的future
                    for future in done:
                        if should_stop:
                            break
                        
                        pending_futures.discard(future)
                        
                        try:
                            result = future.result(timeout=1)  # 快速获取结果
                            # 处理新的返回值格式（包含执行时间，兼容旧格式）
                            if len(result) == 4:
                                window_id, hetero_data, label, _ = result
                            else:
                                window_id, hetero_data, label = result
                            
                            completed += 1
                            last_progress_time = time.time()  # 更新进度时间
                            
                            if completed % 100 == 0 or completed == total_to_build:
                                elapsed = time.time() - start_time
                                if completed > 0:
                                    avg_time = elapsed / completed
                                    remaining = avg_time * (total_to_build - completed)
                                    # 添加内存使用信息
                                    try:
                                        import psutil
                                        process = psutil.Process(os.getpid())
                                    except ImportError:
                                        psutil = None
                                    if psutil:
                                        mem_info = process.memory_info()
                                        mem_mb = mem_info.rss / (1024 * 1024)
                                        mem_str = f", 主进程内存: {mem_mb/1024:.1f}GB"
                                    else:
                                        mem_str = ""
                                    print(f"  进度: {completed}/{total_to_build} ({100*completed/total_to_build:.1f}%) - "
                                          f"已用时间: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟, "
                                          f"失败: {failed_count}{mem_str}, "
                                          f"当前chunk数据: {len(incremental_saver.current_chunk_data)}个")
                                    sys.stdout.flush()
                            
                            if hetero_data is not None:
                                incremental_saver.add_subgraph(window_id, hetero_data)
                                
                                # 检查是否达到本次运行的最大子图数
                                if max_subgraphs_per_run is not None:
                                    # 计算当前新构建的子图数（只计算本次运行新增的）
                                    # 优化：不加载chunk文件内容，只计算数量
                                    new_count = len(incremental_saver.current_chunk_data)
                                    # 只计算本次运行新创建的chunk文件数量（每个chunk约chunk_size个子图）
                                    new_chunk_count = len(incremental_saver.chunk_files) - initial_chunk_count
                                    new_count += new_chunk_count * incremental_saver.chunk_size
                                    
                                    if new_count >= max_subgraphs_per_run:
                                        print(f"\n  达到本次运行的最大子图数限制（{max_subgraphs_per_run}），保存进度并退出...")
                                        print(f"  当前已构建: {new_count} 个新子图（本次运行）")
                                        sys.stdout.flush()
                                        # 取消剩余任务
                                        for f in pending_futures:
                                            if not f.done():
                                                f.cancel()
                                        # 设置标志，跳出循环
                                        should_stop = True
                                        break
                            else:
                                failed_count += 1
                        except FutureTimeoutError:
                            window_id = future_to_window.get(future, "unknown")
                            print(f"  ⚠️ 警告: 窗口 {window_id} 获取结果失败")
                            print(f"    窗口ID: {window_id}, Future状态: done={future.done()}, cancelled={future.cancelled()}")
                            sys.stdout.flush()
                            completed += 1
                            failed_count += 1
                        except Exception as e:
                            window_id = future_to_window.get(future, "unknown")
                            error_msg = str(e)
                            print(f"  ⚠️ 警告: 窗口 {window_id} 处理失败")
                            print(f"    窗口ID: {window_id}")
                            if "terminated abruptly" in error_msg.lower():
                                print(f"    错误类型: 进程被系统终止（可能是内存不足OOM）")
                            else:
                                print(f"    错误类型: {type(e).__name__}")
                                print(f"    错误信息: {error_msg[:200]}")
                            sys.stdout.flush()
                            completed += 1
                            failed_count += 1
                
                # 检查是否长时间没有进度更新（超过5分钟）
                current_time = time.time()
                if current_time - last_progress_time > 300:  # 5分钟
                    remaining_count = len(pending_futures)
                    print(f"  ⚠️ 警告: 已超过5分钟没有进度更新（上次更新: {completed}个），正在等待的任务数: {remaining_count}")
                    sys.stdout.flush()
                    last_progress_time = current_time  # 更新，避免重复打印
            
            print(f"\n  处理完成: 成功 {completed - failed_count}, 失败 {failed_count}, 总计 {completed}")
            sys.stdout.flush()
        except KeyboardInterrupt:
            print(f"\n  收到中断信号，正在清理...")
            sys.stdout.flush()
            raise
        except Exception as e:
            print(f"\n  发生异常: {e}，正在清理...")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 确保executor正确关闭，这会等待所有worker进程完成
            if executor is not None:
                try:
                    print(f"  正在关闭executor...")
                    sys.stdout.flush()
                    executor.shutdown(wait=True)
                    print(f"  ✓ executor已关闭")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"  警告: 关闭executor时出错: {e}")
                    sys.stdout.flush()
                executor_ref[0] = None
            
            # 清理共享内存
            if shared_loader.shared_memory is not None:
                try:
                    shared_loader.shared_memory.close()
                    shared_loader.shared_memory.unlink()
                    print(f"  已清理共享内存: {shared_loader.shared_memory_name}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"  警告: 清理共享内存失败: {e}")
                    sys.stdout.flush()
    else:
        print(f"\n2. 使用串行模式构建子图...")
        sys.stdout.flush()
        
        # 串行模式：加载共享KG和预计算索引
        shared_loader = SharedKGLoader(kg_file)
        kg_data = shared_loader.load_shared_kg()
        precomputed_indices = PrecomputedIndices(kg_data)
        precomputed_indices.compute_indices()
        
        # 全局变量设置（用于串行模式）
        global _shared_kg_data, _precomputed_indices
        _shared_kg_data = kg_data
        _precomputed_indices = precomputed_indices
        
        completed = 0
        total_to_build = len(remaining_window_ids)
        start_time = time.time()
        
        for i, window_id in enumerate(remaining_window_ids):
            if (i + 1) % 10 == 0 or (i + 1) == total_to_build:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (total_to_build - i - 1)
                    print(f"  进度: {i+1}/{total_to_build} ({100*(i+1)/total_to_build:.1f}%) - "
                          f"已用时间: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟")
                    sys.stdout.flush()
            
            result = build_single_subgraph_optimized(
                (window_id, num_hops, max_neighbors, allowed_windows, remaining_labels[i])
            )
            window_id, hetero_data, label = result[:3]  # 兼容3或4个返回值
            completed += 1
            
            if hetero_data is not None:
                incremental_saver.add_subgraph(window_id, hetero_data)
                
                # 检查是否达到本次运行的最大子图数
                if max_subgraphs_per_run is not None:
                    # 计算当前新构建的子图数（只计算本次运行新增的）
                    # 优化：不加载chunk文件内容，只计算数量
                    new_count = len(incremental_saver.current_chunk_data)
                    # 只计算本次运行新创建的chunk文件数量（每个chunk约chunk_size个子图）
                    new_chunk_count = len(incremental_saver.chunk_files) - initial_chunk_count
                    new_count += new_chunk_count * incremental_saver.chunk_size
                    
                    if new_count >= max_subgraphs_per_run:
                        print(f"\n  达到本次运行的最大子图数限制（{max_subgraphs_per_run}），保存进度并退出...")
                        print(f"  当前已构建: {new_count} 个新子图（本次运行）")
                        sys.stdout.flush()
                        break
    
    # 最终保存（不合并chunks，避免内存峰值）
    print(f"\n3. 最终保存子图...")
    sys.stdout.flush()
    
    # 计算总成功数（包括已存在的和新构建的）
    # ⚠️ 优化：不加载chunk文件内容，只使用文件数量估算，避免内存峰值
    # 如果需要精确数量，可以在finalize()中计算
    total_success = len(existing_subgraphs) + len(incremental_saver.current_chunk_data)
    # 估算已保存的chunk文件中的子图数（每个chunk约chunk_size个子图）
    saved_chunk_count = len(incremental_saver.chunk_files) - initial_chunk_count
    total_success += saved_chunk_count * incremental_saver.chunk_size
    
    # 计算本次运行新增的子图数（只计算本次运行新创建的chunk文件）
    new_subgraphs_count = len(incremental_saver.current_chunk_data)
    new_subgraphs_count += saved_chunk_count * incremental_saver.chunk_size
    
    # 如果达到最大子图数限制，提示用户重新运行
    if max_subgraphs_per_run is not None and new_subgraphs_count >= max_subgraphs_per_run:
        print(f"\n{'='*80}")
        print(f"本次运行已完成 {new_subgraphs_count} 个子图（达到限制 {max_subgraphs_per_run}）")
        print(f"已保存进度，请重新运行程序继续构建剩余子图")
        print(f"{'='*80}")
        sys.stdout.flush()
    
    metadata = {
        'num_hops': num_hops,
        'max_neighbors': max_neighbors,
        'num_windows': len(window_ids),
        'num_success': total_success,
        'window_to_label': {wid: labels[i] for i, wid in enumerate(window_ids)}
    }
    
    # 不合并chunks，避免内存峰值（训练时可以按需加载chunks）
    # 但是需要保存已存在的子图信息，以便后续加载
    final_file = incremental_saver.finalize(metadata, merge_chunks=False, existing_subgraphs=existing_subgraphs, existing_subgraphs_file_path=existing_subgraphs_file_path)
    
    elapsed = time.time() - start_time
    print(f"\n构建完成！总耗时: {elapsed/60:.1f}分钟")
    sys.stdout.flush()
    
    return final_file or output_file, metadata


def main():
    parser = argparse.ArgumentParser(description='预构建子图（优化版）')
    parser.add_argument('--kg-file', type=str, required=True,
                       help='知识图谱文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num-hops', type=int, default=1,
                       help='采样跳数')
    parser.add_argument('--max-neighbors', type=int, default=10,
                       help='最大邻居窗口数')
    parser.add_argument('--max-windows', type=int, default=None,
                       help='最大窗口数（用于测试）')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='并行进程数（默认：CPU核心数）')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='每个chunk的子图数量（默认：1000）')
    parser.add_argument('--max-subgraphs-per-run', type=int, default=None,
                       help='每次运行最多构建的子图数量（达到后保存并退出，用于分批构建避免崩溃）')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test'],
                       help='只构建指定的数据集（train/val/test），如果不指定则构建所有数据集')
    
    args = parser.parse_args()
    
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    
    # ⚠️ 重要：限制最大进程数，避免内存溢出
    # 每个进程会加载完整的KG副本（约8.7GB）
    # 实际内存占用 = 进程数 × 8.7GB
    MAX_SAFE_WORKERS = 5
    if args.num_workers > MAX_SAFE_WORKERS:
        print(f"警告: 请求的进程数({args.num_workers})超过安全限制({MAX_SAFE_WORKERS})，已限制为{MAX_SAFE_WORKERS}")
        sys.stdout.flush()
        args.num_workers = MAX_SAFE_WORKERS
    
    kg_file = Path(args.kg_file)
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("预构建子图工具（优化版）")
    print("="*80)
    print(f"知识图谱: {kg_file}")
    print(f"输出目录: {output_dir}")
    print(f"参数: num_hops={args.num_hops}, max_neighbors={args.max_neighbors}")
    print(f"优化: KG共享内存 + 预计算索引 + 增量保存")
    sys.stdout.flush()
    
    # 加载知识图谱（用于获取窗口和标签）
    data_loader = GlobalKGDataLoader(kg_file)
    data_loader.load_kg()
    
    # 获取所有窗口
    print("\n获取所有窗口...")
    sys.stdout.flush()
    all_windows = list(data_loader.window_idx_map.keys())
    print(f"  总窗口数: {len(all_windows)}")
    sys.stdout.flush()
    
    if args.max_windows is not None:
        all_windows = all_windows[:args.max_windows]
        print(f"  限制窗口数: {len(all_windows)}")
        sys.stdout.flush()
    
    # 获取标签
    print("获取窗口标签...")
    sys.stdout.flush()
    all_labels = [data_loader.get_window_label(wid) for wid in all_windows]
    
    # 划分数据集
    try:
        from sklearn.model_selection import train_test_split
        from collections import Counter
    except ImportError:
        print("错误: 需要安装scikit-learn")
        sys.exit(1)
    
    label_counts = Counter(all_labels)
    min_class_count = min(label_counts.values())
    use_stratify = min_class_count >= 2
    
    train_windows, temp_windows, train_labels, temp_labels = train_test_split(
        all_windows, all_labels, test_size=1-args.train_ratio, random_state=42, 
        stratify=all_labels if use_stratify else None
    )
    
    val_ratio_adjusted = args.val_ratio / (1 - args.train_ratio)
    temp_label_counts = Counter(temp_labels)
    min_temp_class_count = min(temp_label_counts.values())
    use_stratify_temp = min_temp_class_count >= 2
    
    val_windows, test_windows, val_labels, test_labels = train_test_split(
        temp_windows, temp_labels, test_size=1-val_ratio_adjusted, random_state=42, 
        stratify=temp_labels if use_stratify_temp else None
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_windows)}")
    print(f"  验证集: {len(val_windows)}")
    print(f"  测试集: {len(test_windows)}")
    sys.stdout.flush()
    
    # 计算允许扩展的窗口集合
    train_allowed_windows = set(train_windows)
    val_allowed_windows = set(train_windows) | set(val_windows)
    test_allowed_windows = set(train_windows) | set(val_windows) | set(test_windows)
    
    # 根据--split参数决定构建哪些数据集
    splits_to_build = []
    if args.split:
        # 只构建指定的数据集
        splits_to_build = [args.split]
        print(f"\n只构建数据集: {args.split}")
    else:
        # 构建所有数据集（默认行为）
        splits_to_build = ['train', 'val', 'test']
        print(f"\n构建所有数据集: train, val, test")
    sys.stdout.flush()
    
    # 构建各数据集的子图（使用优化版）
    if 'train' in splits_to_build:
        build_and_save_subgraphs_optimized(
            kg_file, train_windows, train_labels, output_dir,
            args.num_hops, args.max_neighbors, train_allowed_windows, "train",
            args.num_workers, args.chunk_size, args.max_subgraphs_per_run
        )
    
    if 'val' in splits_to_build:
        build_and_save_subgraphs_optimized(
            kg_file, val_windows, val_labels, output_dir,
            args.num_hops, args.max_neighbors, val_allowed_windows, "val",
            args.num_workers, args.chunk_size, args.max_subgraphs_per_run
        )
    
    if 'test' in splits_to_build:
        build_and_save_subgraphs_optimized(
            kg_file, test_windows, test_labels, output_dir,
            args.num_hops, args.max_neighbors, test_allowed_windows, "test",
            args.num_workers, args.chunk_size, args.max_subgraphs_per_run
        )
    
    print("\n" + "="*80)
    print("所有子图构建完成！")
    print("="*80)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
