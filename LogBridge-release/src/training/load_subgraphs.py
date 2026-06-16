#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载预构建子图的工具函数（支持chunks格式）
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch_geometric.data import HeteroData


# 全局chunk缓存（避免重复加载同一个chunk）
_chunk_cache = {}


def load_prebuilt_subgraphs(
    subgraph_dir: Path,
    split_name: str = "train",
    load_all: bool = False  # 是否一次性加载所有子图（False时按需加载chunks）
) -> Tuple[Dict[str, HeteroData], Dict]:
    """
    加载预构建的子图（支持chunks格式）
    
    Args:
        subgraph_dir: 子图文件所在目录
        split_name: 数据集名称（train/val/test）
        load_all: 是否一次性加载所有子图（False时返回空字典，按需从chunks加载）
        
    Returns:
        Tuple[Dict[str, HeteroData], Dict]: (子图字典, 元数据)
        注意：如果使用chunks格式且load_all=False，返回空字典，实际子图按需从chunks加载
    """
    subgraph_file = subgraph_dir / f"subgraphs_{split_name}.pt"
    
    if not subgraph_file.exists():
        raise FileNotFoundError(
            f"预构建子图文件不存在: {subgraph_file}\n"
            f"请先运行 build_subgraphs.py 构建子图"
        )
    
    print(f"加载预构建子图: {subgraph_file}")
    try:
        data = torch.load(subgraph_file, map_location='cpu')
    except Exception as e:
        print(f"  ⚠️  文件加载失败（可能已损坏）: {str(e)[:100]}")
        print(f"  将返回空字典，训练时将动态构建子图")
        return {}, {}
    
    # 检查是否是chunks格式（新格式）
    if 'chunk_files' in data:
        # Chunks格式：不合并，按需加载
        chunk_files = data.get('chunk_files', [])
        metadata = data.get('metadata', {})
        
        print(f"  检测到chunks格式: {len(chunk_files)} 个chunk文件")
        print(f"  总子图数: {data.get('num_subgraphs', '未知')}")
        if 'num_hops' in metadata and 'max_neighbors' in metadata:
            print(f"  参数: num_hops={metadata['num_hops']}, max_neighbors={metadata['max_neighbors']}")
        
        # 检查是否有已存在的子图文件（backup）
        existing_subgraphs_file = data.get('existing_subgraphs_file')
        if existing_subgraphs_file:
            existing_file_path = Path(existing_subgraphs_file)
            if not existing_file_path.is_absolute():
                # 相对路径，需要转换为绝对路径
                chunk_dir = Path(data.get('chunk_dir', ''))
                if chunk_dir:
                    existing_file_path = chunk_dir.parent / existing_file_path
                else:
                    existing_file_path = subgraph_file.parent / existing_file_path
            
            if existing_file_path.exists():
                print(f"  检测到已存在子图文件: {existing_file_path.name}")
                metadata['_existing_subgraphs_file'] = str(existing_file_path)
        
        if load_all:
            # 一次性加载所有chunks（可能内存占用大）
            print(f"  正在加载所有chunks到内存...")
            subgraphs = {}
            
            # 先加载已存在的子图（如果有）
            if existing_subgraphs_file and existing_file_path.exists():
                print(f"    加载已存在子图: {existing_file_path.name}")
                existing_data = torch.load(existing_file_path, map_location='cpu')
                if isinstance(existing_data, dict) and 'subgraphs' in existing_data:
                    subgraphs.update(existing_data['subgraphs'])
                else:
                    subgraphs.update(existing_data)
                print(f"      已加载 {len(subgraphs)} 个已存在子图")
            
            # 再加载chunks
            for i, chunk_file in enumerate(chunk_files):
                chunk_path = Path(chunk_file)
                if not chunk_path.is_absolute():
                    chunk_dir = Path(data.get('chunk_dir', ''))
                    if chunk_dir:
                        chunk_path = chunk_dir / chunk_file
                    else:
                        chunk_path = subgraph_file.parent / chunk_file
                
                print(f"    加载chunk {i+1}/{len(chunk_files)}: {chunk_path.name}")
                chunk_data = torch.load(chunk_path, map_location='cpu')
                subgraphs.update(chunk_data)
            print(f"  加载完成: {len(subgraphs)} 个子图")
            return subgraphs, metadata
        else:
            # 不加载，返回空字典和chunk信息（按需加载）
            # 将chunk信息存储在metadata中，供后续按需加载
            metadata['_chunk_files'] = chunk_files
            metadata['_chunk_dir'] = data.get('chunk_dir', '')
            print(f"  使用按需加载模式（chunks），训练时从chunks动态加载")
            if existing_subgraphs_file:
                print(f"  同时支持从backup文件加载已存在子图")
            return {}, metadata
    else:
        # 旧格式：直接包含subgraphs字典
        subgraphs = data.get('subgraphs', {})
        metadata = data.get('metadata', {})
        
        if len(subgraphs) == 0:
            print(f"  ⚠️  子图字典为空")
        else:
            print(f"  加载成功: {len(subgraphs)} 个子图")
            if 'num_hops' in metadata and 'max_neighbors' in metadata:
                print(f"  参数: num_hops={metadata['num_hops']}, max_neighbors={metadata['max_neighbors']}")
        
        return subgraphs, metadata


def load_subgraph_from_chunks(
    window_id: str,
    metadata: Dict,
    chunk_cache: Optional[Dict] = None
) -> Optional[HeteroData]:
    """
    从chunks和backup文件中按需加载单个子图
    
    Args:
        window_id: 窗口ID
        metadata: 包含chunk信息的元数据
        chunk_cache: chunk缓存字典（用于避免重复加载同一个chunk）
        
    Returns:
        Optional[HeteroData]: 子图数据，如果不存在则返回None
    """
    if chunk_cache is None:
        chunk_cache = _chunk_cache
    
    # 先检查已存在的子图文件（backup）
    existing_file = metadata.get('_existing_subgraphs_file')
    if existing_file:
        existing_file_path = Path(existing_file)
        cache_key = f"_existing_{existing_file_path}"
        
        if cache_key not in chunk_cache:
            try:
                existing_data = torch.load(existing_file_path, map_location='cpu')
                if isinstance(existing_data, dict) and 'subgraphs' in existing_data:
                    chunk_cache[cache_key] = existing_data['subgraphs']
                else:
                    chunk_cache[cache_key] = existing_data
            except Exception as e:
                print(f"  警告: 加载已存在子图文件失败 {existing_file_path.name}: {str(e)[:100]}")
                return None
        
        existing_subgraphs = chunk_cache[cache_key]
        if window_id in existing_subgraphs:
            return existing_subgraphs[window_id]
    
    # 再检查chunks
    chunk_files = metadata.get('_chunk_files', [])
    if not chunk_files:
        return None
    
    chunk_dir = Path(metadata.get('_chunk_dir', ''))
    
    # 遍历所有chunks，查找目标窗口
    for chunk_file in chunk_files:
        chunk_file_path = Path(chunk_file)
        if not chunk_file_path.is_absolute() and chunk_dir:
            chunk_file_path = chunk_dir / chunk_file
        
        # 检查chunk是否已加载到缓存
        cache_key = str(chunk_file_path)
        if cache_key not in chunk_cache:
            try:
                chunk_cache[cache_key] = torch.load(chunk_file_path, map_location='cpu')
            except Exception as e:
                print(f"  警告: 加载chunk失败 {chunk_file_path.name}: {str(e)[:100]}")
                continue
        
        # 在chunk中查找目标窗口
        chunk_data = chunk_cache[cache_key]
        if window_id in chunk_data:
            return chunk_data[window_id]
    
    return None


def check_subgraph_compatibility(
    metadata: Dict,
    num_hops: int,
    max_neighbors: int
) -> bool:
    """
    检查预构建子图的参数是否与当前训练参数兼容
    
    Args:
        metadata: 子图元数据
        num_hops: 当前训练使用的num_hops
        max_neighbors: 当前训练使用的max_neighbors
        
    Returns:
        bool: 是否兼容
    """
    if not metadata:
        return False
    if 'num_hops' not in metadata or 'max_neighbors' not in metadata:
        return False
    if metadata['num_hops'] != num_hops:
        return False
    if metadata['max_neighbors'] != max_neighbors:
        return False
    return True
