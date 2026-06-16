#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预计算BERT编码
遍历知识图谱中的所有日志模板，预先计算BERT编码并保存
这样可以避免训练时重复编码，大幅提升训练速度
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Set
import torch
from tqdm import tqdm

# 新目录结构
SRC_ROOT = Path(__file__).resolve().parent.parent  # log_anomaly_diagnosis/src
sys.path.insert(0, str(SRC_ROOT / "model"))
from encoder import TemplateEncoder


def collect_all_templates(kg_file: Path) -> Set[str]:
    """
    收集知识图谱中所有唯一的模板文本
    
    Args:
        kg_file: 知识图谱JSON文件路径
        
    Returns:
        Set[str]: 所有唯一的模板文本集合
    """
    print(f"加载知识图谱: {kg_file}")
    with open(kg_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    templates = set()
    nodes = kg_data.get('nodes', [])
    
    print(f"  总节点数: {len(nodes)}")
    print("  收集模板文本...")
    
    for node in tqdm(nodes, desc="处理节点"):
        if node.get('type') == 'LogInstance':
            properties = node.get('properties', {})
            # 优先使用raw_line，如果没有则使用template_id
            raw_line = properties.get('raw_line', '')
            template_id = properties.get('template_id', '')
            
            if raw_line:
                templates.add(raw_line)
            elif template_id:
                templates.add(template_id)
    
    print(f"  找到 {len(templates)} 个唯一的模板文本")
    return templates


def precompute_embeddings(
    templates: Set[str],
    template_encoder: TemplateEncoder,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    批量预计算模板的BERT编码
    
    Args:
        templates: 模板文本集合
        template_encoder: 模板编码器
        batch_size: 批量大小
        device: 设备
        
    Returns:
        Dict[str, torch.Tensor]: {template_text: embedding}
    """
    print(f"\n开始预计算BERT编码...")
    print(f"  模板数量: {len(templates)}")
    print(f"  批量大小: {batch_size}")
    print(f"  设备: {device}")
    
    template_list = list(templates)
    embeddings_dict = {}
    
    # 批量编码
    for i in tqdm(range(0, len(template_list), batch_size), desc="编码进度"):
        batch_templates = template_list[i:i+batch_size]
        batch_embeddings = template_encoder.encode_batch(batch_templates)
        
        # 将编码结果存储到字典
        for template, embedding in zip(batch_templates, batch_embeddings):
            # 移动到CPU并detach，避免占用GPU内存
            embeddings_dict[template] = embedding.cpu().detach()
    
    print(f"  完成编码: {len(embeddings_dict)} 个模板")
    return embeddings_dict


def save_embeddings(
    embeddings_dict: Dict[str, torch.Tensor],
    output_file: Path,
    model_name: str = 'bert-base-uncased'
):
    """
    保存预编码的向量
    
    Args:
        embeddings_dict: {template_text: embedding}
        output_file: 输出文件路径
        model_name: BERT模型名称（用于元数据）
    """
    print(f"\n保存预编码向量到: {output_file}")
    
    # 准备保存的数据
    save_data = {
        'model_name': model_name,
        'embedding_dim': list(embeddings_dict.values())[0].shape[0] if embeddings_dict else 768,
        'num_templates': len(embeddings_dict),
        'embeddings': embeddings_dict
    }
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_file)
    
    # 计算文件大小
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    print(f"  保存完成: {len(embeddings_dict)} 个模板")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  嵌入维度: {save_data['embedding_dim']}")


def main():
    parser = argparse.ArgumentParser(description='预计算BERT编码')
    parser.add_argument('--kg-file', type=str, required=True,
                       help='知识图谱JSON文件路径')
    parser.add_argument('--output-file', type=str,
                       default='./bert_embeddings.pt',
                       help='输出文件路径（.pt文件）')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                       help='BERT模型名称')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批量编码大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    kg_file = Path(args.kg_file)
    output_file = Path(args.output_file)
    
    if not kg_file.exists():
        print(f"错误: 知识图谱文件不存在: {kg_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("预计算BERT编码")
    print("=" * 60)
    print(f"知识图谱: {kg_file}")
    print(f"输出文件: {output_file}")
    print(f"BERT模型: {args.model_name}")
    print("=" * 60)
    
    # 1. 收集所有唯一的模板文本
    templates = collect_all_templates(kg_file)
    
    if len(templates) == 0:
        print("错误: 没有找到任何模板文本")
        sys.exit(1)
    
    # 2. 初始化模板编码器
    print(f"\n初始化BERT编码器: {args.model_name}")
    template_encoder = TemplateEncoder(model_name=args.model_name, device=args.device)
    
    # 3. 预计算编码
    embeddings_dict = precompute_embeddings(
        templates,
        template_encoder,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 4. 保存
    save_embeddings(embeddings_dict, output_file, args.model_name)
    
    print("\n" + "=" * 60)
    print("✅ 预编码完成！")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"  在训练时指定预编码文件: --bert-embeddings-file {output_file}")


if __name__ == '__main__':
    main()





