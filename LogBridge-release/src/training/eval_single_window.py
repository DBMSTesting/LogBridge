#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 C 验证：用当前最佳模型（epoch 12, val_acc 0.74），把 test 子图改造成单窗口模式（只保留目标窗口的 log + 连接的 entity），看 macro F1 是涨还是跌。

对比基线：当前 val 多窗口 macro F1 = 0.75
"""
import sys
import torch
import torch.multiprocessing as _tmp
_tmp.set_sharing_strategy('file_system')

import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from collections import Counter

SRC_ROOT = Path(__file__).resolve().parent.parent  # log_anomaly_diagnosis/src
for sub in ('model', 'utils', 'pipeline'):
    sys.path.insert(0, str(SRC_ROOT / sub))

from encoder import TemplateEncoder, EntityEncoder
from hgt_anomaly_model import HGTAnomalyModel
from global_kg_loader import GlobalKGDataLoader
from torch_geometric.data import HeteroData

# 路径全部走 CLI 参数，不再硬编码。下面只是 fallback 默认值（iotbench）。
DEFAULT_DATA_ROOT = Path('./datasets/iotbench')

LABEL_NAMES = ['compaction', 'export', 'flush', 'full_cpu', 'full_memory', 'network_bandwidth2']
NUM_CLASSES = 6
DEVICE = 'cuda'


def subgraph_to_single_window(sg):
    """把多窗口子图转为单窗口（只保留目标窗口的 log 节点 + 它们关联的 entity）"""
    target_idx = sg.target_window_idx
    target_wid = sg.target_window_id

    # 1. 找到目标窗口的 log 索引（在原 log_ids 里的位置）
    wl_edges = sg['window', 'CONTAINS', 'log'].edge_index
    target_log_local_idx = []  # original positions
    for i in range(wl_edges.size(1)):
        if wl_edges[0, i].item() == target_idx:
            target_log_local_idx.append(wl_edges[1, i].item())

    if not target_log_local_idx:
        return None

    # 2. 收集这些 log 关联的 entity
    le_edges = sg['log', 'ASSOCIATED_WITH', 'entity'].edge_index if ('log', 'ASSOCIATED_WITH', 'entity') in sg.edge_types else None
    target_log_set = set(target_log_local_idx)
    target_entity_local_idx = set()
    if le_edges is not None:
        for i in range(le_edges.size(1)):
            li = le_edges[0, i].item()
            ei = le_edges[1, i].item()
            if li in target_log_set:
                target_entity_local_idx.add(ei)
    target_entity_local_idx = sorted(target_entity_local_idx)

    # 3. 建立旧→新的 index 映射
    log_new_idx = {old: new for new, old in enumerate(target_log_local_idx)}
    entity_new_idx = {old: new for new, old in enumerate(target_entity_local_idx)}

    # 4. 重建新子图
    new_sg = HeteroData()
    new_sg['window'].num_nodes = 1
    new_sg['log'].num_nodes = len(target_log_local_idx)
    new_sg['entity'].num_nodes = len(target_entity_local_idx)

    # window -> log edges (1 window, all target logs)
    if target_log_local_idx:
        wl_new = torch.tensor([[0]*len(target_log_local_idx), list(range(len(target_log_local_idx)))], dtype=torch.long)
        new_sg['window', 'CONTAINS', 'log'].edge_index = wl_new

    # log -> entity edges (filter and remap)
    if le_edges is not None and target_entity_local_idx:
        new_src, new_dst = [], []
        for i in range(le_edges.size(1)):
            li = le_edges[0, i].item()
            ei = le_edges[1, i].item()
            if li in log_new_idx and ei in entity_new_idx:
                new_src.append(log_new_idx[li])
                new_dst.append(entity_new_idx[ei])
        if new_src:
            new_sg['log', 'ASSOCIATED_WITH', 'entity'].edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
            new_sg['entity', 'REVERSE_ASSOCIATED_WITH', 'log'].edge_index = torch.tensor([new_dst, new_src], dtype=torch.long)

    # 5. 复制元数据
    new_sg.target_window_id = target_wid
    new_sg.target_window_idx = 0  # 现在只有 1 个 window
    new_sg.window_ids = [target_wid]
    new_sg.log_ids = [sg.log_ids[i] for i in target_log_local_idx]
    new_sg.entity_ids = [sg.entity_ids[i] for i in target_entity_local_idx]
    new_sg.label = sg.label

    # 复制 log_template_texts 和 entity_contents
    if hasattr(sg, 'log_template_texts'):
        orig_ltt = sg.log_template_texts
        new_sg.log_template_texts = {lid: orig_ltt[lid] for lid in new_sg.log_ids if lid in orig_ltt}
    if hasattr(sg, 'entity_contents'):
        orig_ec = sg.entity_contents
        new_sg.entity_contents = {eid: orig_ec[eid] for eid in new_sg.entity_ids if eid in orig_ec}

    return new_sg


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Multi-window vs single-window inference comparison')
    ap.add_argument('--data-root', type=Path, required=True,
                    help='dataset root, expects kg_*.json, bert_embeddings_v3.pt, checkpoints_sameclass/best_model.pt, prebuilt_subgraphs_k5_sameclass/chunks_test/')
    ap.add_argument('--kg-file', type=Path, default=None,
                    help='override KG file path (default: <data-root>/kg_*_v3.json)')
    ap.add_argument('--ckpt', type=Path, default=None,
                    help='override checkpoint path (default: <data-root>/checkpoints_sameclass/best_model.pt)')
    ap.add_argument('--bert-emb', type=Path, default=None,
                    help='override BERT embedding file (default: <data-root>/bert_embeddings_v3.pt)')
    ap.add_argument('--chunks-dir', type=Path, default=None,
                    help='override test chunks dir (default: <data-root>/prebuilt_subgraphs_k5_sameclass/chunks_test/)')
    args = ap.parse_args()

    data_root = args.data_root
    KG_FILE = args.kg_file or next(data_root.glob('kg_*_v3.json'))
    BERT_FILE = args.bert_emb or (data_root / 'bert_embeddings_v3.pt')
    CKPT_FILE = args.ckpt or (data_root / 'checkpoints_sameclass' / 'best_model.pt')
    VAL_CHUNKS_DIR = args.chunks_dir or (data_root / 'prebuilt_subgraphs_k5_sameclass' / 'chunks_test')

    print('=' * 70)
    print(f'多窗口 vs 单窗口推理对比 ({data_root.name})')
    print('=' * 70)
    print(f'  KG:    {KG_FILE}')
    print(f'  BERT:  {BERT_FILE}')
    print(f'  CKPT:  {CKPT_FILE}')
    print(f'  CHUNKS:{VAL_CHUNKS_DIR}')

    # 1. 加载 KG（必要：构建 entity_vocab）
    print(f'\n[1] 加载 KG ...', flush=True)
    kg_loader = GlobalKGDataLoader(KG_FILE)
    kg_loader.load_kg()

    # 2. 加载 entity_vocab —— 直接从 checkpoint 取，避免 set 迭代顺序非确定性导致
    #    embedding ↔ entity 错位的 bug（这是之前 v1 单窗口报告"export F1=0"的根因）
    print(f'\n[2] 从 checkpoint 加载 entity_vocab ...', flush=True)
    _ckpt_for_vocab = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
    entity_vocab = _ckpt_for_vocab.get('entity_vocab')
    if not entity_vocab:
        raise RuntimeError('checkpoint 缺少 entity_vocab——无法保证 entity embedding 对齐')
    print(f'  entity_vocab size: {len(entity_vocab)}（从 ckpt 加载）')

    # 3. 加载 BERT 预编码
    print(f'\n[3] 加载 BERT 预编码 ...', flush=True)
    bert_raw = torch.load(BERT_FILE, map_location='cpu')
    if isinstance(bert_raw, dict) and 'embeddings' in bert_raw:
        precomputed_embs = bert_raw['embeddings']
    else:
        precomputed_embs = bert_raw
    print(f'  {len(precomputed_embs)} 个模板')

    # 4. 初始化模型
    print(f'\n[4] 初始化模型 + 加载 checkpoint ...', flush=True)
    template_encoder = TemplateEncoder(
        model_name='bert-base-uncased',
        device=DEVICE,
        precomputed_embeddings=precomputed_embs,
    )
    entity_encoder = EntityEncoder(
        entity_vocab=entity_vocab,
        embedding_dim=128,
        use_bert=False,
        device=DEVICE,
    )
    model = HGTAnomalyModel(
        template_encoder=template_encoder,
        entity_encoder=entity_encoder,
        log_embedding_dim=template_encoder.embedding_dim,
        entity_embedding_dim=128,
        local_transformer_dim=256,
        hgt_hidden_dim=256,
        hgt_num_heads=8,
        hgt_num_layers=2,
        num_classes=NUM_CLASSES,
        device=DEVICE,
    )
    ckpt = torch.load(CKPT_FILE, map_location=DEVICE)
    sd = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  missing keys: {len(missing)}  unexpected: {len(unexpected)}')
    model.eval()

    # 5. 加载 test chunks
    print(f'\n[5] 加载 test chunks ...', flush=True)
    chunks = sorted(VAL_CHUNKS_DIR.glob('chunk_*.pt'))
    print(f'  {len(chunks)} chunks')

    all_preds_multi = []
    all_preds_single = []
    all_labels = []
    multi_win_counts = Counter()
    n_isolated = 0
    n_processed = 0

    with torch.no_grad():
        for ci, cf in enumerate(chunks):
            chunk_data = torch.load(str(cf), map_location='cpu', weights_only=False)
            for wid, sg in chunk_data.items():
                label = sg.label
                if label < 0 or label >= NUM_CLASSES:
                    continue

                # 多窗口推理（原版）
                sg_dev = sg.to(DEVICE)
                logits_multi = model(sg_dev, target_window_idx=sg.target_window_idx)
                pred_multi = logits_multi.argmax(dim=0).item()

                # 单窗口推理
                sg_single = subgraph_to_single_window(sg)
                multi_win_counts[len(sg.window_ids)] += 1
                if sg_single is None:
                    n_isolated += 1
                    pred_single = pred_multi  # fallback
                else:
                    sg_single_dev = sg_single.to(DEVICE)
                    logits_single = model(sg_single_dev, target_window_idx=0)
                    pred_single = logits_single.argmax(dim=0).item()

                all_preds_multi.append(pred_multi)
                all_preds_single.append(pred_single)
                all_labels.append(label)
                n_processed += 1

            print(f'  chunk {ci+1}/{len(chunks)}  processed={n_processed}', flush=True)

    print(f'\n  原子图 window 数分布: {dict(multi_win_counts)}')
    print(f'  无法 single-windowize 的子图: {n_isolated}')

    # 6. 对比
    def report(preds, name):
        acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
        prec, rec, f1, sup = precision_recall_fscore_support(
            all_labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0,
        )
        macro_f1 = float(np.mean(f1))
        print(f'\n[{name}]  Acc={acc:.4f}  Macro F1={macro_f1:.4f}')
        for i, n in enumerate(LABEL_NAMES):
            print(f'    {n:25s}: P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f}  (n={int(sup[i])})')
        return acc, macro_f1

    print('\n' + '=' * 70)
    print('对比报告')
    print('=' * 70)
    acc_m, mf1_m = report(all_preds_multi, '多窗口 (current)')
    acc_s, mf1_s = report(all_preds_single, '单窗口 (proposed)')

    print(f'\n[差异] 多窗口 vs 单窗口')
    print(f'  Acc:        {acc_m:.4f} -> {acc_s:.4f}  ({(acc_s-acc_m)*100:+.2f} pp)')
    print(f'  Macro F1:   {mf1_m:.4f} -> {mf1_s:.4f}  ({(mf1_s-mf1_m)*100:+.2f} pp)')

    # 一致性：两种推理预测一致的比例
    agree = sum(p1 == p2 for p1, p2 in zip(all_preds_multi, all_preds_single))
    print(f'\n两种推理预测一致率: {agree}/{len(all_preds_multi)} = {100*agree/len(all_preds_multi):.1f}%')


if __name__ == '__main__':
    main()
