#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3 子图构建（同类邻居版）：每个目标窗口的邻居必须与目标同类。
绕过原有"全局 allowed_windows"机制，直接对每个 target 计算 per-window allowed set。

使用前提：lightweight_index_v3.pkl 已构建（top-k-candidates=50）。
若同类邻居不足 max_neighbors-1 个，取剩余的（不补充其他类）。

⚠ 注意：val/test 也用 GT label 选邻居 = 标签泄露。
   这适合做"oracle 上界"训练对照实验，**不能直接报告作 benchmark**。
   后续要做无泄漏 val/test 需用 entity-similarity (原版) 推理。

usage:
  python build_v3_subgraphs_sameclass.py
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Set, Tuple, Any, Dict, List

import torch

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# _lib_subgraphs* are helper libs renamed without numeric prefixes for clean import
from _lib_subgraphs import (
    build_single_subgraph_lightweight,
    _init_worker, ANOMALY_TYPE_TO_LABEL,
)
import _lib_subgraphs as _bsl
from _lib_subgraphs_optimized import IncrementalSaver
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED


# 用于 worker 端按需重建 same-class pool（避免 args 携带巨型 set 导致主进程内存爆炸）
_worker_label_to_wids: Dict[int, frozenset] = {}
_worker_allowed_pool: frozenset = frozenset()


def _init_worker_with_pools(index_path: str, allowed_pool_serialized: bytes,
                              label_to_wids_serialized: bytes,
                              log_to_count_serialized: Optional[bytes] = None):
    """Worker 初始化：加载 index + 反序列化 allowed_pool 和 label_to_wids。
    这些数据只传一次到每个 worker，不再随每个 task 传。

    log_to_count_serialized: 可选 dict {log_id: count}，注入到 _bsl._index['log_to_count']，
    供 build_single_subgraph_lightweight 在 hetero_data.log_counts 写入。
    """
    import pickle as _pickle
    _init_worker(index_path)
    global _worker_label_to_wids, _worker_allowed_pool
    _worker_allowed_pool = _pickle.loads(allowed_pool_serialized)
    _worker_label_to_wids = _pickle.loads(label_to_wids_serialized)
    if log_to_count_serialized is not None:
        _bsl._index['log_to_count'] = _pickle.loads(log_to_count_serialized)


def _build_subgraph_compact(args_tuple):
    """轻量 args：(wid, num_hops, max_neighbors, label, use_sameclass)
    Worker 内部按需构建 allowed_windows。"""
    wid, num_hops, max_neighbors, label, use_sameclass = args_tuple
    if use_sameclass:
        allowed = _worker_label_to_wids.get(label, frozenset()) & _worker_allowed_pool
    else:
        allowed = _worker_allowed_pool
    allowed = allowed - {wid}
    return build_single_subgraph_lightweight((wid, num_hops, max_neighbors, allowed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path,
                    default=Path('./datasets/<dataset>'))
    ap.add_argument('--index-file', type=Path,
                    default=Path('./datasets/<dataset>/lightweight_index_v3.pkl'))
    ap.add_argument('--output-dir', type=Path,
                    default=Path('./datasets/<dataset>/prebuilt_subgraphs_k5_sameclass'))
    ap.add_argument('--max-neighbors', type=int, default=5)
    ap.add_argument('--num-hops', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--chunk-size', type=int, default=1000)
    ap.add_argument('--sameclass-splits', nargs='+', default=['train'],
                    help='哪些 split 用同类邻居（默认仅 train；val/test 用 entity-similarity，避免标签泄露）')
    ap.add_argument('--resume', action='store_true', default=True,
                    help='断点续传：跳过 chunks_*/chunk_*.pt 中已有的 window_id')
    args = ap.parse_args()

    print('=' * 80)
    print('v3 同类邻居子图构建（HGT label-aware）')
    print('=' * 80)
    print(f'index: {args.index_file}')
    print(f'output: {args.output_dir}')
    print(f'max_neighbors={args.max_neighbors}, workers={args.num_workers}')

    # 加载索引看 label 信息
    print('\n加载索引...', flush=True)
    with open(args.index_file, 'rb') as f:
        index = pickle.load(f)
    w2l = index['window_to_label']

    # 读三个 split 的 wid（与 build_v3_subgraphs.py 一致：仅异常类）
    # 同时构建 log_to_count: 每条 LogInstance 在压缩 JSON 里的 count（去重前的实际频次）
    # 这个 count 会写到 hetero_data.log_counts，供模型做 count-aware LT 展开
    splits = {}
    log_to_count: Dict[str, int] = {}
    for s in ['train', 'val', 'test']:
        fp = args.input_dir / f'windows_anomaly_{s}_compressed.json'
        with open(fp) as f:
            d = json.load(f)
        anom_wids = []
        for raw_wid, w in d.items():
            anom = (w.get('anomaly_types') or ['normal'])[0]
            lab = ANOMALY_TYPE_TO_LABEL.get(anom, 6)
            if lab in {0, 1, 2, 3, 4, 5}:
                wid = raw_wid if raw_wid.startswith('Window:') else f'Window:{raw_wid}'
                anom_wids.append((wid, lab))
            # 给所有窗口（含 normal）的 log 都建 count（保险：邻居池里可能含 normal）
            wkey = raw_wid[len('Window:'):] if raw_wid.startswith('Window:') else raw_wid
            for log_idx, log_entry in enumerate(w.get('logs') or []):
                lid = f'LogInstance:{wkey}_{log_idx}'
                log_to_count[lid] = int(log_entry.get('count', 1)) or 1
        # 关键修复：shuffle 窗口顺序，避免 chunks 按类分桶（影响 dataloader batch 类别多样性）
        import random as _random
        _random.seed(42)
        _random.shuffle(anom_wids)
        splits[s] = anom_wids
        print(f'  {s}: {len(anom_wids)} 个异常窗口（已 shuffle）')
    print(f'  log_to_count: {len(log_to_count)} 条聚合日志, '
          f'count 范围 [{min(log_to_count.values()) if log_to_count else 0}, '
          f'{max(log_to_count.values()) if log_to_count else 0}], '
          f'avg={sum(log_to_count.values())/max(len(log_to_count),1):.1f}')

    # 全局：每个 label 对应的全部窗口 ID（在索引内的）
    label_to_wids: Dict[int, Set[str]] = defaultdict(set)
    for wid, lab in w2l.items():
        if lab in {0, 1, 2, 3, 4, 5}:
            label_to_wids[lab].add(wid)
    print('\n全集中各 label 窗口数:')
    for lab in sorted(label_to_wids):
        print(f'  label={lab}: {len(label_to_wids[lab])}')

    # leak prevention：train 仅用 train 集；val 用 train+val；test 用 train+val+test
    train_set = {wid for wid, _ in splits['train']}
    val_set = {wid for wid, _ in splits['val']}
    test_set = {wid for wid, _ in splits['test']}
    splits_allowed_pool = {
        'train': train_set,
        'val': train_set | val_set,
        'test': train_set | val_set | test_set,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 对每个 split 构建
    for split_name in ['train', 'val', 'test']:
        wids_labels = splits[split_name]
        pool = splits_allowed_pool[split_name]
        use_sameclass = split_name in args.sameclass_splits
        strategy = '同类邻居（label-aware）' if use_sameclass else 'entity-similarity（无标签泄露）'
        print(f'\n{"=" * 70}\n构建 {split_name} 子图（{strategy}）\n{"=" * 70}')
        print(f'  目标窗口: {len(wids_labels)}, 邻居池(基于 split): {len(pool)}')

        # 断点续传：扫描 chunks_{split} 已有的 window_id
        existing_wids = set()
        if args.resume:
            chunk_dir = args.output_dir / f'chunks_{split_name}'
            if chunk_dir.exists():
                import torch as _torch
                for cf in sorted(chunk_dir.glob('chunk_*.pt')):
                    try:
                        d = _torch.load(str(cf), map_location='cpu', weights_only=False)
                        if isinstance(d, dict):
                            existing_wids.update(d.keys())
                    except Exception as e:
                        print(f'    ⚠ 加载 {cf.name} 失败: {e}')
                if existing_wids:
                    print(f'  [resume] 已有 {len(existing_wids)} 个子图，跳过')

        # 紧凑 args_tuple：仅含 (wid, num_hops, max_neighbors, label, use_sameclass)
        # Worker 内部按需构建 allowed_windows，避免主进程内存爆炸
        compact_args = []
        per_class_avail = Counter()
        for wid, lab in wids_labels:
            if wid in existing_wids:
                continue
            per_class_avail[lab] += 1
            compact_args.append((wid, args.num_hops, args.max_neighbors, lab, use_sameclass))

        if not compact_args:
            print(f'  所有窗口已构建完成，跳过 {split_name}')
            continue
        print(f'  本次需构建: {len(compact_args)} 窗口  per-class: {dict(per_class_avail)}')

        # 序列化 allowed_pool 和 label_to_wids 给 worker（只传一次）
        import pickle as _pickle
        pool_bytes = _pickle.dumps(frozenset(pool), protocol=_pickle.HIGHEST_PROTOCOL)
        ltw_bytes = _pickle.dumps({k: frozenset(v) for k, v in label_to_wids.items()},
                                    protocol=_pickle.HIGHEST_PROTOCOL)
        ltc_bytes = _pickle.dumps(log_to_count, protocol=_pickle.HIGHEST_PROTOCOL)
        print(f'  worker 初始化数据: pool={len(pool_bytes)/1024/1024:.1f}MB, '
              f'label_to_wids={len(ltw_bytes)/1024/1024:.1f}MB, '
              f'log_to_count={len(ltc_bytes)/1024/1024:.1f}MB')

        # 并行构建（chunked submission 避免一次性把所有 future 入队）
        saver = IncrementalSaver(args.output_dir, split_name, chunk_size=args.chunk_size)
        completed = failed = 0
        t0 = time.time()
        BATCH = 500   # in-flight 上限
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker_with_pools,
            initargs=[str(args.index_file), pool_bytes, ltw_bytes, ltc_bytes],
        ) as ex:
            idx = 0
            in_flight = set()
            total = len(compact_args)
            while idx < total or in_flight:
                # 补满 in-flight
                while idx < total and len(in_flight) < BATCH:
                    fu = ex.submit(_build_subgraph_compact, compact_args[idx])
                    in_flight.add(fu)
                    idx += 1
                # 等任意一个完成
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED, timeout=180)
                if not done:
                    print(f'  ⚠ wait 超时无完成（已 submit {idx}, in_flight {len(in_flight)}）', flush=True)
                    continue
                for fu in done:
                    try:
                        wid, sg, label, _ = fu.result(timeout=60)
                        completed += 1
                        if sg is not None:
                            saver.add_subgraph(wid, sg)
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        completed += 1
                        if failed <= 5:
                            print(f'  ⚠ 失败 [{failed}]: {type(e).__name__}: {e}')
                    if completed % 500 == 0 or completed == total:
                        el = time.time() - t0
                        rate = completed / el if el > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(f'  {completed}/{total} ({100*completed/total:.1f}%) '
                              f'failed={failed} elapsed={el:.0f}s ETA={eta:.0f}s', flush=True)

        # 保存 metadata
        metadata = {
            'num_hops': args.num_hops,
            'max_neighbors': args.max_neighbors,
            'num_windows': len(wids_labels),
            'num_success': completed - failed,
            'window_to_label': {wid: lab for wid, lab in wids_labels},
            'sameclass_neighbors': True,
        }
        saver.finalize(metadata, merge_chunks=False)
        print(f'  完成: 成功 {completed - failed} / 失败 {failed}, 用时 {time.time() - t0:.1f}s', flush=True)

    print('\n=== ALL DONE ===')


if __name__ == '__main__':
    main()
