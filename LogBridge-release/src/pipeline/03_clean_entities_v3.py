#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 -> v3 后处理：在 v2 基础上做两件事
  (1) 基于 train 集 cov 过滤：cov<2 的实体直接从所有 split 中 drop（这些实体在 KG 里只是孤岛节点）
  (2) Thread 字段替换为归一化 ThreadGroup（丢掉 97% 单窗口细粒度 Thread）

输入：/.../compressed_v2/windows_anomaly_{train,val,test}_compressed.json
输出：/.../compressed_v3/windows_anomaly_{train,val,test}_compressed.json
       + compression_stats.json + dropped_entities.json
"""
from __future__ import annotations
import argparse, json, sys, time, re
from collections import Counter, defaultdict
from pathlib import Path

# 复用 v2 的 Thread 归一化函数
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '02_log_parsing' / 'entity_extraction'))
from clean_entity_extractor import normalize_thread

# 实体打包成 set，方便比对
ENTITY_KEYS = ['data_regions', 'nodes', 'consensus_groups', 'statistical_entities']


def collect_train_coverage(train_file: Path) -> dict:
    """对 train 做一遍流式扫描，统计 (type, value) -> 窗口数"""
    print(f'[Pass A] computing train coverage...', flush=True)
    t0 = time.time()
    with open(train_file) as f:
        data = json.load(f)
    cov = defaultdict(set)  # (type, val) -> set(window_id)
    cov_thread_group = defaultdict(set)  # ThreadGroup -> set(window_id)

    for wid, w in data.items():
        for r in w['logs']:
            ent = r['entities']
            for v in ent.get('data_regions', []):
                cov[('data_regions', v)].add(wid)
            for v in ent.get('nodes', []):
                cov[('nodes', v)].add(wid)
            for v in ent.get('consensus_groups', []):
                cov[('consensus_groups', v)].add(wid)
            for v in ent.get('statistical_entities', []):
                cov[('statistical_entities', v)].add(wid)
            # Thread 替换为 ThreadGroup 后再算 cov
            if ent.get('thread'):
                tg = normalize_thread(ent['thread'])
                if tg:
                    cov_thread_group[tg].add(wid)
    counts = {k: len(v) for k, v in cov.items()}
    tg_counts = {k: len(v) for k, v in cov_thread_group.items()}
    print(f'  train coverage computed in {time.time()-t0:.1f}s; '
          f'{len(counts):,} stat/struct entities, {len(tg_counts):,} thread groups')
    return counts, tg_counts, len(data)


def filter_split(input_file: Path, output_file: Path, cov: dict, tg_cov: dict,
                 min_cov: int = 2, drop_fine_thread: bool = True) -> dict:
    """重写一个 split：丢 cov<min_cov 的实体 + thread 字段换成 ThreadGroup"""
    print(f'\n[Pass B] {input_file.name} -> {output_file.name}', flush=True)
    t0 = time.time()
    with open(input_file) as f:
        data = json.load(f)

    dropped_count = Counter()
    kept_count = Counter()
    threads_replaced = 0
    threads_dropped = 0
    n_records = 0
    n_records_after_kept = 0
    record_with_zero_entities = 0

    for wid, w in data.items():
        for r in w['logs']:
            n_records += 1
            ent = r['entities']
            new_ent = {}

            # 各类结构化字段：仅保留 cov >= min_cov
            for key in ['data_regions', 'nodes', 'consensus_groups', 'statistical_entities']:
                vals = ent.get(key, [])
                kept = []
                for v in vals:
                    c = cov.get((key, v), 0)
                    if c >= min_cov:
                        kept.append(v)
                        kept_count[key] += 1
                    else:
                        dropped_count[key] += 1
                new_ent[key] = kept
            # node_ips: 保留 nodes 里仍存在的
            kept_ips = {n: ip for n, ip in (ent.get('node_ips') or {}).items()
                        if n in new_ent['nodes']}
            new_ent['node_ips'] = kept_ips

            # Thread 字段 -> ThreadGroup
            tg = normalize_thread(ent.get('thread') or '')
            if drop_fine_thread:
                if tg and tg_cov.get(tg, 0) >= min_cov:
                    new_ent['thread'] = f'ThreadGroup:{tg}'
                    threads_replaced += 1
                else:
                    new_ent['thread'] = None
                    threads_dropped += 1
            else:
                new_ent['thread'] = ent.get('thread')

            # 兜底字段
            new_ent['extra_threads'] = []  # 不再需要

            r['entities'] = new_ent
            # 检查这个记录是否已经空了
            total_ent = (len(new_ent['data_regions']) + len(new_ent['nodes'])
                        + len(new_ent['consensus_groups'])
                        + len(new_ent['statistical_entities'])
                        + (1 if new_ent['thread'] else 0))
            if total_ent == 0:
                record_with_zero_entities += 1
            else:
                n_records_after_kept += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    stats = {
        'split': input_file.stem.replace('windows_anomaly_', '').replace('_compressed', ''),
        'output_size_mb': output_file.stat().st_size / 1024 / 1024,
        'records_total': n_records,
        'records_with_at_least_one_entity': n_records_after_kept,
        'records_with_zero_entities': record_with_zero_entities,
        'threads_replaced_with_group': threads_replaced,
        'threads_dropped': threads_dropped,
        'entity_kept': dict(kept_count),
        'entity_dropped': dict(dropped_count),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    print(f'  records: {n_records:,} ({n_records_after_kept:,} 至少 1 个实体, {record_with_zero_entities:,} 全空)')
    print(f'  thread: replaced->ThreadGroup={threads_replaced:,}  dropped={threads_dropped:,}')
    print(f'  entity kept:    {dict(kept_count)}')
    print(f'  entity dropped: {dict(dropped_count)}')
    print(f'  saved -> {output_file} ({stats["output_size_mb"]:.1f} MB) in {stats["elapsed_sec"]}s')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path,
                    default=Path('./datasets/<dataset>_v2'))
    ap.add_argument('--output-dir', type=Path,
                    default=Path('./datasets/<dataset>'))
    ap.add_argument('--min-cov', type=int, default=2,
                    help='train 中实体最少出现窗口数才保留（默认 2 = 至少跨 2 个窗口）')
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass A: 用 train 算 cov ----
    train_in = args.input_dir / 'windows_anomaly_train_compressed.json'
    cov, tg_cov, n_train_windows = collect_train_coverage(train_in)

    # 持久化 cov 表 + 被 drop 的实体清单
    cov_file = args.output_dir / 'train_entity_coverage.json'
    cov_export = sorted(((k[0], k[1], v) for k, v in cov.items()), key=lambda x: -x[2])[:20000]
    with open(cov_file, 'w') as f:
        json.dump([{'type': t, 'value': v, 'cov': c} for t, v, c in cov_export], f, ensure_ascii=False)
    print(f'  saved top-20000 entity coverage -> {cov_file}', flush=True)

    # ---- Pass B: 重写每个 split ----
    all_stats = []
    for split in ['train', 'val', 'test']:
        in_f = args.input_dir / f'windows_anomaly_{split}_compressed.json'
        out_f = args.output_dir / f'windows_anomaly_{split}_compressed.json'
        st = filter_split(in_f, out_f, cov, tg_cov, min_cov=args.min_cov)
        all_stats.append(st)

    # ---- 持久化 stats ----
    summary = {
        'config': {
            'min_cov': args.min_cov,
            'input_dir': str(args.input_dir),
            'output_dir': str(args.output_dir),
        },
        'n_train_windows': n_train_windows,
        'distinct_entities_in_train': len(cov),
        'distinct_thread_groups_in_train': len(tg_cov),
        'splits': all_stats,
    }
    summary_file = args.output_dir / 'compression_stats.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n=== ALL DONE ===')
    print(f'summary -> {summary_file}')

if __name__ == '__main__':
    main()
