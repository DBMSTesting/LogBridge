#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPC 数据集日志压缩 + 实体抽取 一体化脚本

流程：
  Pass 1 (仅在 train 上运行)：流式扫描，同时
    - Drain3 模板挖掘（depth=4, sim_th=0.4，全局单 miner）
    - 统计实体的 token 频次（用于 RF 过滤）
    并持久化 miner 状态 + valid_tokens。

  Pass 2 (train / val / test 各跑一次)：流式扫描
    - 用已 fit 的 miner 通过 match() 得到 cluster_id（不再新增 cluster）
    - 用已冻结的 valid_tokens 做实体抽取
    - 按 (window, template_id) 分组，每组合并为一条压缩记录

输出位置：
  ./datasets/<dataset>_v1/
    pass1_state.pkl
    valid_tokens.json
    drain3_state.bin
    windows_anomaly_train_compressed.json
    windows_anomaly_val_compressed.json
    windows_anomaly_test_compressed.json
    compression_stats.json

每条压缩记录字段：
    template_id, template_text, raw_line (group 内 first_ts 对应的那条),
    raw_line_last, count, first_ts, last_ts, entities {data_regions, nodes,
    node_ips, thread, consensus_groups, statistical_entities}
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- 项目路径 ----
PROJ = Path('./external_pipeline')
sys.path.insert(0, str(PROJ / '02_log_parsing' / 'entity_extraction'))

from entity_extractor import EntityExtractor  # noqa
from enhanced_entity_extractor import EnhancedEntityExtractor  # noqa
from clean_entity_extractor import CleanEntityExtractor  # noqa

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# 提取消息体（去掉时间戳和线程）
_MSG_RE = re.compile(r"\]\s+[A-Z]+\s+(.+)$")


# ============================================================
#                      Drain3 helpers
# ============================================================

def make_drain_miner(depth: int = 4, sim_th: float = 0.4) -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.drain_depth = depth
    cfg.drain_sim_th = sim_th
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def drain_match_or_unknown(miner: TemplateMiner, line: str):
    """
    优先使用 match() 在已有 cluster 中查询；
    若未匹配则回退为 add_log_message 但**不让它增长 miner**——
    实际上 Drain3 没有提供"只读 add"的开关，所以未匹配时返回 None。
    """
    cluster = miner.match(line)
    if cluster is not None:
        return cluster.cluster_id, cluster.get_template()
    return None, None


# ============================================================
#                      Pass 1
# ============================================================

def pass1_fit(train_file: Path, miner: TemplateMiner, ext: EnhancedEntityExtractor,
              progress_every: int = 100_000):
    """
    流式扫描 train，同时：
      - Drain3 fit
      - 统计实体 token 频次（process_batch_message 不入 cache，省内存）
    """
    print(f'[Pass 1] fit Drain3 + RF on {train_file.name}', flush=True)
    ext.start_batch_mode()
    n_windows = 0
    n_lines = 0
    t0 = time.time()
    with open(train_file, 'rb') as f:
        # 直接 json.load 305MB 是可以的；用 ijson 更省内存但更慢，权衡下选 json.load
        data = json.load(f)
    for wid, w in data.items():
        n_windows += 1
        for log in w.get('logs', []):
            line = log.get('raw_line', '')
            if not line:
                continue
            # Drain3 fit
            miner.add_log_message(line)
            # RF 统计（仅消息体）
            m = _MSG_RE.search(line)
            msg = m.group(1) if m else line
            ext.process_batch_message(msg)
            n_lines += 1
            if n_lines % progress_every == 0:
                el = time.time() - t0
                print(f'  scanned {n_lines:,} lines / {n_windows:,} windows  '
                      f'elapsed={el:.1f}s  rate={n_lines/el:.0f}/s', flush=True)
    ext.finish_batch_mode()
    print(f'[Pass 1] DONE {n_windows} windows / {n_lines:,} lines in {time.time()-t0:.1f}s', flush=True)
    print(f'  drain3 clusters: {len(miner.drain.clusters)}', flush=True)
    print(f'  valid_tokens (TC>=2, RF>=5): {len(ext.statistical_miner.valid_tokens_cache)}', flush=True)
    # 释放 train 数据
    del data


# ============================================================
#                      Pass 2 (compress per split)
# ============================================================

def compress_one_split(input_file: Path, output_file: Path,
                       miner: TemplateMiner, ext: EnhancedEntityExtractor,
                       split_name: str, progress_every: int = 1000) -> dict:
    """对一个 split 做压缩，返回 stats dict。"""
    print(f'\n[Pass 2 / {split_name}] {input_file.name} -> {output_file.name}', flush=True)
    t0 = time.time()
    with open(input_file, 'rb') as f:
        data = json.load(f)
    print(f'  loaded {len(data)} windows', flush=True)

    out = {}
    # 全局统计
    total_logs_before = 0
    total_logs_after = 0
    unknown_template_lines = 0
    per_class_before = defaultdict(int)
    per_class_after = defaultdict(int)
    distinct_per_window = []

    for idx, (wid, w) in enumerate(data.items(), 1):
        logs = w.get('logs', [])
        if not logs:
            continue
        label = (w.get('anomaly_types') or ['normal'])[0]
        total_logs_before += len(logs)
        per_class_before[label] += len(logs)

        # 按 template_id 分组
        groups = defaultdict(list)  # cid -> list[(log_idx, log_dict, entities)]
        for li, log in enumerate(logs):
            line = log.get('raw_line', '')
            if not line:
                continue
            cid, template = drain_match_or_unknown(miner, line)
            if cid is None:
                cid = -1
                template = '<UNKNOWN>'
                unknown_template_lines += 1
            ent = ext.extract_from_log_line(line)
            groups[(cid, template)].append((li, log, ent))

        # 生成压缩记录（按组内 first_ts 排序，保证 raw_line 选取稳定）
        compressed_logs = []
        for (cid, template), items in groups.items():
            items.sort(key=lambda x: x[1].get('timestamp', ''))
            first = items[0][1]
            last = items[-1][1]
            # union 实体
            data_regions = set()
            nodes = set()
            node_ips = {}
            threads = set()
            consensus = set()
            stat_ents = set()
            for _, _, ent in items:
                data_regions.update(ent.data_regions)
                nodes.update(ent.nodes)
                if ent.node_ips:
                    node_ips.update(ent.node_ips)
                if ent.thread:
                    threads.add(ent.thread)
                consensus.update(ent.consensus_groups)
                if hasattr(ent, 'statistical_entities'):
                    stat_ents.update(ent.statistical_entities)
            # thread 字段历史保持单值；如果一个 group 内出现多 thread，concat 会破坏下游解析，
            # 这里取 first 的 thread，多余的 thread 名加入 statistical_entities 作为兜底
            thread_main = items[0][2].thread
            if thread_main is None and threads:
                thread_main = next(iter(threads))
            extra_threads = sorted(t for t in threads if t and t != thread_main)
            if extra_threads:
                # 不会污染 Thread 节点，仅作为兜底；下游若要用可单独取
                pass

            compressed_logs.append({
                'template_id': f'cluster_{cid}' if cid >= 0 else 'cluster_unknown',
                'template_cluster_id': cid,
                'template_text': template,
                'raw_line': first.get('raw_line', ''),       # 喂给 BERT 的代表行
                'raw_line_last': last.get('raw_line', ''),   # 给 LLM 解释生成留底
                'count': len(items),
                'first_ts': first.get('timestamp', ''),
                'last_ts': last.get('timestamp', ''),
                'entities': {
                    'data_regions': sorted(data_regions),
                    'nodes': sorted(nodes),
                    'node_ips': node_ips,
                    'thread': thread_main,
                    'consensus_groups': sorted(consensus),
                    'statistical_entities': sorted(stat_ents),
                    'extra_threads': extra_threads,  # 兜底字段，下游可忽略
                },
                # 兼容字段（旧 KG 构建脚本会读 timestamp）
                'timestamp': first.get('timestamp', ''),
            })

        # 按 first_ts 排序，保证序列稳定
        compressed_logs.sort(key=lambda r: r.get('first_ts', ''))

        out[wid] = {
            'label_file': w.get('label_file', ''),
            'window_start': w.get('window_start', ''),
            'window_end': w.get('window_end', ''),
            'log_count_original': len(logs),
            'log_count': len(compressed_logs),
            'compressed': True,
            'anomaly_types': w.get('anomaly_types', []),
            'logs': compressed_logs,
        }

        total_logs_after += len(compressed_logs)
        per_class_after[label] += len(compressed_logs)
        distinct_per_window.append(len(compressed_logs))

        if idx % progress_every == 0:
            el = time.time() - t0
            print(f'  [{split_name}] processed {idx}/{len(data)} windows '
                  f'elapsed={el:.1f}s avg_records/win={total_logs_after/idx:.1f}', flush=True)

    # 写出
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f'  saved -> {output_file} ({output_file.stat().st_size / (1024*1024):.1f} MB)', flush=True)

    # stats
    if distinct_per_window:
        distinct_per_window.sort()
        n = len(distinct_per_window)
        p50 = distinct_per_window[n // 2]
        p90 = distinct_per_window[int(n * 0.9)]
    else:
        p50 = p90 = 0
    stats = {
        'split': split_name,
        'windows': len(data),
        'logs_before': total_logs_before,
        'logs_after': total_logs_after,
        'compression_ratio_kept': (total_logs_after / total_logs_before) if total_logs_before else 0,
        'compression_ratio_saved': 1 - (total_logs_after / total_logs_before) if total_logs_before else 0,
        'avg_records_per_window': total_logs_after / max(1, len(data)),
        'p50_records_per_window': p50,
        'p90_records_per_window': p90,
        'unknown_template_lines': unknown_template_lines,
        'per_class_logs_before': dict(per_class_before),
        'per_class_logs_after': dict(per_class_after),
        'per_class_kept_ratio': {
            k: per_class_after[k] / per_class_before[k] for k in per_class_before
        },
        'elapsed_sec': round(time.time() - t0, 1),
    }
    print(f'  STATS {split_name}:', flush=True)
    print(f'    logs {total_logs_before:,} -> {total_logs_after:,}  '
          f'kept={stats["compression_ratio_kept"]*100:.1f}%  '
          f'saved={stats["compression_ratio_saved"]*100:.1f}%', flush=True)
    print(f'    avg/p50/p90 records per window: '
          f'{stats["avg_records_per_window"]:.1f}/{p50}/{p90}', flush=True)
    print(f'    unknown-template lines: {unknown_template_lines}', flush=True)
    return stats


# ============================================================
#                      main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path,
                    default=Path('./datasets/<dataset>_raw'))
    ap.add_argument('--output-dir', type=Path,
                    default=Path('./datasets/<dataset>_v1'))
    ap.add_argument('--drain-depth', type=int, default=4)
    ap.add_argument('--drain-sim-th', type=float, default=0.4)
    ap.add_argument('--rf-min', type=int, default=5)
    ap.add_argument('--tc-min', type=int, default=2)
    ap.add_argument('--smoke', action='store_true',
                    help='只跑 val 做 smoke test，不动 train/test')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--extractor', choices=['enhanced', 'clean'], default='enhanced',
                    help='enhanced=旧版 RF/TC 自动学习 (v1), clean=v2 干净抽取器')
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 各 split 文件
    files = {
        'train': args.input_dir / 'windows_anomaly_train.json',
        'val':   args.input_dir / 'windows_anomaly_val.json',
        'test':  args.input_dir / 'windows_anomaly_test.json',
    }
    for k, v in files.items():
        if not v.exists():
            print(f'❌ missing {v}', file=sys.stderr)
            sys.exit(1)

    print('=' * 80)
    print('TPC LOG COMPRESSION + ENTITY EXTRACTION')
    print('=' * 80)
    print(f'input_dir : {args.input_dir}')
    print(f'output_dir: {args.output_dir}')
    print(f'drain_depth={args.drain_depth}  drain_sim_th={args.drain_sim_th}')
    print(f'rf_min={args.rf_min}  tc_min={args.tc_min}')

    miner = make_drain_miner(args.drain_depth, args.drain_sim_th)
    if args.extractor == 'clean':
        print('  使用 v2 CleanEntityExtractor（黑白名单+归一化，无需 RF 学习）')
        ext = CleanEntityExtractor()
    else:
        print('  使用 v1 EnhancedEntityExtractor（RF/TC 自动学习）')
        ext = EnhancedEntityExtractor(enable_statistical_mining=True,
                                      tc_threshold=args.tc_min, min_rf=args.rf_min)

    # ---------- Pass 1 ----------
    if args.smoke:
        # smoke 只用 val 自身 fit，速度快
        print('\n[SMOKE MODE] fit on VAL (not full train)\n')
        pass1_fit(files['val'], miner, ext)
    else:
        pass1_fit(files['train'], miner, ext)

    # 持久化 valid_tokens
    valid_tokens_file = args.output_dir / 'valid_tokens.json'
    with open(valid_tokens_file, 'w') as f:
        json.dump(sorted(ext.statistical_miner.valid_tokens_cache), f)
    print(f'  saved valid_tokens -> {valid_tokens_file}', flush=True)

    # 持久化 token frequency （便于排查），仅 v1 enhanced 抽取器有
    tf = getattr(ext.statistical_miner, 'token_frequency', None)
    if tf and hasattr(tf, 'most_common'):
        tf_file = args.output_dir / 'token_frequency.json'
        with open(tf_file, 'w') as f:
            json.dump(dict(tf.most_common(5000)), f)
        print(f'  saved top-5000 token_frequency -> {tf_file}', flush=True)
    else:
        print('  (skip token_frequency dump — clean extractor has no RF stats)', flush=True)

    # 持久化 miner state（pickle 比 drain3 自带的更直接）
    miner_state_file = args.output_dir / 'pass1_state.pkl'
    with open(miner_state_file, 'wb') as f:
        pickle.dump({
            'drain3_clusters': len(miner.drain.clusters),
            'tc_min': args.tc_min,
            'rf_min': args.rf_min,
        }, f)

    # 持久化 templates_catalog
    templates_catalog = []
    for cluster in sorted(miner.drain.clusters, key=lambda c: c.cluster_id):
        templates_catalog.append({
            'cluster_id': cluster.cluster_id,
            'template': cluster.get_template(),
            'size': cluster.size,
        })
    cat_file = args.output_dir / 'templates_catalog.json'
    with open(cat_file, 'w') as f:
        json.dump(templates_catalog, f, ensure_ascii=False, indent=1)
    print(f'  saved templates_catalog ({len(templates_catalog)} clusters) -> {cat_file}', flush=True)

    # ---------- Pass 2 ----------
    all_stats = []
    splits_to_run = ['val'] if args.smoke else args.splits
    for s in splits_to_run:
        out_file = args.output_dir / f'windows_anomaly_{s}_compressed.json'
        st = compress_one_split(files[s], out_file, miner, ext, s)
        all_stats.append(st)

    # 总 stats
    stats_file = args.output_dir / 'compression_stats.json'
    summary = {
        'config': {
            'drain_depth': args.drain_depth,
            'drain_sim_th': args.drain_sim_th,
            'tc_min': args.tc_min,
            'rf_min': args.rf_min,
        },
        'drain3_total_clusters': len(miner.drain.clusters),
        'valid_tokens_count': len(ext.statistical_miner.valid_tokens_cache),
        'splits': all_stats,
    }
    with open(stats_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n=== ALL DONE ===')
    print(f'stats -> {stats_file}')
    print(json.dumps(summary, indent=2, default=str)[:2000])


if __name__ == '__main__':
    main()
