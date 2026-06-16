#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 iotbench windows_count_100logs.json 过滤异常窗口 + 划分 train/val/test 8/1/1。
输出格式与 TPC/TSBS 一致：windows_anomaly_{train,val,test}.json
"""
import json, time, random
from pathlib import Path
import ijson
from collections import defaultdict, Counter

SRC = Path('./datasets/iotbench_raw/windows_count_100logs.json')
OUT_DIR = Path('./datasets/iotbench_raw')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_LABELS = {'compaction', 'export', 'flush', 'full_cpu', 'full_memory', 'network_bandwidth2'}


def main():
    print(f'流式过滤异常窗口（含至少一个 anomaly type）...', flush=True)
    t0 = time.time()
    anomaly_windows = {}
    label_counts = Counter()

    with open(SRC, 'rb') as f:
        for wid, w in ijson.kvitems(f, ''):
            types = w.get('anomaly_types') or []
            # 取第一个 anomaly type 作为 label
            primary = None
            for t in types:
                if t in ANOMALY_LABELS:
                    primary = t
                    break
            if primary is None:
                continue
            anomaly_windows[wid] = w
            label_counts[primary] += 1
            if len(anomaly_windows) % 5000 == 0:
                print(f'  collected {len(anomaly_windows)} anomaly windows, elapsed {time.time()-t0:.0f}s', flush=True)

    print(f'\n总异常窗口: {len(anomaly_windows)}  耗时 {time.time()-t0:.0f}s')
    print(f'类型分布: {dict(label_counts)}')

    # 按 anomaly type stratified split 8/1/1
    random.seed(42)
    by_label = defaultdict(list)
    for wid, w in anomaly_windows.items():
        types = w.get('anomaly_types') or []
        primary = next((t for t in types if t in ANOMALY_LABELS), None)
        if primary:
            by_label[primary].append(wid)

    train_ids, val_ids, test_ids = [], [], []
    for label, wids in by_label.items():
        random.shuffle(wids)
        n = len(wids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_ids.extend(wids[:n_train])
        val_ids.extend(wids[n_train:n_train + n_val])
        test_ids.extend(wids[n_train + n_val:])

    print(f'\nsplit: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}')

    # 各 split 保存
    for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        out = {wid: anomaly_windows[wid] for wid in ids}
        # label 分布
        lbl_dist = Counter()
        for wid in ids:
            types = anomaly_windows[wid].get('anomaly_types') or []
            primary = next((t for t in types if t in ANOMALY_LABELS), None)
            if primary:
                lbl_dist[primary] += 1
        print(f'  {split_name}: label_dist = {dict(lbl_dist)}')

        fp = OUT_DIR / f'windows_anomaly_{split_name}.json'
        print(f'  saving {fp}...', flush=True)
        t1 = time.time()
        with open(fp, 'w') as f:
            json.dump(out, f, ensure_ascii=False)
        print(f'    saved ({fp.stat().st_size/1024/1024:.1f} MB)  in {time.time()-t1:.0f}s')

    print(f'\n=== ALL DONE ===  total elapsed {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
