#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 TSBS / TPC 的 windows JSON 做两遍流式实体统计（不统计 thread）。

Pass1：仅累计 StatisticalEntityMiner 的全局 token 频次（避免使用
EnhancedEntityExtractor.process_batch_message，防止 _batch_tokens 撑爆内存）。

Pass2：EntityExtractor 提取结构化实体；用固定 valid_tokens 提取统计实体。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import ijson

# 与 enhanced_entity_extractor 中消息切分一致
_MSG_RE = re.compile(r"\]\s+[A-Z]+\s+(.+)$")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from entity_extractor import EntityExtractor  # noqa: E402
from enhanced_entity_extractor import EnhancedEntityExtractor  # noqa: E402


def message_from_raw_line(log_line: str) -> str:
    m = _MSG_RE.search(log_line)
    return m.group(1) if m else log_line


def pass1_token_frequencies(
    paths: List[Path], miner, log_every_windows: int = 2000
) -> Tuple[int, int]:
    """返回 (窗口数近似按日志条数累计的批次数这里用行数, raw_line 条数)。"""
    line_count = 0
    window_count = 0
    since_log = 0
    t0 = time.time()
    for path in paths:
        with path.open("rb") as f:
            for _wid, window in ijson.kvitems(f, ""):
                window_count += 1
                since_log += 1
                logs = window.get("logs") or []
                for log in logs:
                    rl = log.get("raw_line")
                    if not isinstance(rl, str) or not rl:
                        continue
                    line_count += 1
                    msg = message_from_raw_line(rl)
                    toks = miner.extract_tokens_from_message(msg)
                    miner.update_token_frequency(toks)
                if since_log >= log_every_windows:
                    dt = time.time() - t0
                    print(
                        f"[pass1] {path.name} windows≈{window_count} lines={line_count} elapsed={dt:.1f}s",
                        flush=True,
                    )
                    since_log = 0
    return window_count, line_count


def pass2_aggregate(
    paths: List[Path],
    base: EntityExtractor,
    miner,
    valid_tokens: Set[str],
    log_every_windows: int = 2000,
) -> Tuple[Dict[str, Set], Dict[str, str], int, int]:
    agg = {
        "data_regions": set(),
        "nodes": set(),
        "consensus_groups": set(),
        "statistical_entities": set(),
    }
    node_ips: Dict[str, str] = {}
    line_count = 0
    window_count = 0
    since_log = 0
    t0 = time.time()
    for path in paths:
        with path.open("rb") as f:
            for _wid, window in ijson.kvitems(f, ""):
                window_count += 1
                since_log += 1
                logs = window.get("logs") or []
                for log in logs:
                    rl = log.get("raw_line")
                    if not isinstance(rl, str) or not rl:
                        continue
                    line_count += 1
                    ents = base.extract_from_log_line(rl)
                    agg["data_regions"].update(ents.data_regions)
                    agg["nodes"].update(ents.nodes)
                    agg["consensus_groups"].update(ents.consensus_groups)
                    node_ips.update(ents.node_ips)
                    msg = message_from_raw_line(rl)
                    agg["statistical_entities"].update(
                        miner.extract_statistical_entities(msg, valid_tokens)
                    )
                if since_log >= log_every_windows:
                    dt = time.time() - t0
                    print(
                        f"[pass2] {path.name} windows≈{window_count} lines={line_count} elapsed={dt:.1f}s",
                        flush=True,
                    )
                    since_log = 0
    return agg, node_ips, line_count, window_count


def default_paths(name: str, splits: List[str]) -> List[Path]:
    root = Path("./datasets")
    if name == "tsbs":
        base = root / "tsbs_extracted" / "parsed" / "monitor"
    elif name == "tpc":
        base = root / "tpc_extracted" / "parsed" / "monitor"
    else:
        raise ValueError(name)
    return [base / f"windows_anomaly_{sp}.json" for sp in splits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["tsbs", "tpc"],
        choices=["tsbs", "tpc"],
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="要扫描的 split 文件名后缀",
    )
    ap.add_argument("--tc-threshold", type=int, default=2)
    ap.add_argument("--min-rf", type=int, default=5)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "./external_pipeline/06_training/logs/entity_extract_corpus_summary.json"
        ),
    )
    args = ap.parse_args()

    for ds in args.datasets:
        paths = default_paths(ds, args.splits)
        for p in paths:
            if not p.exists():
                print(f"WARN: 缺少文件，跳过: {p}", flush=True)

        paths = [p for p in paths if p.exists()]
        if not paths:
            print(f"ERR: {ds} 无可用 JSON，跳过", flush=True)
            continue

        print(f"\n========== 数据集 {ds.upper()} ==========", flush=True)
        print("文件:", *[str(p) for p in paths], sep="\n  ", flush=True)

        enh = EnhancedEntityExtractor(
            enable_statistical_mining=True,
            tc_threshold=args.tc_threshold,
            min_rf=args.min_rf,
        )
        miner = enh.statistical_miner
        assert miner is not None
        miner.token_frequency.clear()
        miner.valid_tokens_cache = None

        t0 = time.time()
        w1, l1 = pass1_token_frequencies(paths, miner)
        valid_tokens = miner.compute_valid_tokens()
        print(
            f"[{ds}] pass1 完成 windows≈{w1} lines={l1} valid_tokens={len(valid_tokens)} "
            f"time={time.time()-t0:.1f}s",
            flush=True,
        )

        base = EntityExtractor()
        t1 = time.time()
        agg, node_ips, l2, w2 = pass2_aggregate(paths, base, miner, valid_tokens)
        print(
            f"[{ds}] pass2 完成 windows≈{w2} lines={l2} time={time.time()-t1:.1f}s",
            flush=True,
        )

        summary = {
            "dataset": ds,
            "splits": args.splits,
            "paths": [str(p) for p in paths],
            "lines_scanned_pass1": l1,
            "lines_scanned_pass2": l2,
            "statistical_valid_token_count": len(valid_tokens),
            "unique_counts": {
                "data_regions": len(agg["data_regions"]),
                "nodes": len(agg["nodes"]),
                "consensus_groups": len(agg["consensus_groups"]),
                "statistical_entities": len(agg["statistical_entities"]),
                "node_ip_mappings": len(node_ips),
            },
            "entity_types": [
                "data_regions",
                "nodes",
                "consensus_groups",
                "statistical_entities",
                "node_ips (id->ip 映射条数)",
            ],
            "note": "未统计 thread；结构化实体来自 EntityExtractor，统计实体来自 StatisticalEntityMiner（全局 RF）。",
        }

        out_path: Path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 多数据集时追加写入一个 JSON lines 或合并 dict
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        else:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        existing[ds] = summary
        out_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{ds}] 摘要已写入 {out_path}", flush=True)
        print(
            f"[{ds}] 数量: data_regions={summary['unique_counts']['data_regions']} "
            f"nodes={summary['unique_counts']['nodes']} "
            f"consensus_groups={summary['unique_counts']['consensus_groups']} "
            f"statistical_entities={summary['unique_counts']['statistical_entities']} "
            f"node_ip_mappings={summary['unique_counts']['node_ip_mappings']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
