#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估 explanation_pipeline 的输出质量。

输入: explanation_pipeline 生成的 <output_root>/ 目录
  - 每个 class/ 下面: *_evidence.json, *_prompt.md, *_explanation.md
  - _summary.json
输出: 评估表（markdown 打印到 stdout）

评估维度:
  1. 检索准确度（retrieval correctness）
     - top-1 evidence 的 anomaly_classes 是否包含窗口的 GT class
     - top-K 中至少 1 个 evidence 包含 GT class 的比例
  2. LLM 引用率（citation rate）
     - LLM 输出是否提到至少 1 个被检索到的 tuple ID
  3. 关键词出现率（actions-only 评估）
     - LLM 输出是否提到该 anomaly class 相关的 IoTDB 子系统关键词
       例：full_memory → "heap" "GC" "memory" "OOM"
            compaction → "compact" "tsfile" "merge"
  4. 引用准确率 / Citation Validity（NEW）
     - 输出里 (see TID) / Tuple N 中提到的 ID 是否真的在该窗口的检索 evidence 里
     - 是否引用了根本不存在的 tuple ID
  5. 幻觉检测 / Hallucination Rate（NEW）
     - 输出里提到的 IoTDB config 参数、SQL 命令、工具脚本是否真实存在
     - 用 tuples/03_config_parameters.json + 已知命令/工具白名单做对照
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ANOMALY_CLASSES = ['compaction', 'export', 'flush', 'full_cpu', 'full_memory', 'network_bandwidth2']

# 知识库根目录（与 build_retrieval_index.py 的 TUPLES_DIR 一致）
TUPLES_DIR = Path(__file__).resolve().parent / 'tuples'

# 每类相关的关键词（出现在 LLM 输出说明它确实给了 class-specific 建议）
CLASS_KEYWORDS = {
    'compaction': ['compact', 'tsfile', 'merge', 'cross.space', 'inner.space', 'REPAIR', 'target_compaction'],
    # export: TPC 上 export 标的多是 TsFile 加载/缓存类事件
    'export': ['export', 'sync', 'pipe', 'cdc', 'load.tsfile', 'load_active_listening',
               'last_cache_operation_on_load', 'TsFile load'],
    'flush': ['flush', 'memtable', 'wal', 'FLUSH', 'memtable_size'],
    'full_cpu': ['cpu', 'thread', 'mpp_data_exchange', 'pool_size', 'concurrent', 'mpp_data'],
    'full_memory': ['heap', 'memory', 'GC', 'OOM', 'IOTDB_HEAP_SIZE', 'datanode_memory'],
    'network_bandwidth2': ['network', 'ratis', 'consensus', 'tcp', 'bandwidth', 'replication', 'data_region_ratis'],
}


def _normalize_hyphens(s: str) -> str:
    """Some LLMs emit Unicode non-breaking hyphen (U+2011) or other dashes
    inside tuple IDs (e.g. 'TS-HC‑4'). Normalise so citation detection works."""
    return s.replace('‑', '-').replace('–', '-').replace('—', '-').replace('‐', '-')


# ============================================================================
# 权威实体字典（用于幻觉检测）
# ============================================================================

# IoTDB 内置 SQL 关键字 / 维护命令（手动整理 + 来自 05_maintenance_commands_tuples.json）
KNOWN_SQL_COMMANDS = {
    # Status checking
    'SHOW VERSION', 'SHOW VARIABLES', 'SHOW CURRENT_TIMESTAMP', 'SHOW QUERIES',
    'SHOW REGIONS', 'SHOW DATANODES', 'SHOW CONFIGNODES', 'SHOW CLUSTER',
    'SHOW AVAILABLE URLS', 'SHOW SERVICES', 'SHOW DISK_USAGE', 'SHOW STORAGE GROUP',
    'SHOW DATABASES', 'SHOW TIMESERIES', 'SHOW DEVICES', 'SHOW TTL', 'SHOW PIPES',
    'SHOW CONFIGURATION', 'SHOW QUERY PROCESSLIST', 'SHOW CURRENT_SQL_DIALECT',
    # State changing
    'FLUSH', 'CLEAR CACHE', 'KILL QUERY', 'KILL ALL QUERIES',
    'SET SYSTEM TO READONLY', 'SET SYSTEM TO RUNNING', 'SET CONFIGURATION',
    'SET SQL_DIALECT', 'LOAD CONFIGURATION',
    # Data repair
    'START REPAIR DATA', 'STOP REPAIR DATA',
    # Region
    'MIGRATE REGION', 'REMOVE DATANODE', 'REMOVE CONFIGNODE',
    # Query debug
    'EXPLAIN', 'EXPLAIN ANALYZE', 'DEBUG',
    # Schema
    'CREATE DATABASE', 'CREATE TIMESERIES', 'DELETE TIMESERIES', 'CREATE ALIGNED TIMESERIES',
    'DELETE STORAGE GROUP', 'DELETE DATABASE', 'COUNT TIMESERIES', 'COUNT DEVICES',
    # Write
    'INSERT', 'INSERT INTO', 'DELETE',
    # Read
    'SELECT', 'GROUP BY', 'ORDER BY', 'WHERE',
    # TTL
    'SET TTL', 'UNSET TTL',
}

# IoTDB 工具脚本（基于 04_subgraph_building/lightweight, Tools-System docs）
KNOWN_TOOL_SCRIPTS = {
    'health_check.sh', 'health_check.bat',
    'print-iotdb-data-dir.sh', 'print-iotdb-data-dir.bat',
    'print-tsfile.sh', 'print-tsfile.bat',
    'print-tsfile-sketch.sh', 'print-tsfile-sketch.bat',
    'print-tsfile-resource-files.sh', 'print-tsfile-resource-files.bat',
    'start-cli.sh', 'start-cli.bat',
    'start-datanode.sh', 'start-datanode.bat',
    'start-confignode.sh', 'start-confignode.bat',
    'stop-datanode.sh', 'stop-confignode.sh',
    'destroy-datanode.sh',
    'export-tsfile.sh', 'import-tsfile.sh',
    'export-csv.sh', 'import-csv.sh',
    'export-schema.sh',
    'sbin/start-datanode.sh', 'sbin/start-confignode.sh',
}

# IoTDB 环境变量 / 系统设置
KNOWN_ENV_VARS = {
    'IOTDB_HEAP_SIZE', 'IOTDB_JMX_OPTS', 'CONFIGNODE_JMX_OPTS',
    'MAX_DIRECT_MEMORY_SIZE', 'MAX_HEAP_SIZE', 'HEAP_NEWSIZE',
    'JAVA_HOME', 'JAVA_OPTS', 'IOTDB_HOME', 'IOTDB_CONF', 'CLASSPATH',
    'MEMORY_SIZE',
    # JVM flags
    '-Xms', '-Xmx', '-XX:MaxDirectMemorySize', '-XX:+UseG1GC', '-XX:+UseZGC',
}

# Linux 系统命令（运维相关）
KNOWN_SYS_COMMANDS = {
    'lsof', 'netstat', 'ulimit', 'sysctl', 'systemctl',
    'swapoff', 'swapon', 'free',
    'ps', 'top', 'iostat', 'vmstat',
    'curl', 'docker', 'docker-compose',
    'systemctl stop firewalld', 'systemctl start firewalld', 'systemctl disable firewalld',
}


def _load_valid_config_params() -> set[str]:
    """从 03_config_parameters.json 加载全部 226 个 config 参数名作为字典。"""
    cfg_fp = TUPLES_DIR / '03_config_parameters.json'
    if not cfg_fp.exists():
        return set()
    with open(cfg_fp) as f:
        d = json.load(f)
    names = set()
    for p in d.get('parameters') or []:
        n = p.get('name')
        if n:
            names.add(n)
    return names


def _load_all_tuple_ids() -> set[str]:
    """加载库里 214 张卡片的全部 ID。用于检测 LLM 是否编造了不存在的 tuple ID。"""
    all_ids = set()
    for fp in sorted(TUPLES_DIR.glob('*.json')):
        with open(fp) as f:
            d = json.load(f)
        if fp.name.startswith('02_status'):
            # by_anomaly_class 结构
            for codes in (d.get('by_anomaly_class') or {}).values():
                for c in codes or []:
                    if isinstance(c, dict) and c.get('code') is not None:
                        all_ids.add(f'SC-{c["code"]}')
        elif fp.name.startswith('03_config'):
            for p in d.get('parameters') or []:
                n = p.get('name')
                if n:
                    all_ids.add(f'CFG-{n}')
        else:
            for t in d.get('tuples') or []:
                tid = t.get('id')
                if tid:
                    all_ids.add(tid)
    return all_ids


# 全局缓存，第一次加载
_VALID_CFG_NAMES: set[str] | None = None
_ALL_TUPLE_IDS: set[str] | None = None


def get_valid_cfg_names() -> set[str]:
    global _VALID_CFG_NAMES
    if _VALID_CFG_NAMES is None:
        _VALID_CFG_NAMES = _load_valid_config_params()
    return _VALID_CFG_NAMES


def get_all_tuple_ids() -> set[str]:
    global _ALL_TUPLE_IDS
    if _ALL_TUPLE_IDS is None:
        _ALL_TUPLE_IDS = _load_all_tuple_ids()
    return _ALL_TUPLE_IDS


# ============================================================================
# Entity 抽取（regex）
# ============================================================================

# config 名字（典型形式：snake_case，结尾常含 _threshold/size/count/sec/ms/bytes/in_byte/proportion/class/...）
# 我们从两种语境抽取：
#   (a) SET CONFIGURATION 'xxx'='yyy'
#   (b) 单引号或反引号包围的看起来是 config 名的字符串
_CFG_PATTERNS = [
    re.compile(r"SET CONFIGURATION\s+['\"`]?([a-z][a-z0-9_]+[a-z0-9])['\"`]?\s*=", re.IGNORECASE),
    re.compile(
        r"['`]("                             # 引号开头 + 捕获组开
        r"[a-z][a-z0-9_]{4,}"                # snake_case 主体
        r"_(?:threshold|size|count|time|rate|interval|max|min|enable|num|sec|ms|"
        r"bytes?|in_byte|in_b|class|proportion|mb|kb|pool_size|thread|gb|cache|"
        r"hold|tail|memtable|dir|mode|level|protocol|factor|capacity|window|"
        r"round|deadline|len|length)"        # 常见 config 后缀
        r")['`]"                             # 捕获组关 + 引号关
    ),
]

_SQL_PATTERN = re.compile(
    r"\b(SHOW [A-Z_]+(?:\s+[A-Z_]+)?"
    r"|FLUSH"
    r"|KILL\s+(?:QUERY|ALL QUERIES)"
    r"|START REPAIR DATA"
    r"|STOP REPAIR DATA"
    r"|SET SYSTEM TO (?:READONLY|RUNNING)"
    r"|SET CONFIGURATION"
    r"|SET SQL_DIALECT"
    r"|LOAD CONFIGURATION"
    r"|MIGRATE REGION"
    r"|REMOVE (?:DATANODE|CONFIGNODE)"
    r"|EXPLAIN(?:\s+ANALYZE)?"
    r"|SET TTL|UNSET TTL"
    r"|CREATE (?:DATABASE|TIMESERIES|ALIGNED TIMESERIES)"
    r"|DELETE (?:TIMESERIES|STORAGE GROUP|DATABASE)"
    r"|COUNT (?:TIMESERIES|DEVICES)"
    r")\b"
)

_TOOL_PATTERN = re.compile(
    r"(?:^|[\s/`'\"(])"
    r"((?:health_check|print-iotdb-data-dir|print-tsfile(?:-sketch|-resource-files)?|"
    r"start-cli|start-datanode|start-confignode|stop-datanode|stop-confignode|"
    r"destroy-datanode|export-tsfile|import-tsfile|export-csv|import-csv|export-schema)"
    r"\.(?:sh|bat))"
)

_ENV_PATTERN = re.compile(
    r"\b(IOTDB_HEAP_SIZE|IOTDB_JMX_OPTS|CONFIGNODE_JMX_OPTS|"
    r"MAX_DIRECT_MEMORY_SIZE|MAX_HEAP_SIZE|MEMORY_SIZE|"
    r"JAVA_HOME|JAVA_OPTS|IOTDB_HOME|IOTDB_CONF)\b"
)

# 引用形式
_CITE_TID_PATTERN = re.compile(r"\(see\s+([A-Z]{2,4}-[\w.\-]+?)\)", re.IGNORECASE)
_CITE_TUPLEN_PATTERN = re.compile(r"\bTuple\s*(\d+)\b", re.IGNORECASE)


def extract_entities(text: str) -> dict:
    """从 LLM 输出抽取所有 entity mentions，按类型分桶。"""
    text_norm = _normalize_hyphens(text)
    cfg_mentions = set()
    for p in _CFG_PATTERNS:
        for m in p.findall(text_norm):
            cfg_mentions.add(m.lower())
    sql_mentions = set(m.upper().strip() for m in _SQL_PATTERN.findall(text_norm))
    tool_mentions = set(_TOOL_PATTERN.findall(text_norm))
    env_mentions = set(_ENV_PATTERN.findall(text_norm))
    return {
        'config_names': cfg_mentions,
        'sql_commands': sql_mentions,
        'tool_scripts': tool_mentions,
        'env_vars': env_mentions,
    }


def extract_citations(text: str) -> dict:
    """抽取所有 (see XXX) 形式的 ID 引用 + Tuple N 形式的 rank 引用。"""
    text_norm = _normalize_hyphens(text)
    tids = set(_CITE_TID_PATTERN.findall(text_norm))
    tuplens = set(int(n) for n in _CITE_TUPLEN_PATTERN.findall(text_norm))
    return {'cited_ids': tids, 'cited_tuple_ranks': tuplens}


def detect_hallucinations(entities: dict, valid_cfg: set[str]) -> dict:
    """对比 entity 与权威字典，列出幻觉项。"""
    hall_cfg = entities['config_names'] - {c.lower() for c in valid_cfg}
    # SQL: 检查是否在 KNOWN_SQL_COMMANDS 里（前缀匹配，因为 'SHOW REGIONS WHERE...' 是合法的）
    hall_sql = set()
    for cmd in entities['sql_commands']:
        if not any(cmd.startswith(known) or known.startswith(cmd) for known in KNOWN_SQL_COMMANDS):
            hall_sql.add(cmd)
    hall_tool = entities['tool_scripts'] - KNOWN_TOOL_SCRIPTS
    hall_env = entities['env_vars'] - KNOWN_ENV_VARS
    return {
        'hallucinated_configs': hall_cfg,
        'hallucinated_sql': hall_sql,
        'hallucinated_tools': hall_tool,
        'hallucinated_envs': hall_env,
        'total_hallucinated': len(hall_cfg) + len(hall_sql) + len(hall_tool) + len(hall_env),
    }


def validate_citations(citations: dict, evidence: list, all_tuple_ids: set[str]) -> dict:
    """检查输出里的引用是否真实。返回:
    - valid_id_cites:    引用的 ID 在 evidence 里（真实有效）
    - existing_id_cites: 引用的 ID 在 全 214 张库里，但不在本窗口的 evidence
    - bogus_id_cites:    引用的 ID 完全不存在（编造）
    - valid_tuplen:      Tuple N 中 N 在 [1, len(evidence)] 之间
    - invalid_tuplen:    Tuple N 中 N 越界
    """
    evidence_ids = set(e.get('id', '') for e in (evidence or []))
    k = len(evidence_ids)

    valid_id = citations['cited_ids'] & evidence_ids
    existing_id = (citations['cited_ids'] - evidence_ids) & all_tuple_ids
    bogus_id = citations['cited_ids'] - evidence_ids - all_tuple_ids

    valid_tn = set(n for n in citations['cited_tuple_ranks'] if 1 <= n <= k)
    invalid_tn = citations['cited_tuple_ranks'] - valid_tn

    return {
        'valid_id_cites': valid_id,
        'existing_id_cites': existing_id,
        'bogus_id_cites': bogus_id,
        'valid_tuplen': valid_tn,
        'invalid_tuplen': invalid_tn,
        'total_id_cited': len(citations['cited_ids']),
        'total_tuplen_cited': len(citations['cited_tuple_ranks']),
    }


def load_evidence(out_root: Path) -> list[dict]:
    """Load all *_evidence.json files. Each describes one (window, predicted_class, top-K) triple."""
    records = []
    for cls_dir in out_root.iterdir():
        if not cls_dir.is_dir() or cls_dir.name not in ANOMALY_CLASSES:
            continue
        for ef in cls_dir.glob('*_evidence.json'):
            with open(ef) as f:
                d = json.load(f)
            stem = ef.stem.replace('_evidence', '')
            exp_file = cls_dir / f'{stem}_explanation.md'
            d['explanation_path'] = str(exp_file)
            d['explanation_text'] = exp_file.read_text() if exp_file.exists() else ''
            d['cls_dir'] = cls_dir.name
            records.append(d)
    return records


def evaluate(records: list[dict]) -> dict:
    """Compute aggregated metrics."""
    n_total = len(records)
    by_class = defaultdict(list)
    top1_correct = 0
    topk_correct = 0
    cited_at_least_one = 0
    keyword_hit = 0
    explanation_total_chars = 0
    no_explanation = 0

    # === NEW: 幻觉 + 引用准确率 ===
    valid_cfg = get_valid_cfg_names()
    all_tids = get_all_tuple_ids()
    # 累计计数
    total_cfg_mentions = 0
    total_sql_mentions = 0
    total_tool_mentions = 0
    total_env_mentions = 0
    hall_cfg_total = 0
    hall_sql_total = 0
    hall_tool_total = 0
    hall_env_total = 0
    windows_with_hall = 0
    # 引用准确率
    total_id_cites = 0
    valid_id_cites = 0
    existing_id_cites = 0
    bogus_id_cites = 0
    total_tuplen_cites = 0
    valid_tuplen_cites = 0
    invalid_tuplen_cites = 0
    # 例子收集（debug 用）
    hall_examples = {'configs': Counter(), 'sql': Counter(),
                     'tools': Counter(), 'envs': Counter()}
    bogus_id_examples = Counter()

    for r in records:
        gt_class = r['predicted_class']  # we used GT as predicted in stub
        by_class[gt_class].append(r)

        evidence = r.get('evidence') or []
        # 1) top-1 correct
        if evidence:
            top1_classes = evidence[0].get('anomaly_classes') or []
            if gt_class in top1_classes:
                top1_correct += 1
        # 2) top-K correct
        if any(gt_class in (e.get('anomaly_classes') or []) for e in evidence):
            topk_correct += 1
        # 3) citation rate
        explanation = r.get('explanation_text') or ''
        explanation_total_chars += len(explanation)
        if not explanation.strip() or explanation.startswith('[STUB'):
            no_explanation += 1
            continue
        cited_ids = set()
        explanation_norm = _normalize_hyphens(explanation)
        # 字面 tuple ID 引用
        for e in evidence:
            tid = e.get('id', '')
            if tid and tid in explanation_norm:
                cited_ids.add(tid)
        # 兼容 "Tuple 1/2/3..." 形式（LLM 经常用这种缩写而不是完整 ID）
        if re.search(r'\bTuple\s*\d+\b', explanation_norm, re.IGNORECASE):
            cited_ids.add('__tuple_n__')
        if cited_ids:
            cited_at_least_one += 1
        # 4) keyword hit
        kws = CLASS_KEYWORDS.get(gt_class, [])
        for kw in kws:
            if re.search(re.escape(kw), explanation, re.IGNORECASE):
                keyword_hit += 1
                break

        # 5) === NEW: 幻觉检测 ===
        ent = extract_entities(explanation)
        hall = detect_hallucinations(ent, valid_cfg)
        total_cfg_mentions += len(ent['config_names'])
        total_sql_mentions += len(ent['sql_commands'])
        total_tool_mentions += len(ent['tool_scripts'])
        total_env_mentions += len(ent['env_vars'])
        hall_cfg_total += len(hall['hallucinated_configs'])
        hall_sql_total += len(hall['hallucinated_sql'])
        hall_tool_total += len(hall['hallucinated_tools'])
        hall_env_total += len(hall['hallucinated_envs'])
        if hall['total_hallucinated'] > 0:
            windows_with_hall += 1
        for n in hall['hallucinated_configs']:
            hall_examples['configs'][n] += 1
        for n in hall['hallucinated_sql']:
            hall_examples['sql'][n] += 1
        for n in hall['hallucinated_tools']:
            hall_examples['tools'][n] += 1
        for n in hall['hallucinated_envs']:
            hall_examples['envs'][n] += 1

        # 6) === NEW: 引用准确率 ===
        cite = extract_citations(explanation)
        val = validate_citations(cite, evidence, all_tids)
        total_id_cites += val['total_id_cited']
        valid_id_cites += len(val['valid_id_cites'])
        existing_id_cites += len(val['existing_id_cites'])
        bogus_id_cites += len(val['bogus_id_cites'])
        total_tuplen_cites += val['total_tuplen_cited']
        valid_tuplen_cites += len(val['valid_tuplen'])
        invalid_tuplen_cites += len(val['invalid_tuplen'])
        for bid in val['bogus_id_cites']:
            bogus_id_examples[bid] += 1
        # 把 per-window 结果挂回 records 供按类汇总
        r['_hall'] = hall['total_hallucinated']
        r['_valid_id'] = len(val['valid_id_cites'])
        r['_bogus_id'] = len(val['bogus_id_cites'])
        r['_existing_id'] = len(val['existing_id_cites'])

    n_with_llm = n_total - no_explanation
    return {
        'n_total': n_total,
        'n_with_llm_output': n_with_llm,
        'avg_explanation_chars': explanation_total_chars / max(n_with_llm, 1),
        'top1_retrieval_correct': top1_correct,
        'topk_retrieval_correct': topk_correct,
        'top1_pct': top1_correct / n_total if n_total else 0,
        'topk_pct': topk_correct / n_total if n_total else 0,
        'cited_at_least_one': cited_at_least_one,
        'cite_rate': cited_at_least_one / n_with_llm if n_with_llm else 0,
        'keyword_hit': keyword_hit,
        'keyword_hit_rate': keyword_hit / n_with_llm if n_with_llm else 0,
        # === NEW 指标 ===
        'total_cfg_mentions': total_cfg_mentions,
        'total_sql_mentions': total_sql_mentions,
        'total_tool_mentions': total_tool_mentions,
        'total_env_mentions': total_env_mentions,
        'hallucinated_cfg': hall_cfg_total,
        'hallucinated_sql': hall_sql_total,
        'hallucinated_tool': hall_tool_total,
        'hallucinated_env': hall_env_total,
        'windows_with_hallucination': windows_with_hall,
        'hallucination_window_rate': windows_with_hall / n_with_llm if n_with_llm else 0,
        'cfg_hallucination_rate': hall_cfg_total / max(total_cfg_mentions, 1),
        'sql_hallucination_rate': hall_sql_total / max(total_sql_mentions, 1),
        'tool_hallucination_rate': hall_tool_total / max(total_tool_mentions, 1),
        'env_hallucination_rate': hall_env_total / max(total_env_mentions, 1),
        'total_id_cites': total_id_cites,
        'valid_id_cites': valid_id_cites,
        'existing_id_cites': existing_id_cites,
        'bogus_id_cites': bogus_id_cites,
        'id_cite_validity_rate': valid_id_cites / max(total_id_cites, 1),
        'id_cite_bogus_rate': bogus_id_cites / max(total_id_cites, 1),
        'total_tuplen_cites': total_tuplen_cites,
        'valid_tuplen_cites': valid_tuplen_cites,
        'invalid_tuplen_cites': invalid_tuplen_cites,
        'tuplen_validity_rate': valid_tuplen_cites / max(total_tuplen_cites, 1),
        # Top 例子
        'top_hallucinated_configs': hall_examples['configs'].most_common(15),
        'top_hallucinated_sql': hall_examples['sql'].most_common(10),
        'top_hallucinated_tools': hall_examples['tools'].most_common(5),
        'top_hallucinated_envs': hall_examples['envs'].most_common(5),
        'top_bogus_ids': bogus_id_examples.most_common(15),
        'by_class': dict(by_class),
    }


def print_report(metrics: dict, out_root: Path):
    print('=' * 70)
    print(f'Explanation Pipeline Evaluation — {out_root}')
    print('=' * 70)
    print(f'\n## 总体指标')
    print(f'  窗口总数:                {metrics["n_total"]}')
    print(f'  LLM 有输出的窗口数:      {metrics["n_with_llm_output"]}')
    print(f'  平均输出长度 (chars):    {metrics["avg_explanation_chars"]:.0f}')
    print(f'\n## 检索准确度（GT class ∈ evidence.anomaly_classes）')
    print(f'  Top-1 命中:  {metrics["top1_retrieval_correct"]}/{metrics["n_total"]} ({100*metrics["top1_pct"]:.1f}%)')
    print(f'  Top-K 命中:  {metrics["topk_retrieval_correct"]}/{metrics["n_total"]} ({100*metrics["topk_pct"]:.1f}%)')
    print(f'\n## LLM 输出质量')
    print(f'  引用了至少 1 个检索 tuple ID: {metrics["cited_at_least_one"]}/{metrics["n_with_llm_output"]} ({100*metrics["cite_rate"]:.1f}%)')
    print(f'  提到 class-specific 关键词:   {metrics["keyword_hit"]}/{metrics["n_with_llm_output"]} ({100*metrics["keyword_hit_rate"]:.1f}%)')

    # === NEW: 幻觉率 ===
    print(f'\n## 幻觉率（提到的实体是否真实存在）')
    print(f'  Config 参数: {metrics["hallucinated_cfg"]:>5} 幻觉 / {metrics["total_cfg_mentions"]:>5} 提及 ({100*metrics["cfg_hallucination_rate"]:>5.1f}%)')
    print(f'  SQL 命令:   {metrics["hallucinated_sql"]:>5} 幻觉 / {metrics["total_sql_mentions"]:>5} 提及 ({100*metrics["sql_hallucination_rate"]:>5.1f}%)')
    print(f'  工具脚本:   {metrics["hallucinated_tool"]:>5} 幻觉 / {metrics["total_tool_mentions"]:>5} 提及 ({100*metrics["tool_hallucination_rate"]:>5.1f}%)')
    print(f'  环境变量:   {metrics["hallucinated_env"]:>5} 幻觉 / {metrics["total_env_mentions"]:>5} 提及 ({100*metrics["env_hallucination_rate"]:>5.1f}%)')
    print(f'  含幻觉的窗口数: {metrics["windows_with_hallucination"]}/{metrics["n_with_llm_output"]} ({100*metrics["hallucination_window_rate"]:>5.1f}%)')
    if metrics.get('top_hallucinated_configs'):
        print(f'  Top-10 幻觉 config 名（按出现窗口数）:')
        for name, n in metrics['top_hallucinated_configs'][:10]:
            print(f'    {name:<55} {n} 次')
    if metrics.get('top_hallucinated_sql'):
        print(f'  Top 幻觉 SQL:')
        for cmd, n in metrics['top_hallucinated_sql'][:5]:
            print(f'    {cmd:<55} {n} 次')
    if metrics.get('top_hallucinated_tools'):
        print(f'  Top 幻觉工具:')
        for t, n in metrics['top_hallucinated_tools']:
            print(f'    {t:<55} {n} 次')

    # === NEW: 引用准确率 ===
    print(f'\n## 引用准确率（输出里的 ID 是否真的在检索 evidence 里）')
    print(f'  Tuple ID 引用 (see XXX) 总数:    {metrics["total_id_cites"]}')
    print(f'    ✓ Valid (在本窗口 evidence 内):  {metrics["valid_id_cites"]:>5} ({100*metrics["id_cite_validity_rate"]:>5.1f}%)')
    print(f'    ◐ Existing (在库内但非本窗口):    {metrics["existing_id_cites"]:>5}')
    print(f'    ✗ Bogus (完全编造的 ID):         {metrics["bogus_id_cites"]:>5} ({100*metrics["id_cite_bogus_rate"]:>5.1f}%)')
    print(f'  "Tuple N" 短引用总数:             {metrics["total_tuplen_cites"]}')
    print(f'    ✓ Valid (1 ≤ N ≤ k):             {metrics["valid_tuplen_cites"]:>5} ({100*metrics["tuplen_validity_rate"]:>5.1f}%)')
    print(f'    ✗ Invalid (N 越界):              {metrics["invalid_tuplen_cites"]:>5}')
    if metrics.get('top_bogus_ids'):
        print(f'  Top 编造的 ID:')
        for bid, n in metrics['top_bogus_ids'][:10]:
            print(f'    {bid:<55} {n} 次')

    print(f'\n## 按类细分')
    valid_cfg = get_valid_cfg_names()
    all_tids = get_all_tuple_ids()
    print(f'{"Class":<22} {"N":>4}  {"Top-1":>7}  {"Cite":>7}  {"KwHit":>7}  {"Hall%":>7}  {"CiteVld%":>9}')
    for cls in ANOMALY_CLASSES:
        recs = metrics['by_class'].get(cls, [])
        if not recs:
            print(f'  {cls:<20} {0:>4}')
            continue
        n = len(recs)
        top1 = sum(1 for r in recs
                   if r.get('evidence') and cls in (r['evidence'][0].get('anomaly_classes') or []))
        cite = sum(1 for r in recs
                   if r.get('explanation_text')
                   and (any(e.get('id', '') in _normalize_hyphens(r['explanation_text'])
                            for e in (r.get('evidence') or []))
                        or re.search(r'\bTuple\s*\d+\b',
                                      _normalize_hyphens(r['explanation_text']),
                                      re.IGNORECASE)))
        kws = CLASS_KEYWORDS.get(cls, [])
        kw = sum(1 for r in recs
                 if r.get('explanation_text')
                 and any(re.search(re.escape(k), r['explanation_text'], re.IGNORECASE) for k in kws))
        # NEW: per-class hallucination + cite validity
        n_hall = sum(1 for r in recs if r.get('_hall', 0) > 0)
        total_cited = sum(r.get('_valid_id', 0) + r.get('_existing_id', 0) + r.get('_bogus_id', 0) for r in recs)
        total_valid = sum(r.get('_valid_id', 0) for r in recs)
        cite_vld_pct = 100 * total_valid / total_cited if total_cited else 100.0
        hall_pct = 100 * n_hall / n
        print(f'  {cls:<20} {n:>4}  {top1}/{n:<5} {cite}/{n:<5} {kw}/{n:<5} {hall_pct:>5.1f}%  {cite_vld_pct:>7.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, required=True,
                    help='explanation_pipeline 的 --output-root 目录')
    args = ap.parse_args()
    records = load_evidence(args.output_root)
    if not records:
        print(f'No records found under {args.output_root}')
        return
    metrics = evaluate(records)
    print_report(metrics, args.output_root)
    # Also save to JSON for further analysis
    summary_out = args.output_root / '_evaluation.json'
    summary_out.write_text(json.dumps({
        k: v for k, v in metrics.items() if k != 'by_class'
    }, indent=2, ensure_ascii=False))
    print(f'\nSaved metrics JSON: {summary_out}')


if __name__ == '__main__':
    main()
