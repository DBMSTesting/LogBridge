#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 干净实体抽取器 — 替代 enhanced_entity_extractor 用于压缩与 KG 构建。

四层设计：
  Layer 1  领域结构化实体  (TPC 真实存在的 token 形态)
    - Database:        root.<sg>(.<ts>)?         (root.server)
    - ConsensusGroup:  group-<hexid>             ( <digit>@group-000100000002 -> group-000100000002 )
    - NodeRef:         node<digit>               ( "1@group-..." 中的 1@ )
    - TsFile:          tsfile-<short-id>         ( 1705243672551-1-0-0.tsfile -> tsfile-1705243672551 )
    - LogInprogress:   log_inprogress_<N>
    - ClientId:        client-<HEX>              (统一截取前 6 位避免散开)

  Layer 2  子系统关键词标签 (按业务语义白名单)
    Subsystem:Flush  Subsystem:Compaction  Subsystem:MemTable  Subsystem:TsFile
    Subsystem:WAL    Subsystem:Memory      Subsystem:Network   Subsystem:Compression
    Subsystem:Election  Subsystem:Schema   Subsystem:GC        Subsystem:Query
    Subsystem:Sync   Subsystem:Heartbeat   Subsystem:Cache     Subsystem:Procedure

  Layer 3  Thread 归一化（细+粗）
    fine:   原始线程名（保留给 LogInstance 当独有标识）
    coarse: ThreadGroup:<去掉 pool-N、尾部 -N、SubTask-N 的简称>
    例： pool-7-IoTDB-Flush-SubTask-1  -> ThreadGroup:IoTDB-Flush-SubTask
         1@group-000100000002-StateMachineUpdater -> ThreadGroup:StateMachineUpdater
         grpc-default-executor-4 -> ThreadGroup:grpc-default-executor

  Layer 4  统计实体黑名单（替代原 RF/TC 自动学习）
    黑名单（DROP）：
      - 长度 >= 40
      - 纯 hex (>=12 char)
      - UUID
      - Java Class:Line (Raft/Netty/gRPC 框架的全部 drop)
      - Raft/protocol 常量: t:N  padding=N  streamId=N  arrayIndex=N
                            length=N  endStream=*  followerCommit=N
                            commitIndex=N  matchIndex=N  nextIndex=N
                            cid=N  seq=N  index=N
      - 纯数字 token
    白名单（KEEP）：
      - 含业务关键词的 token （Flush/Compaction/MemTable/TsFile/WAL/Memory/...）
      - 来自 IoTDB 包（org.apache.iotdb.*）的 Class:Line 引用

输出仍然兼容原 EnhancedLogEntities 字段名，下游 build_knowledge_graph.py 不需要改。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


# =====================================================================
# 1. 各层正则
# =====================================================================

# ---- Layer 1: 领域结构化实体 ----
RE_THREAD_HEAD = re.compile(
    r'^\s*-?\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+\[([^\]]+)\]\s+[A-Z]+\s+'
)
RE_RATIS_GROUP = re.compile(r'(\d+)@group-([0-9a-fA-F]+)')        # 1@group-000100000002
RE_TSFILE      = re.compile(r'\b(\d{10,})-\d+-\d+-\d+\.tsfile\b')  # 1705243672551-1-0-0.tsfile
RE_LOGPROG     = re.compile(r'\b(log_inprogress_\d+)\b')
RE_CLIENT      = re.compile(r'\bclient-([0-9A-Fa-f]{6,})\b')
RE_ROOT        = re.compile(r'\b(root\.[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)?)\b')
RE_OLD_DATA_REGION = re.compile(r'(?:DataRegion|root\.sg\d+)\[(\d+)\]', re.IGNORECASE)
RE_OLD_NODE_ID = re.compile(r'nodeId\s*=\s*(\d+)', re.IGNORECASE)
RE_OLD_PEER = re.compile(
    r'Peer\s*\{\s*groupId\s*=\s*[^,]+,\s*endpoint\s*=\s*TEndPoint\s*\(\s*ip\s*:\s*'
    r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s*,\s*port\s*:\s*\d+\s*\)\s*,\s*nodeId\s*=\s*(\d+)\s*\}',
    re.IGNORECASE
)
RE_OLD_TENDPT = re.compile(
    r'TEndPoint\s*\(\s*ip\s*:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', re.IGNORECASE
)
RE_OLD_CGID = re.compile(
    r'TConsensusGroupId\s*\(\s*type\s*:\s*(\w+)\s*,\s*id\s*:\s*(\d+)\s*\)', re.IGNORECASE
)

# ---- Layer 2: 子系统关键词字典 ----
SUBSYSTEM_KEYWORDS = [
    ('Flush',       [r'\bFlush\b', r'IoTDB-Flush', r'flush\b']),
    ('Compaction',  [r'\bCompaction\b', r'\bcompact\b', r'CompactionRecover']),
    ('MemTable',    [r'\bMemTable\b']),
    ('TsFile',      [r'\bTsFile\b', r'TsFileIOWriter', r'TsFileResource', r'\.tsfile\b']),
    ('WAL',         [r'\bWAL\b', r'log_inprogress', r'SegmentedRaftLog', r'WriteAheadLog']),
    ('Memory',      [r'\bSystemInfo\b', r'OutOfMemory', r'\bmemory\b', r'MemoryManager']),
    ('Network',     [r'\bNetty\b', r'\bgRPC\b', r'grpc-', r'INBOUND', r'OUTBOUND', r'\bsocket\b']),
    ('Compression', [r'CompressionRatio', r'compression']),
    ('Election',    [r'\belected\b', r'requestVote', r'\bLeader\b.*\bElection\b']),
    ('Schema',      [r'SchemaRegion', r'\bSchemaCache\b', r'\bSchemaTree\b']),
    ('GC',          [r'\bFullGC\b', r'\bMinorGC\b', r'GarbageCollect']),
    ('Query',       [r'\bQueryStart\b', r'QueryEngine', r'Coordinator']),
    ('Cache',       [r'PartitionCache', r'\bSchemaCache\b', r'BloomFilter']),
    ('Procedure',   [r'ProcedureExecutor', r'CreateRegionGroups', r'StateMachineProcedure']),
]
SUBSYSTEM_RE = [(name, re.compile('|'.join(pats), re.IGNORECASE))
                for name, pats in SUBSYSTEM_KEYWORDS]

# ---- Layer 3: Thread 归一化 ----
# 规则：
#   1) 去掉 "pool-<N>-" 前缀  (pool-7-IoTDB-Flush-... -> IoTDB-Flush-...)
#   2) 去掉 "<digit>@group-<hex>-" 前缀  (1@group-000100000002-StateMachineUpdater -> StateMachineUpdater)
#   3) 去掉尾部数字序号 -<N>$  (-1, -2, -SubTask-1 -> -SubTask)
#   4) 去掉 server-thread 后的数字  (3-server-thread1 -> server-thread)
#   5) 去掉 grpc-default-executor 后的数字  (grpc-default-executor-4 -> grpc-default-executor)
#   6) 去掉 grpc-default-worker-ELG 后的数字
RE_TG_POOL_PREFIX  = re.compile(r'^pool-\d+-')
RE_TG_GROUP_PREFIX = re.compile(r'^\d+@group-[0-9a-fA-F]+-')
RE_TG_SERVER       = re.compile(r'^\d+-server-thread\d*$')
RE_TG_GRPC_EXEC    = re.compile(r'^grpc-default-executor-\d+$')
RE_TG_GRPC_WORKER  = re.compile(r'^grpc-default-worker(-ELG-\d+-\d+)?$')
RE_TG_TAIL_NUM     = re.compile(r'-\d+$')
RE_TG_TRACE_SUFFIX = re.compile(r'\$\d{8}_\d{6}_\d+_\d+$')   # IoTDB query trace id


def normalize_thread(thread: str) -> Optional[str]:
    """返回 ThreadGroup 形式的归一名；输入若太短或不合法则返回 None。"""
    if not thread:
        return None
    s = thread.strip()
    # server-thread 类
    if RE_TG_SERVER.match(s):
        return 'server-thread'
    # grpc 类
    if RE_TG_GRPC_EXEC.match(s):
        return 'grpc-default-executor'
    if RE_TG_GRPC_WORKER.match(s):
        return 'grpc-default-worker'
    # 去掉前缀
    s = RE_TG_POOL_PREFIX.sub('', s)
    s = RE_TG_GROUP_PREFIX.sub('', s)
    # 去掉 query trace id 后缀 ($YYYYMMDD_HHMMSS_NNNNN_N)
    s = RE_TG_TRACE_SUFFIX.sub('', s)
    # 反复去掉尾部 -N
    prev = None
    while prev != s:
        prev = s
        s = RE_TG_TAIL_NUM.sub('', s)
    s = s.strip('-')
    return s if len(s) >= 2 else None


# ---- Layer 4: 统计实体黑/白名单 ----
RE_PURE_HEX     = re.compile(r'^[0-9a-fA-F]+$')
RE_UUID         = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
RE_PURE_DIGIT   = re.compile(r'^\d+$')
# 形如 t:1 / padding=0 / nextIndex=2509 / streamId=7 等协议常量与时间相关变量
PROTOCOL_KV_KEYS = {
    't', 'padding', 'streamId', 'arrayIndex', 'length', 'endStream',
    'followerCommit', 'commitIndex', 'matchIndex', 'nextIndex', 'cid',
    'seq', 'index', 'old', 'new', 'startIndex', 'endIndex', 'maxIndex',
    'minIndex', 'pid', 'port', 'term'
}
RE_KV = re.compile(r'^([A-Za-z_][A-Za-z_0-9]*)[:=](.+)$')

# Raft / Netty / gRPC 框架代码引用 (Class:LineNumber 格式) — 全部 drop
FRAMEWORK_CLASS_PREFIXES = (
    'SegmentedRaftLog', 'StateMachineUpdater', 'RaftServerImpl',
    'PendingRequests', 'LeaderStateImpl', 'FollowerInfoImpl',
    'WatchRequests', 'AbstractInternalLogger', 'GrpcServerProtocolService',
    'GrpcServerProtocolClient', 'RaftLog', 'LogAppender', 'LogEntryHeader',
    'GrpcServer', 'GrpcLogAppender',
)
RE_CLASSLINE = re.compile(r'^([A-Z][A-Za-z0-9_$]+):(\d{1,4})$')

# IoTDB 业务包名前缀 — 含这些 token 的 Class:Line 引用直接放行
BUSINESS_CLASS_KEYWORDS = (
    'TsFile', 'Flush', 'Compaction', 'MemTable', 'CompressionRatio',
    'SystemInfo', 'PartitionCache', 'SchemaCache', 'IoTDB',
    'StorageEngine', 'DataRegion',
)


def is_business_class(token: str) -> bool:
    return any(kw in token for kw in BUSINESS_CLASS_KEYWORDS)


def stat_token_filter(tok: str) -> bool:
    """True = keep, False = drop."""
    if not tok or len(tok) < 3:
        return False
    if len(tok) >= 40:
        return False
    if RE_UUID.match(tok):
        return False
    if RE_PURE_HEX.match(tok) and len(tok) >= 12:
        return False
    if RE_PURE_DIGIT.match(tok):
        return False
    # 文件后缀类（.tsfile / .wal 等）已经被 Layer 1 显式抽取，丢弃 Layer 4 的原始形式
    if tok.endswith('.tsfile') or tok.endswith('.wal') or tok.endswith('.log'):
        return False
    # 包名片段（iotdb.tsfile / ratis.thirdparty 等）— 含 . 且无运算符
    if '.' in tok and ':' not in tok and '=' not in tok and '[' not in tok:
        # 这一定是 package fragment，不是业务实体
        return False
    # 端点类 N:port — 在统计层无意义（Layer 1 的 NodeIP 已覆盖）
    if RE_KV.match(tok) and RE_KV.match(tok).group(1).isdigit():
        return False

    # KV 类协议常量
    m = RE_KV.match(tok)
    if m and m.group(1) in PROTOCOL_KV_KEYS:
        return False

    # Class:Line 引用
    m = RE_CLASSLINE.match(tok)
    if m:
        cls = m.group(1)
        if cls.startswith(FRAMEWORK_CLASS_PREFIXES):
            return False
        if is_business_class(tok):
            return True
        # 其他未明确分类的 class:line — 默认 drop（保守清洁）
        return False

    return True


# ---- Token 候选提取（沿用原 enhanced 的规则） ----
TOKEN_PATTERNS = [
    r'\b[a-zA-Z_]+[0-9]+[a-zA-Z0-9_]*\b',
    r'\b[0-9]+[a-zA-Z_]+[a-zA-Z0-9_]*\b',
    r'\b[a-zA-Z0-9_]+\[[0-9]+\]\b',
    r'\b[a-zA-Z0-9_]+=[0-9]+\b',
    r'\b[a-zA-Z0-9_]+:[0-9]+\b',
    r'\b[a-zA-Z0-9_-]+\.(?:tsfile|wal|log|txt|conf)\b',
]
COMPILED_TOKEN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TOKEN_PATTERNS]


# =====================================================================
# 2. 数据类
# =====================================================================

@dataclass
class CleanLogEntities:
    """与下游 build_knowledge_graph.py 兼容的字段名 + 内部 v2 增强字段"""
    # 兼容字段（原 EnhancedLogEntities 接口）
    data_regions: Set[str] = field(default_factory=set)
    nodes: Set[str] = field(default_factory=set)
    node_ips: Dict[str, str] = field(default_factory=dict)
    thread: Optional[str] = None
    consensus_groups: Set[str] = field(default_factory=set)
    statistical_entities: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            'data_regions':         sorted(self.data_regions),
            'nodes':                sorted(self.nodes),
            'node_ips':             self.node_ips,
            'thread':               self.thread,
            'consensus_groups':     sorted(self.consensus_groups),
            'statistical_entities': sorted(self.statistical_entities),
        }


# =====================================================================
# 3. 主提取函数
# =====================================================================

def extract_message(log_line: str) -> str:
    """剥离 时间戳 + [线程] + LEVEL，返回消息体；失败时返回原始行。"""
    m = re.search(r'\]\s+[A-Z]+\s+(.+)$', log_line)
    return m.group(1) if m else log_line


def extract_clean_entities(log_line: str) -> CleanLogEntities:
    """对单行日志做 v2 干净抽取。"""
    e = CleanLogEntities()

    # ---- Layer 1 ----
    # Thread (full)
    m = RE_THREAD_HEAD.search(log_line)
    if m:
        e.thread = m.group(1)

    # Database / StorageGroup
    for db in RE_ROOT.findall(log_line):
        # 'root.*' / 'root.server' / 'root.server.xxx' 等
        e.data_regions.add(f'Database:{db}')

    # Old-format DataRegion (DataRegion[N]) — 保留以备 TSBS 复用
    for dr_id in RE_OLD_DATA_REGION.findall(log_line):
        e.data_regions.add(f'DataRegion:{dr_id}')

    # Consensus group — TPC 真实形态
    for node_pref, gid in RE_RATIS_GROUP.findall(log_line):
        e.consensus_groups.add(f'group-{gid}')
        e.nodes.add(f'node{node_pref}')

    # Old-format ConsensusGroupId
    for cg_type, cg_id in RE_OLD_CGID.findall(log_line):
        e.consensus_groups.add(f'{cg_type}:{cg_id}')

    # Old-format Peer (with IP) -> 仍然提取 nodeId 和 IP
    for ip, nid in RE_OLD_PEER.findall(log_line):
        e.nodes.add(nid)
        e.node_ips[nid] = ip
    # 兜底 nodeId=N
    for nid in RE_OLD_NODE_ID.findall(log_line):
        e.nodes.add(nid)

    # TsFile  (将 ts-prefix 当 id)
    for ts_prefix in RE_TSFILE.findall(log_line):
        e.statistical_entities.add(f'TsFile:{ts_prefix}')

    # LogInprogress
    for lp in RE_LOGPROG.findall(log_line):
        e.statistical_entities.add(f'LogProgress:{lp}')

    # ClientId
    for cid in RE_CLIENT.findall(log_line):
        e.statistical_entities.add(f'ClientId:{cid[:8]}')   # 取前 8 位

    # ---- Layer 2: Subsystem keyword tags ----
    msg = extract_message(log_line)
    for sub_name, sub_re in SUBSYSTEM_RE:
        if sub_re.search(msg):
            e.statistical_entities.add(f'Subsystem:{sub_name}')

    # ---- Layer 3: Thread group (粗) ----
    if e.thread:
        tg = normalize_thread(e.thread)
        if tg:
            e.statistical_entities.add(f'ThreadGroup:{tg}')

    # ---- Layer 4: 过滤后的统计实体 ----
    # 候选 token
    candidates: Set[str] = set()
    for pat in COMPILED_TOKEN_PATTERNS:
        candidates.update(pat.findall(msg))
    # 应用过滤
    for tok in candidates:
        if stat_token_filter(tok):
            # 这些已经被 Layer 1/2 覆盖的不重复加（可选，但加了也无害，去重即可）
            e.statistical_entities.add(tok)

    return e


# =====================================================================
# 4. 兼容封装：和原 enhanced_entity_extractor API 对齐
# =====================================================================

class CleanEntityExtractor:
    """与 EnhancedEntityExtractor 接口对齐，但内部用 v2 干净逻辑。"""

    def __init__(self, *args, **kwargs):
        # 不再做 RF/TC 学习，所以参数全部忽略
        self._batch_mode = False

    def start_batch_mode(self):
        self._batch_mode = True

    def process_batch_message(self, message: str):  # 兼容空操作
        return

    def finish_batch_mode(self):
        self._batch_mode = False

    @property
    def statistical_miner(self):
        # 兼容原接口（compress_windows.py 里读了 valid_tokens_cache 等）
        return _DummyMiner()

    def extract_from_log_line(self, log_line: str) -> CleanLogEntities:
        return extract_clean_entities(log_line)

    def extract_from_message(self, message: str, thread: Optional[str] = None) -> CleanLogEntities:
        # 直接拼一个虚假行
        fake = f'2024-01-01 00:00:00,000 [{thread or "unknown"}] INFO  noop - {message}'
        return extract_clean_entities(fake)


class _DummyMiner:
    token_frequency = {}
    valid_tokens_cache = set()

    def compute_valid_tokens(self):
        return set()


def extract_clean_entities_from_log_line(log_line: str) -> CleanLogEntities:
    return extract_clean_entities(log_line)


# =====================================================================
# 5. 单元测试
# =====================================================================

if __name__ == '__main__':
    test_lines = [
        '- 2024-01-14 14:53:26,293 [pool-6-IoTDB-Flush-4] INFO  org.apache.iotdb.db.storageengine.dataregion.flush.CompressionRatio:103 - Compression ratio is 18.22897380274395',
        '- 2024-01-14 14:50:04,653 [1@group-000100000002-SegmentedRaftLogWorker] DEBUG org.apache.ratis.server.raftlog.segmented.SegmentedRaftLogWorker:370 - 1@group-000100000002-SegmentedRaftLogWorker: flush SegmentedRaftLogOutputStream(/iotdb/data/datanode/consensus/data_region/47474747-4747-4747-4747-000100000002/current/log_inprogress_830)',
        '- 2024-01-14 14:51:05,434 [grpc-default-worker-ELG-3-3] DEBUG org.apache.ratis.thirdparty.io.netty.util.internal.logging.AbstractInternalLogger:214 - [id: 0x133f2326, L:/172.20.0.12:10760 - R:/172.20.0.11:56058] INBOUND DATA: streamId=7 padding=0 endStream=false length=16384 bytes=0275739f21b00000027573ac68200000027573b0d4f00000027573c9b4200000027573e971200000027573edddf00000027574049a4000000275741822400000',
        '- 2024-01-14 14:55:59,999 [2-server-thread1] DEBUG org.apache.ratis.server.impl.RaftServerImpl:1436 - 2@group-000100000001: succeeded to handle AppendEntries. Reply: 3<-2#15152:OK-t1,SUCCESS,nextIndex=5491,followerCommit=5489,matchIndex=5490',
        '- 2024-01-14 14:47:46,277 [pool-27-IoTDB-ClientRPC-Processor-3$20240114_144746_00004_1] DEBUG org.apache.iotdb.db.queryengine.plan.analyze.cache.partition.PartitionCache:276 - [Database Cache] miss when search device root.server',
        '- 2024-01-14 14:47:52,564 [1@group-000100000002-StateMachineUpdater] DEBUG org.apache.iotdb.tsfile.write.writer.RestorableTsFileIOWriter:95 - 1705243672551-1-0-0.tsfile is opened.',
    ]
    print('=== v2 clean entity extractor 测试 ===\n')
    for i, line in enumerate(test_lines, 1):
        e = extract_clean_entities(line)
        print(f'[Test {i}] {line[:140]}...')
        print(f'  thread:    {e.thread}')
        print(f'  databases: {sorted(e.data_regions)}')
        print(f'  nodes:     {sorted(e.nodes)}')
        print(f'  cgs:       {sorted(e.consensus_groups)}')
        print(f'  stat_ents: {sorted(e.statistical_entities)}')
        print()
