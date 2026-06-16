# 检索语料库 (214 entries) 的构建过程

> METHOD.md 里把"检索"当成黑盒，本文把这个盒子打开 ——
> 214 条 entries 的每一条到底从哪里来、按什么 schema 组织、
> 为什么有的标 1 个 anomaly class 有的标 6 个。

---

## 1. 原始材料

两条来源：

```
10_explanation_data/
├── docs_curated/        # IoTDB 官方文档（手动从 GitHub apache/iotdb 截取）
│   ├── User-Manual/                   (3 files: Maintenance-cmd, Load-Balance, Query-Performance, Data-Recovery)
│   ├── Tools-System/                  (3 files: Monitor-Tool, Maintenance-Tool, Health-Check)
│   ├── Deployment-and-Maintenance/    (3 files: Monitoring-panel-deployment, Database-Resources, Environment-Requirements)
│   ├── Technical-Insider/             (3 files: Cluster-data-partitioning, Encoding-and-Compression, Publication)
│   ├── FAQ/                           (Frequently-asked-questions.md)
│   └── Reference/                     (Common-Config-Manual.md, Status-Codes.md)
│
└── issues/              # GitHub apache/iotdb 已关闭 issues
    ├── all_issues.json             # 449 个 closed issue 的全量元信息
    ├── relevant_issues.json        # 按 anomaly 相关性手筛后剩 29 个
    └── comments/                   # 每个 relevant issue 的 comments
        ├── issue_15182.json
        └── ...
```

### Issue 过滤标准

29 个 relevant_issues 是从 449 个 closed issues 中过滤出来的（手动 + 关键词）：
- 标题或 body 包含 `compaction|flush|wal|memory|oom|gc|cpu|network|ratis|consensus|tsfile|leak|stuck|timeout` 等关键词
- 已被 maintainer 给出回复或 fix 的（state=closed + comments>=1）
- 排除文档错别字、客户端连接问题等与运行时异常无关的 issue

29 个里我又按 anomaly 直接相关度二次筛选，最终只有 14 个被写入 tuples
（剩下 15 个标题/正文相关但根因不是我们关心的 6 类）。

---

## 2. 10 个 JSON 文件的提取

每个 JSON 是一轮"读文档 → 抽出 N 个五元组"的产物。逐个介绍：

### 01_faq_tuples.json (14 tuples, 11.5 KB)

来源：`FAQ/Frequently-asked-questions.md`

提取方法：手动 — 把 FAQ 的 Q&A 改写成 `(symptom, root_cause, fix)` 三元组，加上 anomaly classes 标签。

例：FAQ-1.10 "How to deal with estimated out of memory errors" →

```json
{
  "id": "FAQ-1.10",
  "title": "Estimated OOM during query execution",
  "symptom": "Status code 301: There is not enough memory to execute current fragment instance",
  "root_cause": "Query memory estimation exceeds chunk_timeseriesmeta_free_memory_proportion budget",
  "fix": ["Increase IOTDB_HEAP_SIZE", "Raise datanode_memory_proportion[2] (query)", ...],
  "anomaly_classes": ["full_memory"],
  "iotdb_subsystems": ["Memory", "Query"],
  ...
}
```

### 02_status_codes_categorized.json (18 entries, 9.1 KB)

来源：`Reference/Status-Codes.md`

这文件不是"tuples 列表"——它有两个并行结构：
- `by_anomaly_class: {class: [code dict, ...]}` — 权威码表，每个码标了 anomaly_classes
- `key_anomaly_diagnostic_codes: {class: [code_number, ...]}` — 高信号"smoking gun"码

[`_load_status_codes()`](build_retrieval_index.py#L94-L139) 遍历 by_anomaly_class，按 code 去重，
产生 18 个 entry：`SC-301`, `SC-606`, `SC-712`, `SC-1100` 等。

如果该 code 也出现在 key_anomaly_diagnostic_codes 里，加 `tags=["smoking_gun"]`。

例：`SC-606 WRITE_PROCESS_REJECT` →

```json
{
  "id": "SC-606",
  "title": "Status code 606: WRITE_PROCESS_REJECT",
  "symptom": "Writing data rejected error",
  "root_cause": "Backpressure: system cannot accept more writes (MemTable saturated, Flush queue full)",
  "fix": ["Force FLUSH ON CLUSTER", "Tune reject_proportion / flush_proportion", ...],
  "anomaly_classes": ["flush"],
  "tags": ["status code", "606", "smoking_gun"]
}
```

### 03_config_parameters.json (120 entries, 82.5 KB)

来源：`Reference/Common-Config-Manual.md`

最大的一个文件。原始结构 `parameters: [{name, description, type, default, effective, anomaly_classes, iotdb_subsystems}]`
共 226 条 IoTDB 配置项。

**anomaly_classes 是怎么标的？** —— 半自动 + 人工 review。
对每条 config 的 description 跑关键词匹配 + 手动确认：
- description 含 `memtable|flush|wal_*` → `flush`
- description 含 `compaction|cross_*|inner_*|target_compaction_*` → `compaction`
- description 含 `memory|heap|cache|gc` → `full_memory`
- description 含 `concurrent|thread_pool|worker|mpp` → `full_cpu`
- description 含 `ratis|consensus|replication|heartbeat` → `network_bandwidth2`
- description 含 `export|sync|pipe|cdc|load` → `export`

[`_load_config_params()`](build_retrieval_index.py#L141-L173) **过滤掉 anomaly_classes 为空的**（约 106 条无标签 config），
保留 120 条做检索。每条转成：

```json
{
  "id": "CFG-target_compaction_file_size",
  "title": "Config: target_compaction_file_size",
  "symptom": "",            # config 没有"症状"
  "root_cause": "<description from doc>",   # 复用 description
  "fix": ["Default: 2147483648; Effective: hot-load; Tune via SET CONFIGURATION ..."],
  "config_params": ["target_compaction_file_size"],
  "anomaly_classes": ["compaction"],
  "tags": ["config", "long"]
}
```

### 04-09: 6 个文档主题文件 (合计 48 entries)

| File | Tuples | 来源文档 | 提取方式 |
|---|---|---|---|
| 04_monitor_tool | 8 | Tools-System/Monitor-Tool_apache.md | 手动：每个 Dashboard 板块（CPU/Memory/Disk/GC/Network）→ 1-2 tuple |
| 05_maintenance_commands | 8 | User-Manual/Maintenance-commands_apache.md | 手动：每条运维 SQL（FLUSH, REPAIR DATA, KILL QUERY, SET SYSTEM TO READONLY, SET CONFIGURATION, SHOW DISK_USAGE, DEBUG SQL, SHOW REGIONS）→ 1 tuple |
| 06_user_manual | 7 | Load-Balance.md + Data-Recovery.md + Query-Performance-Analysis.md | 手动：每个核心操作（add datanode, migrate region, REMOVE CONFIGNODE, START REPAIR, EXPLAIN, EXPLAIN ANALYZE）→ 1 tuple |
| 07_tools_system | 6 | Health-Check-Tool.md + Maintenance-Tool_apache.md | 手动：每个 CLI 工具 / 配置技巧（health_check.sh, ulimit, swap off, print-iotdb-data-dir, print-tsfile, print-tsfile-resource-files）→ 1 tuple |
| 08_deployment_maintenance | 12 | Environment-Requirements.md + Database-Resources.md + Monitoring-panel-deployment.md | 手动：sizing 表 + 各 dashboard 面板的具体指标 → tuple |
| 09_technical_insider | 7 | Encoding-and-Compression.md + Cluster-data-partitioning.md | 手动：编码选择、压缩选择、TimePartition 大小、SeriesSlot、replication factor、leader balance、TTL → tuple |

每个 tuple 都有完整 `(title, symptom, root_cause, fix, config_params, anomaly_classes, iotdb_subsystems, version, tags)` 9 个字段。

### 10_relevant_issues_tuples.json (14 entries, 17.6 KB)

来源：`issues/relevant_issues.json` + `issues/comments/issue_*.json`

提取方法：读 issue body + maintainer 回复，归纳成 tuple。最高价值——
因为这是真实运维场景 + 已知 fix。

例：ISS-8403 →
```json
{
  "id": "ISS-8403",
  "title": "Cluster insert with batchSize=10000 throws `RatisRequestFailedException: Ratis request failed`",
  "symptom": "Cluster write fails late in big batch insert (377万 rows). Tablet size 1000 OK, 10000 fails",
  "root_cause": "Ratis limits each write request to 4MB. batchSize=10000 produces ~6MB request, exceeding limit",
  "fix": [
    "SET CONFIGURATION 'data_region_ratis_log_appender_buffer_size_max'='16777216'",
    "Or reduce batch size below 4MB-equivalent (~6000 rows)"
  ],
  "anomaly_classes": ["network_bandwidth2"],
  "iotdb_subsystems": ["Network", "Election"],
  "issue_url": "https://github.com/apache/iotdb/issues/8403"
}
```

---

## 3. 三种加载器统一 schema

[`load_all_entries()`](build_retrieval_index.py#L176-L186) 根据文件名前缀分派：

```python
for fp in sorted(TUPLES_DIR.glob('*.json')):
    if fp.name.startswith('02_status'):    extend(_load_status_codes(fp))    # 18 entries
    elif fp.name.startswith('03_config'):  extend(_load_config_params(fp))   # 120 entries
    else:                                  extend(_load_faq_style(fp))      # 76 entries
```

最终每个 entry 都是同一份 schema：

```python
{
    'id': str,              # SC-606 / CFG-target_compaction_file_size / FAQ-1.10 / ISS-8403 / MON-3 ...
    'origin_file': str,     # which JSON it came from
    'source': str,          # original doc path
    'doc_type': str,        # 'User-Manual' / 'Reference' / 'Tools-System' / ...
    'title': str,
    'symptom': str,         # for configs this is empty
    'root_cause': str,
    'fix': List[str],
    'config_params': List[str],
    'anomaly_classes': List[str],   # subset of 6 anomaly classes
    'iotdb_subsystems': List[str],  # Flush/Compaction/MemTable/TsFile/WAL/Memory/Network/...
    'version': str,
    'tags': List[str],
    'issue_url': str,       # only for issues
}
```

---

## 4. 数量分布与"质量信号"

```
=== 来源分布 ===
  120 Common-Config-Manual          ← 03_config
   18 Status-Codes                  ← 02_status
   14 Frequently-asked-questions    ← 01_faq
   14 relevant_issues               ← 10
   12 Environment-Requirements      ← 08
    8 Monitor-Tool                  ← 04
    8 Maintenance-commands          ← 05
    7 Load-Balance                  ← 06
    7 Encoding-and-Compression      ← 09
    6 Health-Check-Tool             ← 07
   ... (剩下都是 1-3 个 entry 的小源)
   = 214 total

=== 按 anomaly_classes 长度分布 ===
  0 classes: 8 entries     ← 部分 config 仍漏标，下次清理
  1 classes: 138 entries   ← 高信号、单一异常类型
  2 classes: 55 entries    ← 跨子系统的（如 compaction+flush）
  3 classes: 9 entries     ← 较通用
  4-6 classes: 4 entries   ← 通用（监控/资源规划这种全适用的）
```

**138 个单类 entry 是检索质量的基石**。它们直接对应一个异常类型，
配合 `specificity boost (score / √len(classes))` 重排，
确保检索结果不会被 4-6 类的通用 entry 长期占据。

---

## 5. 嵌入用文本怎么拼

[`entry_to_text()`](build_retrieval_index.py#L189-L205) 把每个 entry 拼成一段供 BERT 编码：

```
Title: <title>
Symptom: <symptom>
Root cause: <root_cause>
Fix: <fix items joined by ' | '>
Anomaly classes: <comma-separated>
Subsystems: <comma-separated>
Tags: <comma-separated>
```

例：SC-606 →
```
Title: Status code 606: WRITE_PROCESS_REJECT
Symptom: Writing data rejected error
Root cause: Backpressure: system cannot accept more writes (MemTable saturated, Flush queue full)
Fix: Force FLUSH ON CLUSTER | Tune reject_proportion / flush_proportion | ...
Anomaly classes: flush
Subsystems: MemTable, Flush, WAL
Tags: status code, 606, smoking_gun
```

然后 BERT mean-pool 出 768d 向量，L2-normalize，存入 `retrieval_index.pt` 的 `embeddings` 矩阵（shape=(214, 768)）。

---

## 6. 加新材料 / 扩充语料库

要加入新数据源，只需：

1. **加新文档**：把 `.md` 放进合适的 `docs_curated/` 子目录
2. **写新的 tuples JSON**：复用 `_load_faq_style()` 期望的 schema
   ```json
   {
     "source": "<original doc path>",
     "doc_type": "<category>",
     "tuples": [{"id": "...", "title": "...", "symptom": "...", "root_cause": "...",
                 "fix": ["..."], "anomaly_classes": ["..."], "iotdb_subsystems": ["..."]}]
   }
   ```
3. **重建索引**：`python build_retrieval_index.py --rebuild`

要加入 GitHub 新 issue：append 进 `10_relevant_issues_tuples.json` 的 `tuples` 数组。

不需要改任何加载/嵌入代码。

---

## 7. 现存语料的不足

| 不足 | 影响 | 修复 |
|---|---|---|
| 8 个 entry 仍 anomaly_classes 为空 | 检索时被忽略 | 复审 03_config 与 04_monitor，补标 |
| issue tuples 偏少 (14 个) | 真实运维场景不够多 | 把 issues/comments 里其余 15 个 relevant 补成 tuple |
| `export` 类 entry 偏少 | LLM 解释 export 时可用证据少 | 找更多 IoTConsensus / Sync-Pipe 相关材料 |
| 缺少"时序模式 ↔ 异常类型"的 tuple | LLM 不知道某种 log 频率组合 → 某种异常 | 需要从训练集统计后生成 "pattern tuple"（未来） |
