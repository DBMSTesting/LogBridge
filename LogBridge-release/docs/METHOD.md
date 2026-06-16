# 解释生成 (Explanation Generation) 方法详解

> 目的：把 HGT 分类器对一个日志窗口的"硬分类结果"（6 类异常之一）转换为
> 一份**带证据引用、面向运维**的根因分析报告。
>
> 本文档基于一个真实端到端样例完整走一遍流水线。

---

## 0. 全景

```
┌────────────────────────┐                  ┌──────────────────────┐
│ HGT classifier         │   predicted_     │ explanation pipeline │
│  (training-time logic) │  ─class───────▶ │  (this directory)    │
└────────────────────────┘                  └──────────────────────┘

  test window  ──▶  context  ──▶  retrieval query  ──▶  evidence (top-K)
                                                                │
                                                                ▼
                                    prompt template  ──▶  Generator LLM
                                                                │
                                                                ▼
                                                          explanation.md
```

**输入**：单个窗口（来自 `*_compressed.json`）+ 分类器预测的类别  
**输出**：一份 markdown 根因报告（≤ 400 字，强制引用 tuple ID）

**核心代码文件**：
- [explanation_pipeline.py](explanation_pipeline.py) — 整条流水线
- [build_retrieval_index.py](build_retrieval_index.py) — 检索索引（构建 + retrieve()）
- [demo_pipeline_output/](demo_pipeline_output/) — 6 个 demo 窗口的产物

**知识库**：[10_explanation_data/extracted_tuples/](../10_explanation_data/extracted_tuples/) 的 10 个 JSON
共 214 条 entry（76 FAQ-style + 24 status codes + 114 config params）。

---

## 1. 选择的样例

`label4:count:26667:1705029224.339`  ——  iotbench 测试集，ground-truth = `network_bandwidth2`

| 字段 | 值 |
|---|---|
| 时间窗口 | `2024-01-12T03:13:44.339000+00:00` → `2024-01-12T03:13:44.347000+00:00` (8 ms) |
| `log_count_original` | 100 (原始日志) |
| `log_count` | 9 (Drain3 压缩后唯一模板数) |
| `anomaly_types[0]` (用作预测) | `network_bandwidth2` |

**特殊性**：iotbench 窗口的模板与 anomaly 类型**几乎无字面关联**（前面诊断证据：每类
top 模板 cosine = 1.000）。因此 LLM 必须依靠**频率分布的细节** + **检索证据**，
不能依靠日志关键词。这反而让样例更有说服力。

---

## 2. 流水线 5 步分解

### 步骤 1 — `extract_window_context()` — 提取窗口上下文

入口：[explanation_pipeline.py:47](explanation_pipeline.py#L47)

把 100 条原始日志（已压缩为 9 条聚合 log + `count` 字段）转换为结构化摘要：

```
top-5 templates by count:
  cluster_3  x38  "bytes, total memory size: <*> bytes."
  cluster_5  x12  "LogDispatcher$LogDispatcherThread:360 - <*> startIndex: <*>
                   maxIndex: <*> pendingEntries size: <*> bufferedEntries size: <*>"
  cluster_9  x12  "DataRegion[1] : accumulated a <*> <*> <*> <*> from <*> <*> <*>"
  cluster_2  x10  "IndexController:77 - update index from currentIndex <*> to <*>"
  cluster_4  x9   "Send <*> <*> to ConsensusGroup:TConsensusGroupId(type:DataRegion, <*>"

top entities:
  data_regions x9, nodes x9, consensus_groups x9, statistical_entities x9, ...
```

`count` 字段是关键 — 8ms 窗口里 `cluster_5` 出现 12 次 ≠ 普通心跳，是 LogDispatcher
在被压力打满的特征。

### 步骤 2 — `build_retrieval_query()` — 构建检索 query

入口：[explanation_pipeline.py:88](explanation_pipeline.py#L88)

把上面所有内容拼成一段约 1.5KB 的文本：

```
Predicted anomaly: network_bandwidth2
Top log templates by count:
  [cluster_3 x38] 2024-01-12 <*> <*> INFO <*> - <*> <*> <*> bytes, total memory size: <*> bytes.
  [cluster_5 x12] 2024-01-12 <*> <*> INFO org.apache.iotdb.consensus.iot.logdispatcher.LogDispatcher$LogDispatcherThread:360 - <*> startIndex: <*> ...
  ...
Frequent entities: data_regions(x9), nodes(x9), consensus_groups(x9), ...
```

### 步骤 3 — `retrieve()` — 向 214 条 entry 检索 top-5

入口：[build_retrieval_index.py:retrieve](build_retrieval_index.py)

1. **编码**：query 经 BERT (bert-base-uncased, mean-pool, 768d) → 单位向量
2. **打分**：与所有 entry 嵌入做点积 (cosine)
3. **类别过滤**：只保留 `network_bandwidth2 ∈ anomaly_classes` 的 entry
4. **specificity boost 重排**：`score / √(len(anomaly_classes))`
   - 例：被标 6 类的通用 entry 会被降权 √6≈2.45 倍
   - 防止 "Enable Prometheus" 这种通用条目永远排第一
5. 返回 top-5

实际 top-5：

| Rank | sim | tuple ID | classes | title |
|---|---|---|---|---|
| 1 | 0.868 | **ISS-10882** | network_bandwidth2 | Docker: ConfigNodeClient `TEndPoint down` |
| 2 | 0.846 | **ISS-8403** | network_bandwidth2 | Cluster insert with batchSize=10000 → `RatisRequestFailedException` |
| 3 | 0.844 | CFG-schema_region_consensus_protocol_class | network_bandwidth2 | Config |
| 4 | 0.843 | CFG-data_region_consensus_protocol_class | network_bandwidth2 | Config |
| 5 | 0.834 | CFG-config_node_consensus_protocol_class | network_bandwidth2 | Config |

注意 ISS-8403 是 GitHub 上一个已修复的真实 bug：

> 用户用 `new Tablet(..., 10000)` 批量插入时报 `RatisRequestFailedException: Ratis request failed`。
> 维护者回复：Ratis 每个 write request 限制 4MB，batchSize=10000 一个请求约 6MB，
> 修复方案是调大 `data_region_ratis_log_appender_buffer_size_max` 到 16MB。

### 步骤 4 — `build_prompt()` — 构建 LLM prompt

入口：[explanation_pipeline.py:153](explanation_pipeline.py#L153)

用固定模板（[PROMPT_TEMPLATE](explanation_pipeline.py#L102)）渲染：

```markdown
You are an Apache IoTDB site reliability engineer. A monitoring window has been
flagged by our classifier. Produce a concise root-cause analysis and action plan.

# Window Context
- Time range: ... → ...
- Log volume: 100 raw → 9 unique-template entries (compressed)
- Classifier prediction: **network_bandwidth2**
- Predicted-class confidence note: N/A (logits hook not wired in this scaffold)

# Top log templates (by within-window count)
[cluster_3 x38] ...
[cluster_5 x12] ... pendingEntries size: <*> bufferedEntries size: <*>
[cluster_9 x12] DataRegion[1] : accumulated ...
...

# Frequent entities in this window
data_regions (x9), nodes (x9), ...

# Retrieved expert knowledge (top-4 relevant tuples)
## Tuple 1: `ISS-10882` (sim=0.868)
- Title: Docker: ConfigNodeClient `The current node may have been down ...`
- Symptom: ...
- Root cause: ...
- Fix: ...
- Anomaly classes: network_bandwidth2
- Subsystems: Network, Election

## Tuple 2: `ISS-8403` (sim=0.846)
- Title: Cluster insert with batchSize=10000 throws RatisRequestFailedException
- Symptom: ...
- Root cause: Ratis 4MB per-request limit; batchSize=10000 → ~6MB request
- Fix: SET CONFIGURATION 'data_region_ratis_log_appender_buffer_size_max'='16777216'
- Anomaly classes: network_bandwidth2
- Subsystems: Network, Election

## Tuple 3-5: 配置参数 ...

# Output instructions
Write a report with these sections (markdown):
1. **Summary** — one sentence: what likely happened.
2. **Evidence alignment** — which 1-2 retrieved tuples best match; quote their `id`.
3. **Root cause hypothesis** — be explicit about the IoTDB subsystem.
4. **Recommended actions** — concrete commands or config changes, safest first.
5. **What to monitor next** — which metric/dashboard panel confirms recovery.

Keep total length under 400 words. Cite tuple IDs inline like `(see MON-3)`.
```

总长约 5.4KB。

### 步骤 5 — `call_llm()` — 生成

入口：[explanation_pipeline.py:173](explanation_pipeline.py#L173)

支持三个后端：
- `--backend stub` (默认) — 只写 prompt，不调用 LLM
- `--backend ollama --model qwen35-opus-27b:q4km` — 本地 ollama
- `--backend openai_compat --base-url <url> --model <name>` — 任意 OpenAI 兼容 API

本例用 ollama 的 qwen35-opus-27b:q4km，耗时 ~260s（27B 模型 + 长 prompt）。

LLM 输出：

```markdown
# IoTDB Root-Cause Analysis Report

## 1. Summary
The cluster is experiencing Ratis consensus buffer pressure, likely from large
batch inserts approaching or exceeding the default 4MB per-request limit, causing
backpressure in the LogDispatcher threads.

## 2. Evidence Alignment
**Tuple 2 (ISS-8403)** is the strongest match: the log patterns show LogDispatcher
tracking `pendingEntries` and `bufferedEntries` sizes (cluster_5), with memory
reporting (cluster_3 x38) and active consensus communication (cluster_4) —
consistent with Ratis handling large requests near capacity limits.

**Tuple 1 (ISS-10882)** provides secondary context: ...
```

(输出被 ollama 默认 num_predict 截断；下一版会加 max_tokens=1500)

---

## 3. 为什么这个解释"做对了"

| 维度 | 体现 |
|---|---|
| **跨数据源融合** | 用 GitHub 已修复 bug ISS-8403 (4MB Ratis 限制) 解释当前 window 的 `cluster_5` 模板 |
| **细粒度定位** | 不只是"network 问题"，而是 `Ratis consensus buffer + LogDispatcher` |
| **从压缩数据反推工作负载** | `cluster_5 x12` 高频出现被解读为"large batch insert"——和 ISS-8403 用户场景一致 |
| **可验证性** | 强制引用 tuple ID，运维人员能去 GitHub 查原始 issue |
| **避开关键词陷阱** | 日志里**没有** "network" / "bandwidth" 字样（这是 iotbench 数据的特点），LLM 仍能正确推理 |

---

## 4. 6 个 demo 窗口的整体质量

| 类 | top-1 evidence | 输出质量 |
|---|---|---|
| compaction | TS-HC-4 (print-iotdb-data-dir) | ✓ 合理 — 正确识别 cross-space compaction + DataRegion 累积 |
| export | CFG-last_cache_operation_on_load | ✗ 偏弱 — 日志与 export 类别相关性低 |
| flush | CFG-seq_memtable_flush_interval | ✓ 良 — 识别 seq_memtable flush 触发机制 + 给出 SQL |
| full_cpu | CFG-mpp_data_exchange_max_pool_size | ✓ 合理 — MPP 线程池耗尽假设 |
| full_memory | CFG-sort_buffer_size_in_bytes | ✓ 良 — DataRegion 累积 + memory 报告频率 |
| network_bandwidth2 | ISS-10882 | ★ 优 — Ratis consensus buffer pressure + 引用 ISS-8403 |

**5/6 类输出质量可用**。weak 的 `export` 类的限制源于 iotbench 数据本身（之前
压缩分析证据：6 类窗口的模板分布 cosine = 1.000），不是 pipeline 问题。

---

## 5. 当前限制 & 升级路径

| 限制 | 升级方案 | 工作量 |
|---|---|---|
| Ollama 输出被截断 | call_llm() 加 `options.num_predict=1500` | 1 行代码 |
| LLM 不知分类器的犹豫度 | **Logits hook** — forward() 返回 logits；prompt 里加 "top-2 类的概率差距 = 0.07" | 小 |
| LLM 不知"为什么是这条 log 而非那条" | **HGT attention hook** — forward() 返回 (window→log, log→entity) 边的注意力权重；prompt 里加 "top-3 注意力路径" | 中 |
| 没有"历史上类似案例怎么处理" | **Similar-cases retrieval** — 训完 HGT 后提取所有 train embeddings + Faiss 索引；测试时检索同类 train 窗口 | 大（依赖训完模型） |
| 没有 Verifier/Refiner | **Verifier** = LLM 二次调用，检查输出是否引用了 retrieved tuple ID + 与 predicted_class 一致；**Refiner** = 按 Verifier 反馈重生成 | 中 |
| iotbench 部分类别难分（export） | 数据层面问题（窗口划分=100 条日志 vs 异常区间无因果关系），不是 explanation 层的事 | — |

---

## 6. 如何复现本例

```bash
cd "./external_pipeline/08_explanation_generation"

# 1. 构建检索索引（如果还没建过）
python \
  build_retrieval_index.py --rebuild

# 2. 跑 demo
python \
  explanation_pipeline.py \
  --windows-json ./log_analyse_new/dataset_pre/iotbench_anomaly/compressed_v3/windows_anomaly_test_compressed.json \
  --n-per-class 1 \
  --backend ollama --model qwen35-opus-27b:q4km

# 3. 看输出
ls demo_pipeline_output/<class>/
cat demo_pipeline_output/network_bandwidth2/*_explanation.md
```

要换数据集，只需把 `--windows-json` 指向 TPC/TSBS 的 `compressed_v3` 即可——pipeline
对数据集无感。
