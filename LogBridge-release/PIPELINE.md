# 数据流水线（7 步）

每个数据集（tpc / tsbs / iotbench）走一遍这 7 步，最终得到能喂 train.py 的子图。

```
原始 windows JSON (~100 logs/window)
    │
    │ [01_filter_anomaly]  → 只保留 6 类异常窗口，按 8:1:1 stratified split
    ▼
windows_anomaly_{train,val,test}.json
    │
    │ [02_compress_windows]  → Drain3 模板聚合，每窗 ~8 条 (带 count)
    │ [03_clean_entities_v3]  → 干净实体抽取（DataRegion / Thread / GeneralEntity）
    ▼
windows_anomaly_*_compressed.json  (符号链接到 datasets/<name>/)
    │
    │ [04_build_kg]  → 全量 KG（Window / LogInstance / Entity 三类节点）
    ▼
kg_<name>_v3.json   (≈300MB-1GB)
    │
    │ [05_precompute_bert]  → BERT-base-uncased 每个 raw_line 一个 768d 向量
    ▼
bert_embeddings_v3.pt  (~1GB)
    │
    │ [06_build_lightweight_index]  → 每窗口 top-k 候选邻居 + entity 索引
    ▼
lightweight_index_v3.pkl  (~170MB)
    │
    │ [07_build_subgraphs_sameclass]  → 多窗口子图（target + 4 邻居）
    │   - train: 同类邻居（label-aware augmentation）
    │   - val/test: entity-similarity 邻居（推理时会被 strip 成单窗口）
    ▼
prebuilt_subgraphs_k5_sameclass/
    ├── chunks_train/  (chunk_*.pt)
    ├── chunks_val/
    └── chunks_test/
```

下面对每步给出参数 / 时间 / 关键说明。

---

## 01. filter_anomaly

[`src/pipeline/01_filter_anomaly.py`](src/pipeline/01_filter_anomaly.py)

- 输入：原始 windows JSON（如 iotbench 的 `windows_count_100logs.json`）
- 输出：`windows_anomaly_{train,val,test}.json`
- 关键参数：`ANOMALY_LABELS = {compaction, export, flush, full_cpu, full_memory, network_bandwidth2}`
- 切分：每类内 random shuffle (seed=42) → 8/1/1
- 时间：iotbench ~5min（从 6.2GB 流式过滤）

⚠ **TPC/TSBS 早就过滤好了**，这步只为新数据集（iotbench）写过。

## 02. compress_windows

[`src/pipeline/02_compress_windows.py`](src/pipeline/02_compress_windows.py)

- 输入：上一步的 `windows_anomaly_*.json`
- 输出：`windows_anomaly_*_compressed.json`
- 关键：Drain3 增量解析，每条 log 得到 `template_id` + 在窗口内的 `count`（频率）
- 保留字段：`template_id, template_text, raw_line, count, first_ts, last_ts, entities`
- 时间：iotbench train ~30s

## 03. clean_entities_v3

[`src/pipeline/03_clean_entities_v3.py`](src/pipeline/03_clean_entities_v3.py)

- 干净实体抽取：从 raw_line 提取 DataRegion / Thread / GeneralEntity，去掉数字噪声
- 用 train 集的 entity 覆盖率作为筛选（min_cov=2）
- 输出：原 JSON 就地添加 `entities` 字段

## 04. build_kg

[`src/pipeline/04_build_kg.py`](src/pipeline/04_build_kg.py)

- 输入：所有 compressed JSON 的 union（含 normal 窗口作上下文）
- 输出：`kg_<name>_v3.json`
- 节点类型：
  - `Window:<window_key>`
  - `LogInstance:<window_key>_<log_idx>`
  - `<EntityType>:<id>`
- 边类型：
  - `Window -[CONTAINS]→ LogInstance`
  - `LogInstance -[ASSOCIATED_WITH]→ Entity`
- 时间：iotbench ~10min（含 entity 抽取）

## 05. precompute_bert

[`src/pipeline/05_precompute_bert.py`](src/pipeline/05_precompute_bert.py)

- 用 `bert-base-uncased`（本地缓存，TRANSFORMERS_OFFLINE=1）
- 输入：KG 中所有 LogInstance 的 raw_line（去重）
- 输出：`bert_embeddings_v3.pt`，dict `{raw_line: 768d tensor}`
- 时间：~2h（一次性；后续训练查表，0 BERT 调用）

## 06. build_lightweight_index

[`src/pipeline/06_build_lightweight_index.py`](src/pipeline/06_build_lightweight_index.py)

- 输入：KG
- 输出：`lightweight_index_v3.pkl`，含：
  - `window_entity_sets`, `entity_to_windows`
  - `window_to_logs`, `log_to_entities`, `log_to_text`
  - `window_top_neighbors`（按 entity 相似度预算 top-50 候选）
  - `window_to_label`
- 关键参数：`--top-k-candidates 50`, `--max-entity-cov` 过滤全局热点实体
- 时间：iotbench ~3min

## 07. build_subgraphs_sameclass

[`src/pipeline/07_build_subgraphs_sameclass.py`](src/pipeline/07_build_subgraphs_sameclass.py)

- 输入：lightweight index + 各 split 的窗口列表
- 输出：`prebuilt_subgraphs_k5_sameclass/{chunks_train,chunks_val,chunks_test}/`
- 每个 target window 的子图含：
  - 1 target + (max_neighbors-1) 邻居窗口
  - 这些窗口的所有 LogInstance + 它们关联的 Entity
  - **`log_counts: Dict[log_id, int]`**（从 compressed JSON 直接读，供 count-aware LT 用）
- 邻居策略：
  - train：**同类邻居**（label-aware）—— `--sameclass-splits train`
  - val/test：entity-similarity（无标签泄露）
  - 当前用户后续会把 val/test 改成单窗口推理；此处仍保留多窗口子图，推理时 strip。
- 关键参数：`--max-neighbors 5 --num-hops 1 --num-workers 8 --chunk-size 1000`
- 时间：iotbench ~4min（已 chunk-shuffle 修复防止 dataloader 按类分桶）

---

## 训练 / 评估

| 脚本 | 作用 |
|---|---|
| [`src/training/train.py`](src/training/train.py) | 主训练（30 epoch，单窗口推理 by default） |
| [`src/training/eval_single_window.py`](src/training/eval_single_window.py) | 用已训好的 ckpt 对比 multi-window vs single-window |
| [`src/training/train_lt_only.py`](src/training/train_lt_only.py) | LT-only 消融（无 HGT 无实体；count-aware 展开 raw_line 序列） |

## 解释生成

| 脚本 | 作用 |
|---|---|
| [`src/explanation/build_retrieval_index.py`](src/explanation/build_retrieval_index.py) | 214 entries → BERT mean-pool → cosine 索引 |
| [`src/explanation/explanation_pipeline.py`](src/explanation/explanation_pipeline.py) | 测试窗口 → 检索 → LLM Generator |
| [`src/explanation/tuples/`](src/explanation/tuples/) | 10 个 JSON 知识库文件 |

详见 [docs/METHOD.md](docs/METHOD.md) 和 [docs/CORPUS.md](docs/CORPUS.md)。
