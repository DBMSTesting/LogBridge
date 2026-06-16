# LogBridge

**Entity-Bridged Heterogeneous Graphs for Long-Horizon Log Anomaly Diagnosis on Apache IoTDB.**

LogBridge is a four-stage framework that turns the raw log stream of a
time-series database into a predicted anomaly category together with an
evidence-grounded, citation-backed diagnostic report. It targets the
six injected anomaly categories of Apache IoTDB
(*compaction*, *export*, *flush*, *full_cpu*, *full_memory*, *network_bandwidth2*).

> This repository contains the source code, scripts, and supporting
> documents of LogBridge. The training datasets and the baseline
> implementations are **not** included; see *Datasets* below.

## Highlights

- **LLM-induced entity rules** (Stage A) — a language model induces a small set
  of typed extraction rules from a template-stratified sample of the corpus;
  at runtime only the deterministic rules are executed, preserving the
  per-window discriminative signal that template-only parsing would erase.
- **Entity-bridged heterogeneous graph** (Stage B) — windows, log instances,
  and typed entities are organised into one heterogeneous graph; entities
  shared by several windows become bridges along which long-horizon
  context can flow.
- **Subgraph-based diagnosis** (Stage C) — a count-aware Local Transformer
  followed by a Heterogeneous Graph Transformer reads out the target
  window's category together with three diagnostic signals
  (top-K class logits, attention weights, top-similar training windows).
- **Evidence-grounded report generation** (Stage D) — Stage C's signals are
  paired with a 214-entry curated IoTDB knowledge base; the language model
  produces a report whose every operational recommendation is back-cited
  to a specific knowledge entry.

## Headline results (Macro-F1)

| Dataset   | Macro-P | Macro-R | Macro-F1 |
|-----------|--------:|--------:|---------:|
| TPC       |  95.21% |  93.06% | **93.93%** |
| IoTBench  |  89.18% |  89.18% | **89.05%** |
| TSBS      |  93.75% |  94.79% | **94.13%** |

See [RESULTS.md](RESULTS.md) for full per-class and ablation tables.

## Repository layout

```
LogBridge/
├── README.md                  # this file
├── PIPELINE.md                # 7-step data pipeline
├── RESULTS.md                 # per-dataset accuracy + ablation tables
├── LICENSE                    # MIT
├── requirements.txt
│
├── src/
│   ├── pipeline/              # 01_filter → 07_subgraphs (preprocessing)
│   │   └── entity_extraction/ # LLM-induced typed entity rules
│   ├── model/                 # template encoder, Local Transformer, HGT, classifier
│   ├── training/              # train.py, eval_single_window.py, train_lt_only.py
│   ├── explanation/           # retrieval + LLM report generation
│   │   └── tuples/            # 214 curated knowledge-base entries
│   └── utils/                 # data loader, knowledge graph loader
│
├── scripts/                   # one-command bash scripts (per dataset)
└── docs/                      # METHOD.md, CORPUS.md, explanation_readme.md
```

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch and `torch-geometric` may need a CUDA-matched install; follow the
[official PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if the default wheel does not match your CUDA version.

### 2. Prepare datasets

Place each preprocessed dataset under `datasets/<name>/`, where `<name>` is one
of `tpc`, `iotbench`, `tsbs`. Each directory should contain:

```
datasets/<name>/
├── kg_<name>_v3.json                # knowledge graph (output of pipeline step 04)
├── bert_embeddings_v3.pt            # precomputed BERT embeddings (step 05)
├── lightweight_index_v3.pkl         # entity-overlap index (step 06)
├── prebuilt_subgraphs_k5_sameclass/ # one-hop subgraphs per window (step 07)
└── windows_anomaly_{train,val,test}_compressed.json
```

The `src/pipeline/01_filter_anomaly.py` → `07_build_subgraphs_sameclass.py`
scripts produce these artefacts from raw IoTDB logs; see [PIPELINE.md](PIPELINE.md)
for the full pre-processing flow. We do not redistribute the IoTDB log
datasets here; see *Datasets* below.

### 3. Train and evaluate

```bash
# Train + classify (default: multi-window training, single-window inference)
bash scripts/run_iotbench.sh train

# Single-window inference on the test set
bash scripts/run_iotbench.sh eval_single_window

# Local-Transformer-only ablation
bash scripts/run_iotbench.sh lt_only

# No-count-aware ablation
bash scripts/run_no_count_ablation.sh iotbench
```

Replace `iotbench` with `tpc` or `tsbs` for the other datasets.

### 4. Explanation generation demo

```bash
# Six categories × 1 window each, full Stage-D pipeline
bash scripts/run_explanation_demo.sh iotbench stub

# To use a real LLM backend (Ollama or any OpenAI-compatible endpoint):
OPENAI_API_KEY=... bash scripts/run_explanation_demo.sh iotbench ollama
```

See [docs/METHOD.md](docs/METHOD.md) for the full Stage-D method specification
and [docs/CORPUS.md](docs/CORPUS.md) for how the 214-entry knowledge base
was built.

## Asymmetric training and inference

LogBridge trains with **multi-window subgraphs** (`max_neighbors=5`,
`num_hops=1`) as regularisation, but uses **single-window inference** at
test time to avoid noise injected by entity-similarity-based neighbour
selection on unlabelled windows.

This is implemented in [`src/training/train.py`](src/training/train.py) via
`--inference-mode single_window`; `val/test` subgraphs are converted on
the fly through `subgraph_to_single_window()`, so no separate val/test
subgraph index needs to be built.

## Core design points

1. **Drain3 templating + entity-equality compression** — each 100-line raw
   window is compressed to roughly 8–30 log nodes after merging consecutive
   lines that share both the same template and the same entity assignments
   (see Stage C, §2 of the paper).
2. **Count-aware Local Transformer** — after compression, each log node
   carries an occurrence count `c_l`; the Local Transformer virtually
   expands each instance to `c_l` copies before contextualising and then
   pools back, recovering the frequency signal stripped by compression.
3. **Entity-bridged heterogeneous graph** — three node types
   (`window`, `log_instance`, `entity`) and two edge types (`CONTAINS`,
   `ASSOCIATED_WITH`); shared entity nodes connect windows that would
   otherwise be disjoint.
4. **HGT message passing on a one-hop subgraph** — the target window is
   classified from the read-out of an attention-pooled subgraph that
   reaches its long-horizon neighbours along entity bridges.
5. **Citation-grounded report generation** — Stage D retrieves the
   most relevant entries from a 214-entry curated IoTDB knowledge base
   (BERT + a specificity-adjusted re-ranker) and asks the language model
   to back-cite every recommendation to a specific entry, making citation
   validity and entity-level hallucination automatically auditable.

## Datasets

The three IoTDB log datasets (TPC, IoTBench, TSBS) used in the paper were
generated by running their respective benchmark workloads against Apache
IoTDB in a four-node Docker cluster with anomaly injection
(workload-driven for `compaction` / `export` / `flush`; Chaos Mesh for the
`full_*` and `network_bandwidth2` resource anomalies). The raw and
preprocessed datasets are not redistributed in this repository; please
contact the authors for access, or regenerate them through the workflow
described in [PIPELINE.md](PIPELINE.md).

## Citation

```bibtex
@inproceedings{logbridge2027,
  title  = {{LogBridge}: Entity-Bridged Heterogeneous Graphs for
           Long-Horizon Log Anomaly Diagnosis on Apache IoTDB},
  author = {Anonymous},
  booktitle = {Proc. International Conference on Data Engineering (ICDE)},
  year   = {2027}
}
```

## License

Released under the [MIT License](LICENSE).
