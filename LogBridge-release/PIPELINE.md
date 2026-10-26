# Pipeline

Run commands from the repository root. Below `DATA` is a new output location.
The raw windows JSON is a mapping from window ID to a window with `logs`,
`window_start`, `window_end`, and `anomaly_types`. Each log has `raw_line` and
`timestamp`. Supply `incident_id` / `event_id` per window or an explicit JSON
`--event-map` mapping window IDs to real fault-injection incidents. Do not infer
incident boundaries from class labels. Overlapping incidents are kept in the same
temporal block. At least three independent temporal blocks are needed.

## 1. Split before rule sampling

```bash
python src/pipeline/prepare_data.py --windows-json raw_windows.json \
  --event-map incident_map.json --dataset iotbench --split-only --output-dir split_raw
python src/pipeline/induce_rules.py prepare \
  --train-json split_raw/windows_anomaly_train.json --output candidates.json
```

Only training logs fit Drain and supply template-stratified samples (up to five
lines per template). The preparation stage extracts parameter slots, fits character
1–4-gram TF-IDF, and applies DBSCAN with cosine distance. Singleton outliers receive
their own candidate so that rare values are not silently lost.

## 2. Induce, validate, review and freeze rules

```bash
python src/pipeline/induce_rules.py generate --candidates candidates.json --output-dir rules_draft
python src/pipeline/induce_rules.py validate --rules rules_draft/draft_rules.json \
  --validation-json split_raw/windows_anomaly_val.json --output validation_matches.json
# Validate draft regex/identity rules on held-out validation logs; correct errors.
python src/pipeline/induce_rules.py freeze --reviewed-rules reviewed_rules.json \
  --validation-record validation_record.json --output frozen_rules.json
```

Generation calls the configured model once per candidate cluster and saves prompts,
outputs and usage. A validation record must identify the reviewer and validation
source. It is a real external review artifact; it is not generated or invented by
the freeze command. Rules contain `type`, `regex` and `identity_fields`.
`regex` may name the extracted value group `value`. Identity context fields include
`directory`, `node_id`, `host`, `database`. Missing required context receives a
window-specific identity rather than creating unsupported cross-window equality.

## 3. Build compressed data and contexts

```bash
DATA=datasets/iotbench
python src/pipeline/prepare_data.py --windows-json raw_windows.json \
  --event-map incident_map.json --rules frozen_rules.json --dataset iotbench \
  --history-limit 5 --output-dir "$DATA"
python src/pipeline/precompute_templates.py --kg-file "$DATA/kg_iotbench.json" \
  --output-file "$DATA/bert_embeddings.pt" --device cpu
```

Preprocessing writes the fixed split manifest, original-to-compressed log indices,
typed entities, rule-file hash, KG, lightweight index and chunked subgraphs.
Only timestamps and severity are removed before parsing. Full original log text is
preserved for reports. Entities with unseen or low-frequency IDs remain present.
Runs merge only adjacent records with equal template and entity assignments.

History uses direct positive entity overlap, ranked by overlap count. Equal scores
use recency then window ID for deterministic ordering. A candidate must belong to
the target split and satisfy `end(candidate) < start(target)`. No recursive expansion,
future context, cross-split context, or category-based neighbor selection is allowed.
The configured history limit excludes the target window itself.

## 4. Train and export diagnosis

```bash
bash scripts/run_iotbench.sh train
python src/training/export_diagnostics.py --data-root "$DATA" \
  --checkpoint "$DATA/checkpoints/best_model.pt" --output "$DATA/diagnostics_test.json"
```

Repeat training with independently recorded seeds when computing mean results. The exported
diagnostics contain six-class probabilities, target-log attention, actual HGT
edge attention, selected source windows and full log evidence. Gold categories
are excluded. Historical records carry source window, time and shared entity IDs.

## 5. Build knowledge index, generate and judge

```bash
python src/explanation/build_retrieval_index.py --rebuild
bash scripts/run_explanation_demo.sh iotbench dry-run
bash scripts/run_explanation_demo.sh iotbench generate
bash scripts/run_explanation_demo.sh iotbench judge
```

Dry-run prompts use a separate output directory. Each live window gets one
generation request and its own report. No report fingerprint batching is supported.
The retrieval query uses prediction as a soft prior, with component/symptom and
parameter applicability gates, and returns up to four entries or none.
Schema/reference failures are recorded, not silently counted as successful reports.

Automatic judging is explicitly marked as automatic. It atomizes asserted facts,
checks them against logs, and checks operations and conditions against the same
full frozen KB. It distinguishes unsupported complete advice from concrete
unsupported technical assertions. Actual usage and responses are saved.

