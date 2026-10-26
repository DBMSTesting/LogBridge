# Method implementation

LogBridge combines contextual entity extraction, bounded historical evidence,
count-aware local encoding, heterogeneous graph aggregation and report generation.
Commands and input formats are described in [PIPELINE.md](../PIPELINE.md).

| Paper section | Implementation |
|---|---|
| 3.1 Sampling, clustering, induction | `src/pipeline/induce_rules.py` |
| 3.1 Contextual extraction and identity | `src/pipeline/prepare_data.py:RuleExtractor` |
| 3.2 Compression and historical selection | `src/pipeline/prepare_data.py` |
| 3.2 Count encoding | `src/model/local_transformer.py` |
| 3.2 HGT and target pooling | `src/model/hgt_model.py`, `hgt_anomaly_model.py` |
| 3.2 MLP classification and training | `src/model/classifier.py`, `src/training/train.py` |
| 3.3 Signal/evidence export | `src/training/export_diagnostics.py` |
| 3.3 Applicable KB retrieval and reports | `src/explanation/grounded_reports.py` |
| LLM roles and audited requests | `src/llm_config.py`, `src/llm_client.py` |

