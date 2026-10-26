# LogBridge

Entity-linked log anomaly diagnosis and evidence-grounded operational reports.

The supported path uses validated contextual extraction rules, event-disjoint
chronological splits, consecutive template/entity run compression, up to five
strictly earlier anomalous windows in the same split, count-feature local encoding,
two-layer log/entity HGT, target-log attention pooling, and a 256/128-hidden MLP.
Training uses unweighted cross-entropy. Window membership nodes do not participate
in HGT propagation. Fault-category labels supervise the target only; they do not
rank or select history. Training, validation and test each use their own past context.

See [PIPELINE.md](PIPELINE.md) for runnable commands and required input contracts.
See [METHOD.md](docs/METHOD.md) for the model and evidence flow.

## Install and verify

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Datasets, injection incident mappings, validated induced rules and trained model
weights are supplied separately. Use a new output directory for preprocessing.
Preprocessing artifacts and checkpoints must use the same context schema and model configuration.

## Models

`src/llm_config.py` configures `gpt-5.5` for offline rule induction and report
generation, and `deepseek-v4-pro` for automatic semantic judging. Override with
`LOGBRIDGE_RULE_MODEL`, `LOGBRIDGE_REPORT_MODEL`, `LOGBRIDGE_JUDGE_MODEL`.
Generation uses `OPENAI_API_KEY` / `OPENAI_BASE_URL`; judging uses
`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`. Actual model, prompt and token usage
are saved for each API response. Access to the configured models requires suitable API credentials.

Report generation requires exported classifier predictions and their own complete
evidence. It never uses a gold label as a prediction, hard-filters the KB by class,
or shares a whole report across different windows. Knowledge embeddings are reused.
No suitable knowledge means concrete verification checks without unsupported fixes.

## Knowledge inventory

The bundled corpus contains **320 uniquely identified entries**, including
226 configuration records, 18 status codes, 62 FAQ/manual/guide entries and
14 issue records. See [CORPUS.md](docs/CORPUS.md) for source locations and schema.
