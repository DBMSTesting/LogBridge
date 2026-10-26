# Knowledge corpus

The corpus is stored in `src/explanation/tuples/` and contains 320 records:

| Category | Records |
|---|---:|
| Configuration parameters | 226 |
| Status codes | 18 |
| FAQ, user manual and operational guides | 62 |
| Resolved issue records | 14 |

Each source JSON records its documentation source. Entries expose a title,
symptom, documented cause, recommended checks or fixes, configuration parameters,
subsystems and applicable anomaly categories. Issue entries also retain issue URLs.
Repeated parameter names with different scopes receive distinct entry identifiers.

Build the retrieval index with:

```bash
python src/explanation/build_retrieval_index.py --rebuild
```

The encoder mean-pools non-padding BERT token embeddings and normalizes them.
Retrieval checks components, symptoms and parameter evidence, then ranks eligible
entries with the predicted category as a soft prior. It returns at most four
entries and may return none. Full knowledge text is used for semantic evaluation.
