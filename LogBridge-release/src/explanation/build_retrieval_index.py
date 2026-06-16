#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a semantic retrieval index over the 76 extracted tuples.

Tuples live in 10_explanation_data/extracted_tuples/{04..10}_*.json (FAQ-style
tuples with title/symptom/root_cause/fix/anomaly_classes). The two
"categorized" files (02_status_codes, 03_config_parameters) are also pulled
in, normalised to the same searchable shape.

Embedding model: sentence-transformers (default: all-mpnet-base-v2, 768d).
Index storage: a single .pt holding the encoded matrix + metadata; cosine
similarity is brute-force (76 entries, <1ms — Faiss overkill).

Usage:
  python build_retrieval_index.py              # build the index
  python build_retrieval_index.py --query "WAL files accumulating, writes rejected"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

TUPLES_DIR = Path(__file__).resolve().parent / 'tuples'
INDEX_OUT = Path(__file__).resolve().parent / 'retrieval_index.pt'
# bert-base-uncased is already locally cached (3.3GB); sentence-transformers/all-mpnet
# download was unreliable in this env. Mean-pooled BERT [CLS]+tokens is adequate for
# a 76-entry retrieval corpus.
DEFAULT_MODEL = 'bert-base-uncased'


class BertMeanEncoder:
    """Lightweight wrapper: BERT with mean-pool over non-pad tokens, L2-normalized."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = 'cuda', max_len: int = 256):
        import os
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        from transformers import BertModel, BertTokenizer
        self.device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = BertModel.from_pretrained(model_name, local_files_only=True).to(self.device).eval()
        self.max_len = max_len
        self.embedding_dim = self.model.config.hidden_size
        self.model_name = model_name

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        out_chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_len,
                                 return_tensors='pt').to(self.device)
            hs = self.model(**enc).last_hidden_state  # (B, L, D)
            mask = enc['attention_mask'].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out_chunks.append(pooled.cpu().numpy())
            if show_progress and (i // batch_size) % 5 == 0:
                print(f'  encoded {min(i+batch_size, len(texts))}/{len(texts)}')
        return np.vstack(out_chunks).astype(np.float32)


def _load_faq_style(fp: Path) -> List[Dict[str, Any]]:
    """Load FAQ-style files (01, 04-10) that have a 'tuples' list."""
    with open(fp) as f:
        d = json.load(f)
    out = []
    src = d.get('source')
    src_str = src if isinstance(src, str) else (src[0] if src else fp.name)
    for t in d.get('tuples') or []:
        out.append({
            'id': t.get('id', ''),
            'origin_file': fp.name,
            'source': src_str,
            'doc_type': d.get('doc_type', ''),
            'title': t.get('title', ''),
            'symptom': t.get('symptom', ''),
            'root_cause': t.get('root_cause', ''),
            'fix': t.get('fix') or [],
            'config_params': t.get('config_params') or [],
            'anomaly_classes': t.get('anomaly_classes') or [],
            'iotdb_subsystems': t.get('iotdb_subsystems') or [],
            'version': t.get('version', ''),
            'tags': t.get('tags') or [],
            'issue_url': t.get('issue_url', ''),
        })
    return out


def _load_status_codes(fp: Path) -> List[Dict[str, Any]]:
    """02_status_codes_categorized.json. The authoritative per-code records live in
    by_anomaly_class -> {class: [code dict]}. key_anomaly_diagnostic_codes is a curated
    {class: [code_number]} list flagging the highest-signal codes — use it to add a
    'smoking_gun' tag without duplicating entries."""
    with open(fp) as f:
        d = json.load(f)
    out: List[Dict[str, Any]] = []
    seen = set()
    # Build smoking-gun set from the curated list (values may be ints or code strings)
    smoking_gun = set()
    for class_codes in (d.get('key_anomaly_diagnostic_codes') or {}).values():
        for code in class_codes or []:
            smoking_gun.add(str(code))
    # Iterate by_anomaly_class for full code dicts
    for cls, codes in (d.get('by_anomaly_class') or {}).items():
        for c in codes or []:
            if not isinstance(c, dict):
                continue
            code = c.get('code')
            if code is None: continue
            code_key = str(code)
            if code_key in seen: continue
            seen.add(code_key)
            anom = c.get('anomaly_classes') or [cls]
            tags = ['status code', code_key] + (c.get('tags') or [])
            if code_key in smoking_gun:
                tags.append('smoking_gun')
            out.append({
                'id': f"SC-{code_key}",
                'origin_file': fp.name,
                'source': d.get('source', ''),
                'doc_type': d.get('doc_type', ''),
                'title': f"Status code {code_key}: {c.get('name','')}",
                'symptom': c.get('meaning', ''),
                'root_cause': c.get('common_root_cause', ''),
                'fix': c.get('fix_hint') or c.get('typical_action') or [],
                'config_params': c.get('related_config') or [],
                'anomaly_classes': anom,
                'iotdb_subsystems': c.get('iotdb_subsystems') or [],
                'version': c.get('version', 'all'),
                'tags': tags,
                'issue_url': '',
            })
    return out


def _load_config_params(fp: Path) -> List[Dict[str, Any]]:
    """03_config_parameters.json: parameters: [{name, description, type, default, effective,
    anomaly_classes, iotdb_subsystems}]. Skip parameters with empty anomaly_classes to keep
    the index focused (226 total → ~100 relevant)."""
    with open(fp) as f:
        d = json.load(f)
    out = []
    for p in d.get('parameters') or []:
        name = p.get('name') or ''
        if not name: continue
        anom = p.get('anomaly_classes') or []
        if not anom:
            # Skip params not tagged to any anomaly class — they're noise for retrieval
            continue
        eff = p.get('effective', '')
        default = p.get('default', '')
        out.append({
            'id': f"CFG-{name}",
            'origin_file': fp.name,
            'source': d.get('source', ''),
            'doc_type': d.get('doc_type', ''),
            'title': f"Config: {name}",
            'symptom': '',  # config params don't have direct symptoms; relevance is via description
            'root_cause': p.get('description', ''),
            'fix': [f"Default: {default}; Effective: {eff}; Tune via SET CONFIGURATION '{name}'='<value>' if hot-reloadable, else iotdb-system.properties + restart"],
            'config_params': [name],
            'anomaly_classes': anom,
            'iotdb_subsystems': p.get('iotdb_subsystems') or [],
            'version': 'all',
            'tags': ['config', p.get('type', '')],
            'issue_url': '',
        })
    return out


def load_all_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for fp in sorted(TUPLES_DIR.glob('*.json')):
        name = fp.name
        if name.startswith('02_status'):
            entries.extend(_load_status_codes(fp))
        elif name.startswith('03_config'):
            entries.extend(_load_config_params(fp))
        else:
            entries.extend(_load_faq_style(fp))
    return entries


def entry_to_text(e: Dict[str, Any]) -> str:
    """Concatenate searchable fields into a single passage for embedding."""
    parts = [
        f"Title: {e['title']}",
        f"Symptom: {e['symptom']}",
        f"Root cause: {e['root_cause']}",
    ]
    if e.get('fix'):
        fix_str = ' | '.join(e['fix']) if isinstance(e['fix'], list) else str(e['fix'])
        parts.append(f"Fix: {fix_str}")
    if e.get('anomaly_classes'):
        parts.append(f"Anomaly classes: {', '.join(e['anomaly_classes'])}")
    if e.get('iotdb_subsystems'):
        parts.append(f"Subsystems: {', '.join(e['iotdb_subsystems'])}")
    if e.get('tags'):
        parts.append(f"Tags: {', '.join(map(str, e['tags']))}")
    return '\n'.join(parts)


def build_index(model_name: str = DEFAULT_MODEL, device: str = 'cuda', out_path: Path = INDEX_OUT) -> Dict[str, Any]:
    entries = load_all_entries()
    print(f'Loaded {len(entries)} entries from {TUPLES_DIR}')
    texts = [entry_to_text(e) for e in entries]
    print(f'Embedding model: {model_name}')
    encoder = BertMeanEncoder(model_name=model_name, device=device)
    embs = encoder.encode(texts, batch_size=16, show_progress=True)
    print(f'Embeddings shape: {embs.shape}')
    payload = {
        'model_name': model_name,
        'embedding_dim': int(embs.shape[1]),
        'entries': entries,
        'texts': texts,
        'embeddings': torch.from_numpy(embs).contiguous(),
    }
    torch.save(payload, str(out_path))
    print(f'Saved index to {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)')
    return payload


def load_index(path: Path = INDEX_OUT) -> Dict[str, Any]:
    return torch.load(str(path), map_location='cpu', weights_only=False)


# Cache encoder across retrieve() calls — loading BERT is slow
_encoder_cache: Dict[str, Any] = {}


def _get_encoder(model_name: str, device: str = 'cpu') -> BertMeanEncoder:
    # default cpu: retrieve() only encodes 1 query at a time → CPU is fast enough,
    # and avoids fighting other GPU users for memory.
    if model_name not in _encoder_cache:
        _encoder_cache[model_name] = BertMeanEncoder(model_name=model_name, device=device)
    return _encoder_cache[model_name]


def retrieve(query: str, index: Dict[str, Any], k: int = 5,
             filter_anomaly_class: str | None = None,
             specificity_boost: bool = True) -> List[Tuple[float, Dict[str, Any]]]:
    """Return top-k (score, entry) pairs by cosine similarity.
    If filter_anomaly_class is given, only consider entries that include it.

    specificity_boost: when class filter is on, penalize entries that match too many
    anomaly classes (they're generic — e.g. a tuple tagged with all 6 classes adds
    no signal beyond "this is an IoTDB issue"). Effective score = sim / sqrt(n_classes).
    """
    encoder = _get_encoder(index['model_name'])
    q_emb = encoder.encode([query])[0]
    q = torch.from_numpy(q_emb)
    embs = index['embeddings']  # (N, D) already normalized
    sims = (embs @ q).cpu().numpy()  # (N,)
    # Build effective score with specificity boost when filtering
    if filter_anomaly_class and specificity_boost:
        eff = np.zeros_like(sims)
        for i, e in enumerate(index['entries']):
            classes = e.get('anomaly_classes') or []
            if filter_anomaly_class not in classes:
                eff[i] = -1.0  # exclude
            else:
                n = max(len(classes), 1)
                eff[i] = sims[i] / np.sqrt(n)
        order = np.argsort(-eff)
    else:
        order = np.argsort(-sims)
    out = []
    for idx in order:
        e = index['entries'][int(idx)]
        if filter_anomaly_class and filter_anomaly_class not in (e.get('anomaly_classes') or []):
            continue
        out.append((float(sims[idx]), e))  # report raw cosine in result for interpretability
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='Force rebuild even if index exists')
    ap.add_argument('--query', type=str, default=None, help='Optional query — print top-5 matches')
    ap.add_argument('--filter-class', type=str, default=None,
                    help='Optional: filter by anomaly class (compaction/export/flush/full_cpu/full_memory/network_bandwidth2)')
    ap.add_argument('--model', type=str, default=DEFAULT_MODEL)
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()

    if args.rebuild or not INDEX_OUT.exists():
        build_index(model_name=args.model)
    index = load_index()
    print(f'Index loaded: {len(index["entries"])} entries, model={index["model_name"]}, dim={index["embedding_dim"]}')

    if args.query:
        print(f'\nQuery: {args.query!r}')
        if args.filter_class:
            print(f'Filter: anomaly_class == {args.filter_class!r}')
        results = retrieve(args.query, index, k=args.k, filter_anomaly_class=args.filter_class)
        print(f'\nTop {len(results)} matches:')
        for rank, (score, e) in enumerate(results, 1):
            cls = ','.join(e.get('anomaly_classes') or []) or '-'
            print(f'\n  [{rank}] score={score:.3f}  id={e["id"]}  classes={cls}')
            print(f'      title: {e["title"][:120]}')
            print(f'      symptom: {e["symptom"][:160]}')
            print(f'      root_cause: {e["root_cause"][:160]}')


if __name__ == '__main__':
    main()
