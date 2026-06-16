#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explanation generation pipeline (scaffold).

Pipeline:
  test window  -->  (predicted_class, top-K templates)  -->  retrieve(query, class) -->  Generator LLM
       |                                                                                    |
       |                                                                                    v
       +-----------------------------------------------------------> [Verifier: stub] --> [Refiner: stub]

Inputs (this scaffold version):
  - A compressed_v3 windows JSON
  - For each demo window, use ground-truth anomaly_types[0] as "predicted_class"
    (real predictions plug in later via --predictions <file>)

Outputs:
  demo_output/<class>/<wid>_evidence.json  — retrieved tuples + window context
  demo_output/<class>/<wid>_prompt.md       — full Generator prompt
  demo_output/<class>/<wid>_explanation.md  — LLM output (or [STUB] if no backend)

LLM backends:
  --backend stub    (default; just write the prompt, no LLM call)
  --backend ollama  (POST localhost:11434/api/generate)
  --backend openai_compat --base-url <url> --model <name>  (any OpenAI-compatible)

Usage:
  python explanation_pipeline.py \
      --windows-json /path/to/windows_anomaly_test_compressed.json \
      --n-per-class 2 --backend stub
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from build_retrieval_index import load_index, retrieve, INDEX_OUT

ANOMALY_CLASSES = ['compaction', 'export', 'flush', 'full_cpu', 'full_memory', 'network_bandwidth2']


# ----------------------------- window context ---------------------------------

def extract_window_context(window_data: dict, top_k_templates: int = 5) -> Dict[str, Any]:
    """Extract the salient features of a window:
    - top-K templates by 'count' (the within-window frequency)
    - all entities seen
    - basic timing / file metadata
    """
    logs = window_data.get('logs') or []
    # rank by count
    logs_sorted = sorted(logs, key=lambda r: -int(r.get('count', 1)))
    top = logs_sorted[:top_k_templates]
    template_lines = []
    for r in top:
        cnt = int(r.get('count', 1))
        tid = r.get('template_id', '')
        text = (r.get('template_text') or r.get('raw_line') or '')[:200]
        template_lines.append(f'  [{tid} x{cnt}] {text}')
    entities = []
    for r in logs:
        for e in r.get('entities', []) or []:
            ev = e if isinstance(e, str) else (e.get('value') or e.get('name') or '')
            if ev:
                entities.append(str(ev))
    entity_counts = Counter(entities).most_common(8)
    return {
        'label_file': window_data.get('label_file', ''),
        'window_start': window_data.get('window_start', ''),
        'window_end': window_data.get('window_end', ''),
        'log_count_original': window_data.get('log_count_original', 0),
        'log_count': window_data.get('log_count', 0),
        'anomaly_types': window_data.get('anomaly_types') or [],
        'top_templates_str': '\n'.join(template_lines),
        'top_templates_raw': [
            {'template_id': r.get('template_id'),
             'count': int(r.get('count', 1)),
             'text': (r.get('template_text') or r.get('raw_line') or '')[:300]}
            for r in top
        ],
        'top_entities': [{'entity': k, 'count': v} for k, v in entity_counts],
    }


def build_retrieval_query(context: Dict[str, Any], predicted_class: str) -> str:
    """Make a single text query for the retrieval index.
    Concatenates predicted class + top template texts + top entities."""
    parts = [f'Predicted anomaly: {predicted_class}']
    parts.append('Top log templates by count:')
    parts.append(context['top_templates_str'])
    if context.get('top_entities'):
        ents = ', '.join([f'{e["entity"]}(x{e["count"]})' for e in context['top_entities'][:5]])
        parts.append(f'Frequent entities: {ents}')
    return '\n'.join(parts)


# ----------------------------- prompt building --------------------------------

PROMPT_TEMPLATE = """You are an Apache IoTDB site reliability engineer. A monitoring window has been
flagged by our classifier. Produce a concise root-cause analysis and action plan.

# Window Context
- Time range: {window_start} → {window_end}
- Log volume: {log_count_original} raw → {log_count} unique-template entries (compressed)
- Classifier prediction: **{predicted_class}**
- Predicted-class confidence note: {confidence_note}

# Top log templates (by within-window count)
{top_templates_str}

# Frequent entities in this window
{entities_str}

# Retrieved expert knowledge (top-{k} relevant tuples)
The classifier's prediction was used as a filter; these tuples come from IoTDB
docs / status codes / config reference / closed GitHub issues.

{evidence_block}

# Output instructions
Write a report with these sections (markdown):

1. **Summary** — one sentence: what likely happened.
2. **Evidence alignment** — which 1-2 retrieved tuples best match the window's symptoms; quote their `id`.
3. **Root cause hypothesis** — be explicit about the IoTDB subsystem (e.g. Compaction / Flush / WAL).
4. **Recommended actions** — concrete commands or config changes, ordered safest first.
5. **What to monitor next** — which metric/dashboard panel confirms recovery.

Keep total length under 400 words. Cite tuple IDs inline like `(see MON-3)`.
"""


PROMPT_TEMPLATE_ACTIONS_ONLY = """You are an Apache IoTDB SRE. A window has been flagged.
Output **only** a concrete action plan (commands / config tweaks). Cite tuple IDs.

# Window
- Classifier prediction: **{predicted_class}**
- Top log templates (by count):
{top_templates_str}
- Frequent entities: {entities_str}

# Retrieved expert tuples (top-{k}, filtered by predicted class)
{evidence_block}

# Output
3-5 bullet points. Each: one concrete action + cited tuple id `(see ID)`.
Order safest first. ≤ 120 words total. No preamble.
"""


def render_evidence(evidence: List[Tuple[float, Dict[str, Any]]]) -> str:
    out = []
    for rank, (score, e) in enumerate(evidence, 1):
        fix_str = '\n      '.join((e.get('fix') or [])[:4]) or '(no explicit fix)'
        out.append(
            f'## Tuple {rank}: `{e["id"]}` (sim={score:.3f})\n'
            f'- Source: {e.get("source")}\n'
            f'- Title: {e.get("title")}\n'
            f'- Symptom: {e.get("symptom", "(none)")[:300]}\n'
            f'- Root cause: {e.get("root_cause", "(none)")[:300]}\n'
            f'- Fix:\n      {fix_str}\n'
            f'- Anomaly classes: {", ".join(e.get("anomaly_classes") or [])}\n'
            f'- Subsystems: {", ".join(e.get("iotdb_subsystems") or [])}\n'
        )
    return '\n'.join(out)


def build_prompt(context: Dict[str, Any], predicted_class: str,
                 evidence: List[Tuple[float, Dict[str, Any]]],
                 confidence_note: str = 'N/A (logits hook not wired in this scaffold)',
                 actions_only: bool = False) -> str:
    entities_str = ', '.join([f'{e["entity"]} (x{e["count"]})' for e in context.get('top_entities', [])]) or '(none)'
    template = PROMPT_TEMPLATE_ACTIONS_ONLY if actions_only else PROMPT_TEMPLATE
    kwargs = dict(
        window_start=context['window_start'],
        window_end=context['window_end'],
        log_count_original=context['log_count_original'],
        log_count=context['log_count'],
        predicted_class=predicted_class,
        confidence_note=confidence_note,
        top_templates_str=context['top_templates_str'],
        entities_str=entities_str,
        k=len(evidence),
        evidence_block=render_evidence(evidence),
    )
    # actions-only 模板没用到部分字段——直接 format 会跳过未用的，但同时确保不缺
    return template.format(**kwargs)


# ----------------------------- LLM backends -----------------------------------

def call_llm(prompt: str, backend: str = 'stub', model: Optional[str] = None,
             base_url: Optional[str] = None, timeout: int = 1500) -> str:
    """Dispatch to selected backend. Returns the model's textual response (or a stub marker)."""
    if backend == 'stub':
        return ('[STUB — no LLM call] To run for real, re-invoke with --backend ollama '
                f'or --backend openai_compat --base-url ... --model ...\nPrompt length: {len(prompt)} chars')

    if backend == 'ollama':
        import urllib.request, urllib.error
        model = model or 'qwen35-opus-27b:q4km'
        url = (base_url or 'http://localhost:11434').rstrip('/') + '/api/generate'
        # /no_think disables the verbose <think>...</think> trace on qwen3 family
        body = json.dumps({
            'model': model,
            'prompt': '/no_think\n' + prompt,
            'stream': False,
            'options': {'num_predict': 400, 'temperature': 0.3},
        }).encode()
        req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
                text = resp.get('response', '').strip() or '[empty response from ollama]'
                # Strip any residual <think>...</think> block if /no_think was ignored
                import re
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
                return text
        except urllib.error.URLError as e:
            return f'[ollama error: {e}]'

    if backend == 'openai_compat':
        import urllib.request, urllib.error
        if not base_url:
            return '[openai_compat error: --base-url required]'
        url = base_url.rstrip('/') + '/v1/chat/completions'
        model = model or 'moonshot-v1-32k'
        body = json.dumps({
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.3,
            'max_tokens': 800,
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY", "dummy")}',
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
                msg = resp['choices'][0]['message']
                content = (msg.get('content') or '').strip()
                # deepseek-v4-flash sometimes falls back to reasoning mode and returns
                # empty content with reasoning_content populated. Use reasoning as fallback.
                if not content:
                    reasoning = (msg.get('reasoning_content') or '').strip()
                    if reasoning:
                        return f'[reasoning fallback]\n{reasoning}'
                return content or '[empty content from openai_compat]'
        except urllib.error.URLError as e:
            return f'[openai_compat error: {e}]'
        except (KeyError, json.JSONDecodeError) as e:
            return f'[openai_compat parse error: {e}]'

    return f'[unknown backend: {backend}]'


# ----------------------------- demo orchestration -----------------------------

def select_demo_windows(windows: Dict[str, dict], n_per_class: int = 2,
                        classes: List[str] = ANOMALY_CLASSES, seed: int = 42) -> Dict[str, List[Tuple[str, dict]]]:
    """Pick n_per_class windows from each class. Deterministic via seed."""
    import random
    rng = random.Random(seed)
    by_class: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
    for wid, w in windows.items():
        atype = (w.get('anomaly_types') or [None])[0]
        if atype in classes:
            by_class[atype].append((wid, w))
    out: Dict[str, List[Tuple[str, dict]]] = {}
    for c in classes:
        lst = by_class.get(c, [])
        rng.shuffle(lst)
        out[c] = lst[:n_per_class]
    return out


def sample_random_windows(windows: Dict[str, dict], n: int,
                           classes: List[str] = ANOMALY_CLASSES,
                           seed: int = 42) -> Dict[str, List[Tuple[str, dict]]]:
    """Random sample N windows from ALL anomaly windows (not stratified by class).
    Returns {class: [(wid, w)]} keyed by ground-truth class."""
    import random
    rng = random.Random(seed)
    pool: List[Tuple[str, dict, str]] = []
    for wid, w in windows.items():
        atype = (w.get('anomaly_types') or [None])[0]
        if atype in classes:
            pool.append((wid, w, atype))
    rng.shuffle(pool)
    picked = pool[:n]
    out: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
    for wid, w, cls in picked:
        out[cls].append((wid, w))
    return dict(out)


def _short_wid(wid: str) -> str:
    return wid.replace(':', '_').replace('/', '_')[:80]


def _compute_fingerprint(predicted_class: str,
                          evidence: List[Tuple[float, Dict[str, Any]]],
                          n: int = 3) -> Tuple[str, Tuple[str, ...]]:
    """指纹 = (predicted_class, sorted top-n tuple IDs).

    predicted_class **必须** 是指纹的一部分——同 fingerprint 集群里所有窗口
    都属于同一个异常类，绝不跨类合并。
    """
    top_ids = tuple(sorted(e['id'] for _, e in evidence[:n]))
    return (predicted_class, top_ids)


def run_demo(windows_json: Path, n_per_class: int, top_k_templates: int, retrieve_k: int,
             backend: str, model: Optional[str], base_url: Optional[str],
             output_root: Path,
             sample_n: Optional[int] = None,
             actions_only: bool = False,
             cluster_by_fingerprint: bool = False,
             fingerprint_topn: int = 3) -> Dict[str, Any]:
    print(f'Loading windows from {windows_json} ...', flush=True)
    with open(windows_json) as f:
        windows = json.load(f)
    print(f'  loaded {len(windows)} windows')
    print('Loading retrieval index ...', flush=True)
    index = load_index(INDEX_OUT)
    print(f'  index: {len(index["entries"])} entries, dim={index["embedding_dim"]}')

    if sample_n is not None and sample_n > 0:
        selected = sample_random_windows(windows, n=sample_n)
        total = sum(len(v) for v in selected.values())
        print(f'  random sample n={sample_n}: actually got {total} windows')
        print(f'  per-class: {[(c, len(v)) for c, v in selected.items()]}')
    else:
        selected = select_demo_windows(windows, n_per_class=n_per_class)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    t0 = time.time()

    # =========================================================================
    # PASS 1: 对所有窗口做检索 + 构 prompt（不调 LLM）。同时写 evidence/prompt 文件。
    # 这一步对每窗口都执行，因为我们仍要保留 per-window evidence/prompt 作为审计入口。
    # =========================================================================
    window_records: List[Dict[str, Any]] = []  # 顺序保留
    print('\n--- PASS 1: retrieval + prompt build ---')
    n_total_planned = sum(len(v) for v in selected.values())
    n_done = 0
    for cls, items in selected.items():
        out_dir = output_root / cls
        out_dir.mkdir(exist_ok=True)
        for wid, wdata in items:
            short = _short_wid(wid)
            ctx = extract_window_context(wdata, top_k_templates=top_k_templates)
            query = build_retrieval_query(ctx, predicted_class=cls)
            evidence = retrieve(query, index, k=retrieve_k, filter_anomaly_class=cls)
            prompt = build_prompt(ctx, predicted_class=cls, evidence=evidence,
                                  actions_only=actions_only)
            evidence_summary = [{
                'rank': i + 1, 'score': float(s),
                'id': e['id'], 'title': e['title'],
                'anomaly_classes': e.get('anomaly_classes'),
                'source': e.get('source'),
            } for i, (s, e) in enumerate(evidence)]
            (out_dir / f'{short}_evidence.json').write_text(
                json.dumps({'window_id': wid, 'predicted_class': cls,
                            'context': {k: v for k, v in ctx.items() if k != 'top_templates_str'},
                            'evidence': evidence_summary}, indent=2, ensure_ascii=False)
            )
            (out_dir / f'{short}_prompt.md').write_text(prompt)
            top1_score = float(evidence[0][0]) if evidence else 0.0
            fingerprint = _compute_fingerprint(cls, evidence, n=fingerprint_topn) if cluster_by_fingerprint else None
            window_records.append({
                'wid': wid, 'short': short, 'cls': cls, 'out_dir': out_dir,
                'evidence': evidence, 'prompt': prompt,
                'top1_id': evidence[0][1]['id'] if evidence else None,
                'top1_score': top1_score,
                'fingerprint': fingerprint,
            })
            n_done += 1
            if n_done % 50 == 0 or n_done == n_total_planned:
                print(f'  retrieval {n_done}/{n_total_planned}', flush=True)
    print(f'  pass 1 done in {time.time()-t0:.1f}s')

    # =========================================================================
    # PASS 2: 调 LLM。两种模式：
    #   - cluster_by_fingerprint=False: 对每窗口调一次（原版行为）
    #   - cluster_by_fingerprint=True: 按指纹分桶，每桶选 top1_score 最高者为
    #                                  代表跑一次，再把结果复制给同桶成员
    # =========================================================================
    print('\n--- PASS 2: LLM generation ---')
    clusters: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}

    if cluster_by_fingerprint:
        # 分桶（predicted_class 已是指纹一部分 → 自动不跨类合并）
        buckets: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
        for rec in window_records:
            buckets[rec['fingerprint']].append(rec)
        print(f'  total windows: {len(window_records)} → clusters: {len(buckets)}'
              f'  (compression {len(window_records)/max(len(buckets),1):.2f}x)')
        # 按 class 聚合打印
        by_cls = defaultdict(int)
        for fp in buckets:
            by_cls[fp[0]] += 1
        print(f'  clusters per class: {dict(by_cls)}')

        # 给每个 cluster 一个稳定 ID（按代表 wid 排序）
        for ci, (fp, members) in enumerate(sorted(buckets.items(),
                                                   key=lambda kv: (-len(kv[1]), kv[0])), start=1):
            # 代表：top1_score 最高者
            members.sort(key=lambda r: -r['top1_score'])
            rep = members[0]
            cluster_id = f'c{ci:04d}_{rep["cls"]}_{(fp[1][0] if fp[1] else "none")[:30]}'
            t_llm = time.time()
            explanation = call_llm(rep['prompt'], backend=backend, model=model, base_url=base_url)
            elapsed = time.time() - t_llm
            clusters[fp] = {
                'cluster_id': cluster_id,
                'predicted_class': rep['cls'],
                'fingerprint_top_ids': list(fp[1]),
                'representative_wid': rep['wid'],
                'member_wids': [m['wid'] for m in members],
                'n_members': len(members),
                'explanation_chars': len(explanation),
                'llm_seconds': elapsed,
            }
            print(f'  cluster {cluster_id[:70]:70s} | n={len(members):>3} | top1={rep["top1_id"]} '
                  f'(sim={rep["top1_score"]:.2f}) | llm={elapsed:.1f}s', flush=True)
            # 复制 explanation 到所有成员窗口
            for m in members:
                (m['out_dir'] / f'{m["short"]}_explanation.md').write_text(explanation)
                summary.append({
                    'class': m['cls'], 'window_id': m['wid'],
                    'top_evidence_id': m['top1_id'],
                    'top_evidence_score': m['top1_score'],
                    'n_evidence': len(m['evidence']),
                    'prompt_chars': len(m['prompt']),
                    'explanation_chars': len(explanation),
                    'cluster_id': cluster_id,
                    'is_representative': (m['wid'] == rep['wid']),
                })
    else:
        # 原版：每窗口一次
        for rec in window_records:
            t_llm = time.time()
            explanation = call_llm(rec['prompt'], backend=backend, model=model, base_url=base_url)
            elapsed = time.time() - t_llm
            (rec['out_dir'] / f'{rec["short"]}_explanation.md').write_text(explanation)
            print(f'  - {rec["wid"][:70]:70s} | top-1 evidence: {rec["top1_id"]} '
                  f'(sim={rec["top1_score"]:.2f}) | llm={elapsed:.1f}s', flush=True)
            summary.append({
                'class': rec['cls'], 'window_id': rec['wid'],
                'top_evidence_id': rec['top1_id'],
                'top_evidence_score': rec['top1_score'],
                'n_evidence': len(rec['evidence']),
                'prompt_chars': len(rec['prompt']),
                'explanation_chars': len(explanation),
            })

    (output_root / '_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if cluster_by_fingerprint:
        clusters_out = {
            f'cluster_{i+1:04d}': {
                **c,
                'fingerprint': list(fp),
            }
            for i, (fp, c) in enumerate(clusters.items())
        }
        (output_root / '_clusters.json').write_text(
            json.dumps(clusters_out, indent=2, ensure_ascii=False)
        )
        n_calls = len(clusters)
        n_win = len(window_records)
        print(f'\nDone in {time.time()-t0:.1f}s. '
              f'LLM calls: {n_calls} (compressed {n_win}→{n_calls}, {n_win/max(n_calls,1):.2f}x)')
        print(f'  Summary:  {output_root}/_summary.json')
        print(f'  Clusters: {output_root}/_clusters.json')
    else:
        print(f'\nDone in {time.time()-t0:.1f}s. Summary: {output_root}/_summary.json')
    return {'n_windows': len(summary), 'output_root': str(output_root)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows-json', type=Path, required=True)
    ap.add_argument('--n-per-class', type=int, default=2)
    ap.add_argument('--top-k-templates', type=int, default=5)
    ap.add_argument('--retrieve-k', type=int, default=5)
    ap.add_argument('--backend', choices=['stub', 'ollama', 'openai_compat'], default='stub')
    ap.add_argument('--model', type=str, default=None,
                    help='ollama model name or openai_compat model id')
    ap.add_argument('--base-url', type=str, default=None,
                    help='base url for openai_compat (e.g. http://localhost:8000)')
    ap.add_argument('--output-root', type=Path,
                    default=THIS_DIR / 'demo_pipeline_output')
    ap.add_argument('--sample-n', type=int, default=None,
                    help='随机抽 N 个窗口（不分类）。设定后覆盖 --n-per-class')
    ap.add_argument('--actions-only', action='store_true', default=False,
                    help='只让 LLM 输出运维建议部分，大幅减少生成时间（适合大量窗口评测）')
    ap.add_argument('--cluster-by-fingerprint', action='store_true', default=False,
                    help='按 (predicted_class, top-N tuple IDs) 指纹分簇，同簇只调一次 LLM '
                         '（同 predicted_class 才合并，不跨类）')
    ap.add_argument('--fingerprint-topn', type=int, default=3,
                    help='指纹用 top-N tuple IDs（默认 3）')
    args = ap.parse_args()

    run_demo(
        windows_json=args.windows_json,
        n_per_class=args.n_per_class,
        top_k_templates=args.top_k_templates,
        retrieve_k=args.retrieve_k,
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        output_root=args.output_root,
        sample_n=args.sample_n,
        actions_only=args.actions_only,
        cluster_by_fingerprint=args.cluster_by_fingerprint,
        fingerprint_topn=args.fingerprint_topn,
    )


if __name__ == '__main__':
    main()
