"""Temporal data preparation, contextual entity identities and run compression."""
import argparse
import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ('compaction', 'export', 'flush', 'full_cpu', 'full_memory', 'network_bandwidth2')
PROTOCOL = 'logbridge-context'

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        parsed = datetime.fromisoformat(str(value).replace(',', '.').replace('Z', '+00:00'))

        return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()

def bounds(window):
    logs = window.get('logs') or []
    start = window.get('window_start')
    end = window.get('window_end')
    if start is None or start == '':
        start = logs[0].get('timestamp') if logs else None
    if end is None or end == '':
        end = logs[-1].get('timestamp') if logs else None
    if start is None or end is None:
        raise ValueError('Window start/end timestamps are required; never infer duration from its ID')
    a, b = timestamp(start), timestamp(end)
    if b < a:
        raise ValueError('Window ends before it starts')
    return a, b

def parsing_text(line):

    line = re.sub(r'^\s*\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d[.,]\d+\s*', '', line)
    return re.sub(r'(?<!\w)(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)(?!\w)\s*', '', line, count=1)

class RuleExtractor:
    def __init__(self, rule_file):
        self.document = json.loads(Path(rule_file).read_text())
        if self.document.get('status') != 'frozen':
            raise ValueError('Extraction requires a validated, frozen rules file')
        self.rules = self.document['rules']
        self.compiled = [(rule, re.compile(rule['regex'])) for rule in self.rules]

    def extract(self, text, source):
        entities = {}
        for rule, pattern in self.compiled:
            for match in pattern.finditer(text):
                value = match.groupdict().get('value') or match.group(0)
                context = {}
                for key in rule.get('identity_fields', []):
                    if key == 'directory':
                        value_start = match.start('value') if 'value' in match.groupdict() else match.start()
                        prefix = text[:value_start]
                        token = re.search(r'([^\s\[\](),]*[/\\])$', prefix)
                        context[key] = token.group(1).rstrip('/\\') if token else ''
                    else:
                        context[key] = str(source.get(key, ''))
                    if not context[key]:

                        context[key] = 'unknown:' + str(source['window_id'])
                identity = json.dumps([rule['type'], value, context], sort_keys=True, ensure_ascii=False)
                eid = 'GeneralEntity:' + hashlib.sha256(identity.encode()).hexdigest()
                entities[eid] = {'id': eid, 'type': rule['type'], 'value': value, 'context': context}
        return [entities[k] for k in sorted(entities)]

def temporal_split(windows, event_map, train_ratio=.8, val_ratio=.1):
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 - train_ratio:
        raise ValueError('Invalid split ratios')
    groups = defaultdict(list)
    for wid, window in windows.items():
        if not any(c in CLASSES for c in window.get('anomaly_types', [])):
            continue
        event = event_map.get(wid) or window.get('incident_id') or window.get('event_id')
        if not event:
            raise ValueError(f'Missing fault-injection incident ID for {wid}; provide --event-map')
        groups[str(event)].append((wid, window))
    intervals = sorted((min(bounds(w)[0] for _, w in items),
                        max(bounds(w)[1] for _, w in items), event, items)
                       for event, items in groups.items())

    blocks = []
    for start, end, event, items in intervals:
        if blocks and start <= blocks[-1]['end']:
            blocks[-1]['end'] = max(end, blocks[-1]['end'])
            blocks[-1]['items'].extend(items); blocks[-1]['events'].append(event)
        else:
            blocks.append({'start': start, 'end': end, 'items': list(items), 'events': [event]})
    if len(blocks) < 3:
        raise ValueError('At least three nonoverlapping incident blocks are needed for temporal splits')
    total = sum(len(b['items']) for b in blocks)
    cumulative = []; count = 0
    for block in blocks:
        count += len(block['items']); cumulative.append(count)
    cut1 = min(range(1, len(blocks)-1), key=lambda i: abs(cumulative[i-1] / total - train_ratio))
    cut2 = min(range(cut1+1, len(blocks)), key=lambda i: abs(cumulative[i-1] / total - train_ratio-val_ratio))
    output = {}; manifest = []
    for split, chosen in zip(('train', 'val', 'test'), (blocks[:cut1], blocks[cut1:cut2], blocks[cut2:])):
        output[split] = {wid: w for b in chosen for wid, w in b['items']}
        manifest.extend({'split': split, 'window_id': wid, 'event_id': str(event_map.get(wid) or w.get('incident_id') or w.get('event_id')),
                         'start': bounds(w)[0], 'end': bounds(w)[1]} for wid, w in output[split].items())
    return output, manifest

def compress_window(wid, window, miner, extractor):
    runs = []
    for original_index, record in enumerate(window.get('logs') or []):
        text = record.get('raw_line') or record.get('message') or ''
        parsed = parsing_text(text)
        matched = miner.match(parsed)

        template = matched.get_template() if matched else parsed
        tid = str(matched.cluster_id) if matched else 'unknown:' + hashlib.sha256(parsed.encode()).hexdigest()
        entities = extractor.extract(text, {**window, 'window_id': wid})
        key = (tid, tuple(e['id'] for e in entities))
        ts = record.get('timestamp')
        if ts is None:
            raise ValueError(f'Missing log timestamp in {wid}')
        if runs and runs[-1]['_key'] == key:
            run = runs[-1]; run['count'] += 1; run['last_ts'] = ts
            run['raw_line_last'] = text; run['original_indices'].append(original_index)
        else:
            runs.append({'_key': key, 'template_id': tid, 'template_text': template,
                         'raw_line': text, 'raw_line_last': text, 'count': 1,
                         'first_ts': ts, 'last_ts': ts, 'typed_entities': entities,
                         'original_indices': [original_index]})
    for run in runs:
        del run['_key']
    return {**window, 'logs': runs, 'log_count_original': len(window.get('logs') or []),
            'log_count': len(runs), 'protocol': PROTOCOL}

def select_history(target, index, limit=5, allowed=None):
    if limit < 0:
        raise ValueError('History limit must be nonnegative')
    entities = set(index['window_to_entities'].get(target, []))
    candidates = set()
    for entity in entities:
        candidates.update(index['entity_to_windows'].get(entity, []))
    start = index['window_bounds'][target][0]
    split = index['window_split'][target]
    ranked = []
    for other in candidates:
        if other == target or (allowed is not None and other not in allowed):
            continue
        if index['window_split'][other] != split or index['window_bounds'][other][1] >= start:
            continue
        score = len(entities & set(index['window_to_entities'][other]))
        if score > 0:
            ranked.append((other, score))
    ranked.sort(key=lambda pair: (-pair[1], -index['window_bounds'][pair[0]][1], pair[0]))
    return [wid for wid, _ in ranked[:limit]]

def build_artifacts(splits, output, dataset, limit=5):
    index = {k: {} for k in ('window_to_logs','window_to_entities','log_to_entities','log_to_text',
                             'entity_to_content','window_to_label','log_to_count','window_bounds','window_split')}
    index['entity_to_windows'] = defaultdict(list)
    index['protocol'] = PROTOCOL
    nodes, edges, entity_nodes = [], [], {}
    for split, windows in splits.items():
        for raw_wid, window in windows.items():
            wid = raw_wid if raw_wid.startswith('Window:') else 'Window:' + raw_wid
            index['window_bounds'][wid] = bounds(window)
            index['window_split'][wid] = split
            category = next(c for c in window['anomaly_types'] if c in CLASSES)
            index['window_to_label'][wid] = CLASSES.index(category)
            nodes.append({'id': wid, 'type': 'Window', 'properties': {'anomaly_types': [category],
                          'window_start': bounds(window)[0], 'window_end': bounds(window)[1]}})
            log_ids, entity_ids = [], set()
            for i, log in enumerate(window['logs']):
                lid = f'LogInstance:{wid.removeprefix("Window:")}_{i}'
                log_ids.append(lid)
                ids = [e['id'] for e in log['typed_entities']]
                entity_ids.update(ids)
                index['log_to_entities'][lid] = ids
                index['log_to_text'][lid] = log['template_text']
                index['log_to_count'][lid] = log['count']
                nodes.append({'id': lid, 'type': 'LogInstance', 'properties': log})
                edges.append({'source': wid, 'target': lid, 'relation': 'CONTAINS'})
                for entity in log['typed_entities']:
                    index['entity_to_content'][entity['id']] = entity['value']
                    entity_nodes[entity['id']] = {'id': entity['id'], 'type': 'GeneralEntity', 'properties': entity}
                    edges.append({'source': lid, 'target': entity['id'], 'relation': 'ASSOCIATED_WITH'})
            index['window_to_logs'][wid] = log_ids
            index['window_to_entities'][wid] = sorted(entity_ids)
            for entity in sorted(entity_ids):
                index['entity_to_windows'][entity].append(wid)
    index['all_windows'] = list(index['window_to_logs'])
    index['window_entity_sets'] = {k: set(v) for k, v in index['window_to_entities'].items()}
    index['entity_to_windows'] = dict(index['entity_to_windows'])
    save(output / f'kg_{dataset}.json', {'nodes': nodes + list(entity_nodes.values()), 'edges': edges,
                                          'protocol': PROTOCOL})
    with (output / 'context_index.pkl').open('wb') as f:
        pickle.dump(index, f)
    import torch
    import subgraphs as builder
    builder._index = index
    dest = output / 'contexts'
    for split in ('train','val','test'):
        labels, chunk_files, chunk = {}, [], {}
        chunk_dir = dest / f'chunks_{split}'; chunk_dir.mkdir(parents=True, exist_ok=True)
        for wid in index['all_windows']:
            if index['window_split'][wid] != split:
                continue
            _, graph, label, _ = builder.build_single_subgraph_lightweight((wid, 1, limit, None))
            if graph is None:
                raise RuntimeError(f'Could not build graph for {wid}')
            graph.protocol = PROTOCOL
            graph.selected_history = select_history(wid, index, limit)
            chunk[wid] = graph; labels[wid] = label
            if len(chunk) >= 1000:
                name = f'chunk_{len(chunk_files):05d}.pt'; torch.save(chunk, chunk_dir / name)
                chunk_files.append(name); chunk = {}
        if chunk:
            name = f'chunk_{len(chunk_files):05d}.pt'; torch.save(chunk, chunk_dir / name); chunk_files.append(name)
        torch.save({'chunk_files': chunk_files, 'chunk_dir': str(chunk_dir.resolve()),
                    'num_subgraphs': len(labels), 'metadata': {'protocol': PROTOCOL, 'window_to_label': labels,
                    'num_hops': 1, 'max_neighbors': limit}}, dest / f'subgraphs_{split}.pt')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--windows-json', type=Path, required=True)
    parser.add_argument('--event-map', type=Path); parser.add_argument('--rules', type=Path)
    parser.add_argument('--split-only', action='store_true', help='Write raw train/val/test inputs for offline rule induction')
    parser.add_argument('--dataset', required=True); parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--history-limit', type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Output directory must be empty')
    windows = json.loads(args.windows_json.read_text())
    events = json.loads(args.event_map.read_text()) if args.event_map else {}
    splits, manifest = temporal_split(windows, events)
    if args.split_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for split, records in splits.items():
            save(args.output_dir / f'windows_anomaly_{split}.json', records)
        save(args.output_dir / 'split_manifest.json', {'protocol': PROTOCOL, 'windows': manifest})
        return
    if args.rules is None:
        raise ValueError('--rules is required after validation and freezing')
    extractor = RuleExtractor(args.rules)
    from drain3 import TemplateMiner
    miner = TemplateMiner()
    for window in splits['train'].values():
        for log in window.get('logs') or []:
            miner.add_log_message(parsing_text(log.get('raw_line') or log.get('message') or ''))
    compressed = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        compressed[split] = {wid: compress_window(wid, w, miner, extractor) for wid, w in records.items()}
        save(args.output_dir / f'windows_anomaly_{split}_compressed.json', compressed[split])
    save(args.output_dir / 'split_manifest.json', {'protocol': PROTOCOL, 'windows': manifest,
         'rules_sha256': hashlib.sha256(args.rules.read_bytes()).hexdigest()})
    build_artifacts(compressed, args.output_dir, args.dataset, args.history_limit)

if __name__ == '__main__':
    main()
