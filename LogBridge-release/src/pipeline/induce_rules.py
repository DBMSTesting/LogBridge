"""Train-only template sampling, character clustering and audited rule induction."""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_config import RULE_INDUCTION_MODEL
from llm_client import call_json
from prepare_data import parsing_text, save, RuleExtractor

def prepare(windows, seed=42, quota=5):
    from drain3 import TemplateMiner
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    miner = TemplateMiner()
    for window in windows.values():
        for log in window.get('logs') or []:
            miner.add_log_message(parsing_text(log.get('raw_line') or log.get('message') or ''))
    strata = defaultdict(list)
    for window in windows.values():
        for log in window.get('logs') or []:
            text = parsing_text(log.get('raw_line') or log.get('message') or '')
            cluster = miner.match(text)
            if cluster:
                strata[cluster.cluster_id].append((text, cluster.get_template()))
    rng = random.Random(seed); examples = []; sampled = 0
    for key in sorted(strata):
        for text, template in rng.sample(strata[key], min(quota, len(strata[key]))):
            sampled += 1
            parameters = miner.extract_parameters(template, text, exact_matching=True) or []
            for parameter in parameters:
                examples.append({'value': parameter.value, 'template': template, 'context': text})
    if not examples:
        raise ValueError('No candidate parameters in the training sample')
    matrix = TfidfVectorizer(analyzer='char', ngram_range=(1, 4)).fit_transform([e['value'] for e in examples])
    labels = DBSCAN(eps=.45, min_samples=2, metric='cosine').fit_predict(matrix)
    groups = defaultdict(list)
    for i, (label, example) in enumerate(zip(labels, examples)):
        groups[str(label) if label >= 0 else f'noise-singleton-{i}'].append(example)
    return {'seed': seed, 'quota_per_template': quota, 'sampled_lines': sampled,
            'templates': len(strata), 'parameters': len(examples), 'clusters': dict(groups)}

def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--train-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('generate'); p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True); p.add_argument('--model', default=RULE_INDUCTION_MODEL)
    p = sub.add_parser('freeze'); p.add_argument('--reviewed-rules', type=Path, required=True)
    p.add_argument('--validation-record', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('validate'); p.add_argument('--rules', type=Path, required=True)
    p.add_argument('--validation-json', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        save(args.output, prepare(json.loads(args.train_json.read_text())))
    elif args.command == 'generate':
        candidates = json.loads(args.candidates.read_text())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rules = []
        for cluster, examples in candidates['clusters'].items():
            output = args.output_dir / f'{cluster}.json'
            if output.exists():
                result = json.loads(output.read_text())
            else:
                prompt = ('Decide whether the cluster denotes a diagnostic system entity or incidental noise. '
                          'For entities infer type, extraction regex and contextual identity_fields '
                          '(directory, node_id, host, or database). The regex may use a named value group. '
                          'Specify context needed to distinguish equal strings belonging to different objects. '
                          'Return {"label":"entity|noise","type":"...","regex":"...",'
                          '"identity_fields":[],"rationale":"..."}. Examples:\n' + json.dumps(examples[:40]))
                parsed, audit = call_json(prompt, args.model)
                result = {'parsed': parsed, 'audit': audit, 'prompt': prompt}; save(output, result)
            rule = result['parsed']
            if rule['label'] == 'entity':
                re.compile(rule['regex']); rules.append(rule)
        save(args.output_dir / 'draft_rules.json', {'status': 'draft', 'rules': rules,
                                                   'induction_model': args.model})
    elif args.command == 'validate':

        document = json.loads(args.rules.read_text())
        extractor = object.__new__(RuleExtractor)
        extractor.document = document
        extractor.compiled = [(r, re.compile(r['regex'])) for r in document['rules']]
        rows = []
        for wid, window in json.loads(args.validation_json.read_text()).items():
            for i, log in enumerate(window.get('logs') or []):
                text = log.get('raw_line') or log.get('message') or ''
                rows.append({'window_id':wid, 'log_index':i, 'text':text,
                             'entities':extractor.extract(text, {**window,'window_id':wid})})
        save(args.output, {'validation_source':str(args.validation_json), 'logs':len(rows),
             'logs_with_matches':sum(bool(row['entities']) for row in rows), 'rows':rows,
             'note':'Match coverage is not recall or precision; reviewer must identify missed/false matches.'})
    else:
        reviewed = json.loads(args.reviewed_rules.read_text())
        validation = json.loads(args.validation_record.read_text())
        if not validation.get('reviewer') or not validation.get('validation_source'):
            raise ValueError('A real reviewer and held-out validation source must be recorded')
        for rule in reviewed['rules']:
            re.compile(rule['regex'])
        reviewed.update(status='frozen', validation_record=validation)
        save(args.output, reviewed)

if __name__ == '__main__': main()
