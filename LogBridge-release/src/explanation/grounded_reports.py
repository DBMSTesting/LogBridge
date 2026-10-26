"""Window-specific report generation from exported classifier evidence."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_config import REPORT_GENERATION_MODEL, REPORT_JUDGE_MODEL
from llm_client import call_json
from build_retrieval_index import load_index, retrieve, entry_to_text

SYSTEM_RULES = '''The upstream prediction is an unverified lead, not a confirmed cause.
Do not reclassify the window. Absence of an explicit abnormal event does not establish normality.
Attention only selects evidence and is not causal proof. Counts refer to identical consecutive
template/entity assignments; they do not establish occurrence counts of other concrete values.
Separate target observations from historical observations and cite source log IDs for facts.
Explain what needs verification, why to check, how to check and how to interpret the result.
Corrective actions must be conditional and supported by cited KB text. Never invent commands,
parameters or thresholds. If knowledge is insufficient, provide concrete safe checks but no
unsupported corrective actions. Every report must reflect this window's own evidence.'''

SCHEMA = {
    'observed_facts': [{'fact': '', 'evidence_ids': []}],
    'diagnostics_to_verify': [{'hypothesis': '', 'check': '', 'why': '', 'decision_rule': '', 'evidence_ids': []}],
    'conditional_actions': [{'condition': '', 'action': '', 'citation_ids': []}],
    'limitations': [],
}

COMPONENTS = {
    'Network': r'network|grpc|socket|connect|packet|bandwidth|netty',
    'WAL': r'\bwal\b|raft|ratis|consensus|logdispatcher|logappender',
    'Election': r'election|leader|ratis|raft',
    'Compaction': r'compact', 'Flush': r'flush|memtable',
    'Memory': r'memory|heap|\bgc\b|outofmemory', 'Query': r'query|mpp|fragment',
    'TsFile': r'tsfile|export|load.*file', 'Schema': r'schema|metadata|partitioncache',
    'Procedure': r'procedure|migration', 'Concurrency': r'thread|pool|queue',
    'MemTable': r'memtable|write buffer',
}

def tokens(text):
    stop = {'check','with','from','this','that','iotdb','node','data','system','error','config','when','then'}
    return {w for w in re.findall(r'[a-z][a-z0-9_.-]{3,}', text.lower()) if w not in stop}

def retrieve_applicable(index, prediction, evidence, k=4):
    if not 0 <= k <= 4:
        raise ValueError('Retrieval permits zero to four entries')
    observed = '\n'.join(x['text'] + '\n' + x.get('text_last', '') for x in evidence)
    components = {c for c, pattern in COMPONENTS.items() if re.search(pattern, observed, re.I)}
    words = tokens(observed)
    query = f'Unverified classifier lead: {prediction}\n' + observed
    ranked = retrieve(query, index, k=len(index['entries']))
    candidates, audits = [], []
    for similarity, entry in ranked:
        overlap = components & set(entry.get('iotdb_subsystems') or [])
        signatures = words & tokens(' '.join(str(entry.get(key) or '') for key in ('title','symptom','root_cause','tags')))
        class_match = prediction in (entry.get('anomaly_classes') or [])
        parameter_match = any(p.lower() in observed.lower() for p in entry.get('config_params') or [])
        family = entry['id'].split('-')[0]
        if family == 'CFG':
            eligible = parameter_match or (bool(overlap) and len(signatures) >= 3)
        elif family in ('ISS','FAQ','SC'):
            eligible = bool(overlap) and len(signatures) >= 3
        else:
            eligible = bool(overlap) and bool(signatures)
        audit = {'id': entry['id'], 'eligible': eligible, 'component_overlap': sorted(overlap),
                 'symptom_overlap': sorted(signatures), 'parameter_match': parameter_match,
                 'prediction_soft_match': class_match, 'similarity': float(similarity)}
        audits.append(audit)
        if eligible:
            score = float(similarity) + .035 * bool(overlap) + .025 * class_match + min(.04, .01*len(signatures))
            candidates.append((score, entry))
    candidates.sort(key=lambda x: (-x[0], x[1]['id']))
    return [entry for _, entry in candidates[:k]], audits

def sanitize_evidence(snapshot):
    required = ('window_id', 'prediction', 'confidence', 'evidence', 'selected_history')
    if any(key not in snapshot for key in required):
        raise ValueError('Expected an exported classifier snapshot, including selected_history and evidence')
    allowed_history = set(snapshot['selected_history'])
    evidence = []
    for item in snapshot['evidence']:
        if item['source_window'] != snapshot['window_id'] and item['source_window'] not in allowed_history:
            raise ValueError('Evidence is not part of the actual classifier context')
        evidence.append({key: item[key] for key in ('id','text','text_last','source_window','timestamp_first',
                         'timestamp_last','shared_entities','count') if key in item})

    return {'window_id': snapshot['window_id'], 'prediction': snapshot['prediction'],
            'confidence': snapshot['confidence'], 'top_classes': snapshot.get('top_classes', []),
            'target_log_attention': snapshot.get('target_log_attention', []),
            'log_entity_attention': snapshot.get('log_entity_attention', []),
            'log_ids': snapshot.get('log_ids', []), 'entity_ids': snapshot.get('entity_ids', []),
            'entities': snapshot.get('entities', {}),
            'selected_history': snapshot['selected_history'], 'evidence': evidence}

def validate_report(report, evidence, knowledge):
    errors = []
    if not isinstance(report, dict) or not set(SCHEMA) <= set(report):
        return ['missing report fields']
    log_ids = {x['id'] for x in evidence}; kb_ids = {x['id'] for x in knowledge}
    for fact in report.get('observed_facts') or []:
        refs = fact.get('evidence_ids') or []
        if not fact.get('fact') or not refs or not set(refs) <= log_ids:
            errors.append('fact without valid evidence')
    checks = report.get('diagnostics_to_verify') or []
    if not checks:
        errors.append('at least one verification check required')
    for check in checks:
        if any(not check.get(key) for key in ('check','why','decision_rule')):
            errors.append('incomplete verification check')
        if not set(check.get('evidence_ids') or []) <= log_ids:
            errors.append('invalid check evidence')
    for action in report.get('conditional_actions') or []:
        if not action.get('condition') or not action.get('action'):
            errors.append('incomplete action')
        refs = action.get('citation_ids') or []
        if not refs or not set(refs) <= kb_ids:
            errors.append('action without knowledge citation')
    return sorted(set(errors))

def generate(snapshot, index, output, model=REPORT_GENERATION_MODEL, dry_run=False):
    material = sanitize_evidence(snapshot)
    knowledge, retrieval_audit = retrieve_applicable(index, material['prediction'], material['evidence'])
    prompt = SYSTEM_RULES + '\nSchema:\n' + json.dumps(SCHEMA) + '\nInput:\n' + json.dumps(
        {**material, 'retrieved_knowledge': knowledge}, ensure_ascii=False)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if output.exists():
        old = json.loads(output.read_text())
        if (old.get('prompt_sha256') == digest and old.get('model') == model
                and old.get('dry_run') == dry_run):
            return
        raise FileExistsError(f'Refusing to overwrite changed run: {output}')
    result = {'window_id': material['window_id'], 'model': model, 'prompt': prompt,
              'prompt_sha256': digest, 'generation_input': material, 'knowledge': knowledge,
              'retrieval_audit': retrieval_audit, 'dry_run': dry_run}
    if not dry_run:
        report, audit = call_json(prompt, model)
        errors = validate_report(report, material['evidence'], knowledge)
        result.update(report=report, api_audit=audit, validation_errors=errors)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))

def judge(record, index, model=REPORT_JUDGE_MODEL):

    prompt = '''Evaluate the report anonymously. Atomize asserted checkable facts only; a clearly
unverified hypothesis is not an asserted event. Unsupported causal assertions count as unsupported.
Judge facts only against supplied logs, never the prediction or KB. For each conditional_actions
entry assess full KB support for operation and conditions, specific actionable steps, and whether
it contains concrete unsupported commands/config/parameters/thresholds/technical assertions.
Missing full KB support does not imply an unsupported concrete technical assertion.
No checkable facts or no actions yields a null rate, never 100%; preserve zero denominators.
Keep each diagnostics_to_verify check separate from conditional_actions. Flag prediction treated
as confirmed and normality inferred from absence. Return {"facts":[{"claim":"", "verdict":
"supported|unsupported|not_checkable", "evidence_ids":[],"reason":""}], "actions":[{
"index":0,"kb_supported":false,"kb_ids":[],"effective":false,"unsupported_technical_assertion":false,
"reason":""}],"prediction_as_confirmed":false,"normality_from_absence":false}.
''' + json.dumps({'input': record['generation_input'], 'report': record['report'],
                  'frozen_kb': index['entries']}, ensure_ascii=False)
    parsed, audit = call_json(prompt, model, role='judge')
    expected = len(record['report'].get('conditional_actions') or [])
    actions = parsed.get('actions') or []
    errors = []
    if len(actions) != expected or sorted(x.get('index', -1) for x in actions) != list(range(expected)):
        errors.append('advice count or indices differ from report')
    kb_ids = {x['id'] for x in index['entries']}
    for action in actions:
        if not set(action.get('kb_ids') or []) <= kb_ids:
            errors.append('invalid knowledge ID')
        for key in ('kb_supported','effective','unsupported_technical_assertion'):
            if not isinstance(action.get(key), bool): errors.append('invalid boolean: ' + key)
    if not isinstance(parsed.get('facts'), list) or any(x.get('verdict') not in
           ('supported','unsupported','not_checkable') for x in parsed.get('facts') or []):
        errors.append('invalid fact verdicts')
    return {'model': model, 'automatic_not_human': True, 'parsed': parsed, 'api_audit': audit,
            'prompt': prompt, 'validation_errors': sorted(set(errors))}

def summarize_judgments(directory):
    total = {'reports':0, 'supported_facts':0, 'checkable_facts':0, 'actions':0,
             'supported_actions':0, 'effective_actions':0, 'unsupported_technical_actions':0,
             'no_checkable_facts_reports':0, 'no_actions_reports':0, 'invalid_judgments':0}
    for path in sorted(directory.glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('validation_errors'):
            total['invalid_judgments'] += 1; continue
        value = record['parsed']; facts = value['facts']; actions = value['actions']
        n = sum(f['verdict'] in ('supported','unsupported') for f in facts)
        total['reports'] += 1; total['checkable_facts'] += n
        total['supported_facts'] += sum(f['verdict']=='supported' for f in facts)
        total['actions'] += len(actions)
        total['supported_actions'] += sum(a['kb_supported'] for a in actions)
        total['effective_actions'] += sum(a['effective'] for a in actions)
        total['unsupported_technical_actions'] += sum(a['unsupported_technical_assertion'] for a in actions)
        total['no_checkable_facts_reports'] += n == 0
        total['no_actions_reports'] += len(actions) == 0
    def ratio(numerator, denominator):
        return {'numerator':total[numerator], 'denominator':total[denominator],
                'rate':total[numerator]/total[denominator] if total[denominator] else None}
    return {'aggregation':'micro over all facts/actions; empty denominators are null', 'counts': total,
            'explanation_support':ratio('supported_facts','checkable_facts'),
            'knowledge_support':ratio('supported_actions','actions'),
            'effective_advice':ratio('effective_actions','actions'),
            'unsupported_technical_assertions':ratio('unsupported_technical_actions','actions')}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--index', type=Path); parser.add_argument('--model', default=REPORT_GENERATION_MODEL)
    parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--judge', action='store_true')
    parser.add_argument('--judge-model', default=REPORT_JUDGE_MODEL)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        result = summarize_judgments(args.output_root / 'judgments')
        (args.output_root / 'metrics.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2)); return
    index = load_index(args.index) if args.index else load_index()
    from build_retrieval_index import load_all_entries
    if {e['id'] for e in index['entries']} != {e['id'] for e in load_all_entries()}:
        raise ValueError('Stale knowledge index; rebuild after loading all documented entries')
    rows = json.loads(args.diagnostics.read_text())
    if isinstance(rows, dict): rows = list(rows.values())
    for snapshot in rows:
        name = hashlib.sha256(snapshot['window_id'].encode()).hexdigest()[:20] + '.json'
        output = args.output_root / name
        if args.judge:
            record = json.loads(output.read_text())
            target = args.output_root / 'judgments' / name
            if not target.exists():
                result = judge(record, index, args.judge_model)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            generate(snapshot, index, output, args.model, args.dry_run)

if __name__ == '__main__': main()
