"""Export predictions plus exactly the historical evidence used by the model."""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
for folder in (SRC, SRC/'model', SRC/'pipeline', SRC/'utils', SRC/'training'):
    sys.path.insert(0, str(folder))
import torch
from encoder import TemplateEncoder, EntityEncoder
from hgt_anomaly_model import HGTAnomalyModel
from load_subgraphs import load_prebuilt_subgraphs
from prepare_data import CLASSES, PROTOCOL

def export_one(model, graph, windows):
    model(graph, target_window_idx=graph.target_window_idx)
    signal = model.last_diagnostics
    probs = signal['probabilities']; predicted = max(range(len(probs)), key=probs.__getitem__)
    target = graph.target_window_id
    history = list(graph.selected_history)
    target_entities = {e['id'] for log in windows[target]['logs'] for e in log['typed_entities']}
    evidence = []; entities = {}
    weights = dict(zip(signal['target_log_ids'], signal['target_log_attention']))
    for wid in [target] + history:
        for i, log in enumerate(windows[wid]['logs']):
            entities.update({e['id']:e for e in log['typed_entities']})
            evidence.append({'id': f'LogInstance:{wid.removeprefix("Window:")}_{i}',
                'text': log['raw_line'], 'text_last': log['raw_line_last'],
                'source_window': wid, 'timestamp_first': log['first_ts'],
                'timestamp_last': log['last_ts'], 'count': log['count'],
                'shared_entities': sorted(target_entities & {e['id'] for e in log['typed_entities']})})

    evidence.sort(key=lambda item: (item['source_window'] != target,
                                   -weights.get(item['id'], 0.0) if item['source_window'] == target else 0.0))
    return {'window_id': target, 'prediction': CLASSES[predicted], 'confidence': probs[predicted],
            'top_classes': [{'class': CLASSES[i], 'probability': probs[i]} for i in sorted(range(6), key=probs.__getitem__, reverse=True)],
            'selected_history': history, 'evidence': evidence,
            'target_log_attention': list(zip(signal['target_log_ids'], signal['target_log_attention'])),
            'log_entity_attention': signal['log_entity_attention'], 'entity_ids': graph.entity_ids,
            'log_ids': graph.log_ids, 'entities': entities, 'protocol': PROTOCOL}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train','val','test'], default='test'); parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError('Checkpoint context schema does not match the dataset')
    embeddings = torch.load(args.data_root/'bert_embeddings.pt', map_location='cpu', weights_only=False)
    embeddings = embeddings.get('embeddings', embeddings)
    templates = TemplateEncoder(precomputed_embeddings=embeddings, device=args.device)
    entities = EntityEncoder(checkpoint['entity_vocab'], device=args.device)
    model = HGTAnomalyModel(templates, entities, log_embedding_dim=templates.embedding_dim,
                            num_classes=6, device=args.device).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
    graphs, metadata = load_prebuilt_subgraphs(args.data_root/'contexts', args.split, load_all=True)
    if metadata.get('protocol') != PROTOCOL: raise ValueError('Context schema mismatch')
    windows = {}
    for split in ('train','val','test'):
        raw = json.loads((args.data_root/f'windows_anomaly_{split}_compressed.json').read_text())
        windows.update({wid if wid.startswith('Window:') else 'Window:'+wid:w for wid,w in raw.items()})
    with torch.no_grad():
        results = [export_one(model, graph.to(args.device), windows) for graph in graphs.values()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
