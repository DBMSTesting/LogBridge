import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT/'src', ROOT/'src/pipeline', ROOT/'src/model', ROOT/'src/explanation', ROOT/'src/training'):
    sys.path.insert(0, str(folder))
import torch
from prepare_data import (bounds, temporal_split, RuleExtractor, compress_window, select_history,
                        build_artifacts, PROTOCOL)
from local_transformer import LocalTransformerEncoder
from encoder import EntityEncoder
from hgt_anomaly_model import HGTAnomalyModel
from export_diagnostics import export_one
from grounded_reports import sanitize_evidence, validate_report, generate, summarize_judgments


def window(start, texts, event='event'):
    return {'window_start': start, 'window_end': start+1, 'incident_id': event,
            'anomaly_types':['flush'], 'logs':[{'raw_line': text,'timestamp': start+i*.01}
                                              for i,text in enumerate(texts)]}


class FakeMiner:
    def match(self, text):
        class Cluster:
            cluster_id = 1
            def get_template(self): return 'open <*>'
        return Cluster()


class FakeTemplates:
    embedding_dim = 768
    def encode_batch(self, texts):
        return torch.ones(len(texts), 768)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rule_file = self.root/'rules.json'
        self.rule_file.write_text(json.dumps({'status':'frozen','rules':[{'type':'File',
            'regex':r'(?P<value>\d+-\d+-\d+-\d+\.tsfile)', 'identity_fields':['directory']}]}))
        self.extractor = RuleExtractor(self.rule_file)

    def test_identity_and_run_length(self):
        w = window(10, ['open /a/1-0-0-0.tsfile','open /a/1-0-0-0.tsfile',
                        'open /b/1-0-0-0.tsfile','open /a/1-0-0-0.tsfile'])
        compressed = compress_window('w', w, FakeMiner(), self.extractor)
        self.assertEqual([x['count'] for x in compressed['logs']], [2,1,1])
        a,b,c = compressed['logs']
        self.assertEqual(a['typed_entities'],c['typed_entities'])
        self.assertNotEqual(a['typed_entities'],b['typed_entities'])
        self.assertEqual(sum(x['count'] for x in compressed['logs']),4)
        a=self.extractor.extract('open rootA/region/1-0-0-0.tsfile',{'window_id':'w'})
        b=self.extractor.extract('open rootB/region/1-0-0-0.tsfile',{'window_id':'w'})
        self.assertNotEqual(a[0]['id'],b[0]['id'])

    def test_temporal_event_split(self):
        windows = {str(i):window(i*10,['x'],f'event{i}') for i in range(10)}
        splits, manifest = temporal_split(windows,{})
        self.assertLess(max(bounds(w)[1] for w in splits['train'].values()),
                        min(bounds(w)[0] for w in splits['val'].values()))
        event_sets = [{x['event_id'] for x in manifest if x['split']==s} for s in ('train','val','test')]
        self.assertFalse(event_sets[0] & event_sets[2])
        windows['0'].pop('incident_id')
        with self.assertRaises(ValueError): temporal_split(windows,{})

    def test_history_scope_and_exact_limit(self):
        ids = [f'w{i}' for i in range(9)]
        index = {'window_to_entities':{w:['e'] for w in ids},'entity_to_windows':{'e':ids},
                 'window_split':{w:'test' for w in ids},
                 'window_bounds':{w:(i*10,i*10+1) for i,w in enumerate(ids)}}
        index['window_split']['w6']='train'
        chosen = select_history('w7',index,5)
        self.assertEqual(chosen,['w5','w4','w3','w2','w1'])
        self.assertNotIn('w8',chosen)
        self.assertEqual(select_history('w0',index),[])

    def test_count_without_expansion(self):
        encoder=LocalTransformerEncoder(768,dropout=0).eval(); x=torch.randn(1,3,768)
        a=encoder(x,[1,1,1]); b=encoder(x,[1,100,1])
        self.assertEqual(b.shape,(1,3,256)); self.assertFalse(torch.allclose(a,b))

    def test_graph_model_export_and_no_gold(self):
        from load_subgraphs import load_prebuilt_subgraphs
        splits={s:{} for s in ('train','val','test')}
        for i in range(3):
            raw=window(i*10,['open /a/1-0-0-0.tsfile'])
            splits['test'][str(i)]=compress_window(str(i),raw,FakeMiner(),self.extractor)
        build_artifacts(splits,self.root,'synthetic',5)
        graphs,metadata=load_prebuilt_subgraphs(self.root/'contexts','test',load_all=True)
        self.assertEqual(metadata['protocol'],PROTOCOL)
        graph=graphs['Window:2']; self.assertEqual(len(graph.selected_history),2)
        from train import WindowDataset
        dataset=WindowDataset(['Window:2'],[2],prebuilt_subgraph_dir=self.root/'contexts',
                              split_name='test',num_hops=1,max_neighbors=5)
        loaded,label,target_idx=dataset[0]
        self.assertEqual(loaded.target_window_id,'Window:2'); self.assertEqual(label,2)
        entities=EntityEncoder({},device='cpu')
        self.assertTrue(torch.equal(entities.encode_batch(graph.entity_ids),torch.zeros(len(graph.entity_ids),128)))
        model=HGTAnomalyModel(FakeTemplates(),entities,log_embedding_dim=768,device='cpu').eval()
        normalized={'Window:'+wid:w for wid,w in splits['test'].items()}
        with torch.no_grad(): snapshot=export_one(model,graph,normalized)
        self.assertEqual(len(snapshot['log_entity_attention']),4)
        snapshot['gold']='flush'; clean=sanitize_evidence(snapshot)
        self.assertNotIn('gold',clean); self.assertEqual(len(clean['evidence']),3)
        self.assertEqual(len(snapshot['target_log_attention']),1)
        loss=model(graph).sum(); loss.backward()
        self.assertIsNotNone(model.local_transformer.count_embedding.weight.grad)

    def test_report_empty_knowledge_checks_only(self):
        report={'observed_facts':[], 'diagnostics_to_verify':[{'check':'Read heap usage',
            'why':'Verify pressure','decision_rule':'Compare with configured limit','evidence_ids':[]}],
            'conditional_actions':[],'limitations':['No suitable KB entry']}
        self.assertEqual(validate_report(report,[],[]),[])
        report['conditional_actions']=[{'condition':'If high','action':'Change invented_parameter','citation_ids':[]}]
        self.assertTrue(validate_report(report,[],[]))

    def test_unique_reports_and_empty_denominators(self):
        report={'observed_facts':[], 'diagnostics_to_verify':[{'check':'Read process metrics',
                'why':'Verify lead','decision_rule':'Compare with limit'}],
                'conditional_actions':[], 'limitations':['No suitable knowledge']}
        calls=[]
        def fake_call(prompt, model):
            calls.append(prompt); return report, {'actual_model':model,'usage':{}}
        with patch('grounded_reports.retrieve_applicable', return_value=([],[])), \
             patch('grounded_reports.call_json', side_effect=fake_call):
            for i in range(2):
                snapshot={'window_id':f'w{i}','prediction':'flush','confidence':.9,
                          'selected_history':[], 'evidence':[], 'gold':'do-not-pass'}
                generate(snapshot,{},self.root/f'report{i}.json')
        self.assertEqual(len(calls),2)
        self.assertNotEqual(calls[0],calls[1]); self.assertNotIn('do-not-pass',''.join(calls))
        directory=self.root/'judgments'; directory.mkdir()
        (directory/'empty.json').write_text(json.dumps({'parsed':{'facts':[],'actions':[]}}))
        metrics=summarize_judgments(directory)
        self.assertIsNone(metrics['explanation_support']['rate'])
        self.assertIsNone(metrics['knowledge_support']['rate'])
        self.assertEqual(metrics['counts']['no_actions_reports'],1)


if __name__=='__main__': unittest.main()
