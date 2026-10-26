#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train lt only."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT / 'model'))

from local_transformer import LocalTransformerEncoder

ANOMALY_TYPE_TO_LABEL = {
    'compaction': 0, 'export': 1, 'flush': 2,
    'full_cpu': 3, 'full_memory': 4, 'network_bandwidth2': 5,
}
LABEL_TO_NAME = {v: k for k, v in ANOMALY_TYPE_TO_LABEL.items()}
ANOMALY_LABELS = set(ANOMALY_TYPE_TO_LABEL.values())

class CompressedWindowDataset(Dataset):
    """Compressedwindowdataset."""

    def __init__(self, json_path: Path, bert_emb: dict[str, torch.Tensor],
                 emb_dim: int, max_seq_len: int = 200):
        self.bert_emb = bert_emb
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.records = []

        with open(json_path) as f:
            data = json.load(f)
        n_normal_skipped = 0
        for wid, w in data.items():
            anom = (w.get('anomaly_types') or ['normal'])[0]
            label = ANOMALY_TYPE_TO_LABEL.get(anom, 6)
            if label not in ANOMALY_LABELS:
                n_normal_skipped += 1
                continue

            if w.get('protocol') != 'logbridge-context':
                raise ValueError('Rebuild compressed windows with prepare_data.py')
            logs = w['logs']

            raw_lines = []
            for r in logs:
                line = r.get('template_text', '')
                if not line:
                    continue
                raw_lines.append(line)
            if not raw_lines:
                continue

            if len(raw_lines) > self.max_seq_len:
                raw_lines = raw_lines[:self.max_seq_len]
            self.records.append((raw_lines, label))
        print(f'  loaded {json_path.name}: {len(self.records)} 异常窗口  '
              f'(skipped {n_normal_skipped} normal)')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        raw_lines, label = self.records[idx]

        embs = []
        for line in raw_lines:
            if line in self.bert_emb:
                embs.append(self.bert_emb[line])
            else:
                embs.append(torch.zeros(self.emb_dim))
        emb_tensor = torch.stack(embs)
        return emb_tensor, len(embs), label

def collate(batch):
    """变长序列对齐为 max length，返回 mask"""
    max_len = max(b[1] for b in batch)
    D = batch[0][0].size(-1)
    embs = torch.zeros(len(batch), max_len, D)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, (e, l, lbl) in enumerate(batch):
        embs[i, :l] = e
        mask[i, :l] = True
        labels[i] = lbl
    return embs, mask, labels

class LTOnlyModel(nn.Module):
    def __init__(self, input_dim=768, d_model=256, nhead=8, num_layers=2,
                 num_classes=6, dropout=0.1):
        super().__init__()
        self.transformer = LocalTransformerEncoder(
            input_dim=input_dim, d_model=d_model,
            nhead=nhead, num_layers=num_layers,
            dim_feedforward=512, dropout=dropout, max_seq_len=200,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        feats = self.transformer(x, padding_mask=~mask)

        mask_f = mask.float().unsqueeze(-1)
        summed = (feats * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        return self.classifier(pooled)


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, labels_all = [], [], []
    with torch.no_grad():
        for embs, mask, labels in loader:
            embs, mask, labels = embs.to(device), mask.to(device), labels.to(device)
            logits = model(embs, mask)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    return float(np.mean(losses)), preds, labels_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path,
                    default=Path('./datasets/<dataset>'))
    ap.add_argument('--bert-emb', type=Path,
                    default=Path('./datasets/<dataset>/bert_embeddings.pt'))
    ap.add_argument('--output-dir', type=Path,
                    default=Path('./datasets/<dataset>/checkpoints_lt_only'))
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 70)
    print('Local-Transformer-Only 消融实验（无子图无 HGT 无实体）')
    print('=' * 70)
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
    print('=' * 70, flush=True)

    print(f'\n[1] 加载 BERT 预编码: {args.bert_emb}')
    sys.stdout.flush()
    bert_emb_raw = torch.load(args.bert_emb, map_location='cpu')
    if isinstance(bert_emb_raw, dict) and 'embeddings' in bert_emb_raw:
        bert_emb = bert_emb_raw['embeddings']
    else:
        bert_emb = bert_emb_raw
    sample_key = next(iter(bert_emb))
    emb_dim = bert_emb[sample_key].shape[0]
    print(f'  raw_line 模板数: {len(bert_emb)}  emb_dim: {emb_dim}')
    sys.stdout.flush()

    print('\n[2] 加载有序模板序列')
    sys.stdout.flush()
    train_ds = CompressedWindowDataset(args.input_dir / 'windows_anomaly_train_compressed.json',
                                       bert_emb, emb_dim, max_seq_len=150)
    val_ds = CompressedWindowDataset(args.input_dir / 'windows_anomaly_val_compressed.json',
                                     bert_emb, emb_dim, max_seq_len=150)
    test_ds = CompressedWindowDataset(args.input_dir / 'windows_anomaly_test_compressed.json',
                                      bert_emb, emb_dim, max_seq_len=150)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=2)
    print(f'  batches/epoch: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}')

    train_labels = [r[1] for r in train_ds.records]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    print(f'\n[4] 开始训练\n' + '=' * 70, flush=True)
    best_val_acc = 0.0
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss_accum, n_correct, n_total = 0.0, 0, 0
        for embs, mask, labels in train_loader:
            embs, mask, labels = embs.to(args.device), mask.to(args.device), labels.to(args.device)
            logits = model(embs, mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_accum += loss.item() * embs.size(0)
            preds = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += embs.size(0)
        train_loss = train_loss_accum / n_total
        train_acc = n_correct / n_total

        val_loss, val_preds, val_labels_arr = evaluate(model, val_loader, args.device, criterion)
        val_acc = float(np.mean([p == l for p, l in zip(val_preds, val_labels_arr)]))
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f'\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s)')
        print(f'  train  loss={train_loss:.4f}  acc={train_acc:.4f}')
        print(f'  val    loss={val_loss:.4f}  acc={val_acc:.4f}')

        prec, rec, f1, sup = precision_recall_fscore_support(
            val_labels_arr, val_preds, labels=list(range(6)), zero_division=0)
        print(f'  per-class F1: ' + '  '.join(
            f'{LABEL_TO_NAME[i][:5]}={f1[i]:.2f}(s={int(sup[i])})' for i in range(6)))
        print(f'  macro F1 = {np.mean(f1):.4f}', flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / 'best_model.pt')
            print(f'  ✓ 保存最佳模型 (val_acc={val_acc:.4f})')
            patience = 0
        else:
            patience += 1
            print(f'  耐心 {patience}/{args.patience}')
            if patience >= args.patience:
                print(f'\n[早停] val_acc 已 {patience} epoch 未提升')
                break

    print(f'\n[5] 在测试集上评估最佳模型')
    model.load_state_dict(torch.load(args.output_dir / 'best_model.pt'))
    test_loss, test_preds, test_labels_arr = evaluate(model, test_loader, args.device, criterion)
    test_acc = float(np.mean([p == l for p, l in zip(test_preds, test_labels_arr)]))
    print(f'\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}')
    print('\n分类报告:')
    print(classification_report(test_labels_arr, test_preds,
                                target_names=[LABEL_TO_NAME[i] for i in range(6)],
                                zero_division=0, digits=3))
    print('\n混淆矩阵 (行=真实, 列=预测):')
    cm = confusion_matrix(test_labels_arr, test_preds, labels=list(range(6)))
    names = [LABEL_TO_NAME[i][:9] for i in range(6)]
    print(' ' * 12 + ' '.join(f'{n:>9}' for n in names))
    for i, row in enumerate(cm):
        print(f'{names[i]:>10}: ' + ' '.join(f'{v:>9}' for v in row))

if __name__ == '__main__':
    main()
