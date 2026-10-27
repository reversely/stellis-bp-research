"""Multi-task 1D CNN: camera signals -> finger pulse wave and blood pressure.

Follows the MCD-rPPG paper's setup (Egorov et al. 2025, Table 5): 20 s windows, the sum of
MSE losses over the standardised pulse wave and BP, Adam, the authors' subject folds with
fold 0 as the test set and each other fold in turn as validation. A recording's estimate is
the median over its windows.

Usage: uv run python -m stellis_bp_research.train [--inputs rgb,pos] [--seconds 20] [--folds paper|all]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from torch import nn

from .baselines import PROFILE, scores
from .data import BP
from .signals import recording, table, windows

RUNS = Path(__file__).resolve().parents[2] / 'runs'
LR, BATCH = 1e-3, 64


def load_one(name):
    r = recording(name)
    return name, {k: v.astype(np.float32) for k, v in r.items() if k != 'found'}


def dataset(rows, workers=16):
    cache = RUNS / 'signals_cache.npz'  # delete after changing signals.py
    if cache.exists():
        d = np.load(cache, allow_pickle=True)['d'].item()
        if set(rows.name) <= set(d):
            return d
    with Pool(workers) as pool:
        d = dict(pool.imap_unordered(load_one, rows.name, chunksize=4))
    np.savez(cache, d=np.array(d, dtype=object))
    return d


def stack(rows, signals, inputs, seconds, stride):
    xs, ppgs, bps, rec = [], [], [], []
    for i, row in enumerate(rows.itertuples()):
        s = signals[row.name]
        x = windows(np.concatenate([s[k] for k in inputs], 1), seconds, stride)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        p = windows(s['ppg'], seconds, stride)
        p = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-6)
        xs.append(x)
        ppgs.append(p)
        bps.append(np.tile([row.upper_ap, row.lower_ap], (len(x), 1)))
        rec += [i] * len(x)
    return (torch.tensor(np.concatenate(xs)).transpose(1, 2), torch.tensor(np.concatenate(ppgs)),
            torch.tensor(np.concatenate(bps), dtype=torch.float32), np.array(rec))


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, 7, stride, 3), nn.BatchNorm1d(cout), nn.GELU(),
            nn.Conv1d(cout, cout, 7, 1, 3), nn.BatchNorm1d(cout))
        self.skip = nn.Conv1d(cin, cout, 1, stride) if (cin, stride) != (cout, 1) else nn.Identity()

    def forward(self, x):
        return nn.functional.gelu(self.body(x) + self.skip(x))


class Net(nn.Module):
    """Small 1D feature pyramid: the pulse wave is decoded from every level, BP from the pooled top."""
    def __init__(self, channels, width=32, depth=4):
        super().__init__()
        widths = [width * 2 ** min(i, 2) for i in range(depth)]
        self.down = nn.ModuleList(Block(c, w, 1 if i == 0 else 2) for i, (c, w) in enumerate(zip([channels] + widths, widths)))
        self.lateral = nn.ModuleList(nn.Conv1d(w, width, 1) for w in widths)
        self.ppg = nn.Conv1d(width, 1, 7, 1, 3)
        self.bp = nn.Sequential(nn.Dropout(0.3), nn.Linear(widths[-1] * 2, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        levels = []
        for block in self.down:
            x = block(x)
            levels.append(x)
        top = self.lateral[-1](levels[-1])
        for level, lat in zip(levels[-2::-1], self.lateral[-2::-1]):
            top = lat(level) + nn.functional.interpolate(top, size=level.shape[-1], mode='linear')
        pooled = torch.cat([levels[-1].mean(-1), levels[-1].std(-1)], 1)
        return self.ppg(top)[:, 0], self.bp(pooled)


def per_recording(pred, rec, n):
    return np.stack([np.median(pred[rec == i], 0) for i in range(n)])


def run(rows, signals, test, valid, args, device='cuda'):
    torch.manual_seed(args.seed + int(valid))
    parts = {k: rows[m].reset_index(drop=True) for k, m in (
        ('train', ~rows.fold.isin([test, valid])), ('valid', rows.fold == valid), ('test', rows.fold == test))}
    data = {k: stack(v, signals, args.inputs, args.seconds, args.stride if k == 'train' else args.seconds / 2)
            for k, v in parts.items()}
    mean, std = data['train'][2].mean(0), data['train'][2].std(0)
    model = Net(data['train'][0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
    x, p, y, _ = data['train']
    best, state = np.inf, None

    def predict(k):
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(b.to(device))[1].cpu() for b in data[k][0].split(512)])
        return per_recording((out * std + mean).numpy(), data[k][3], len(parts[k]))

    for epoch in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(x)).split(BATCH):
            xb = x[idx].to(device)
            if args.augment:
                xb = xb * (1 + 0.1 * torch.randn(len(xb), 1, 1, device=device))
            ppg, bp = model(xb)
            loss = nn.functional.mse_loss(ppg, p[idx].to(device)) * args.ppg_weight + \
                nn.functional.mse_loss(bp, ((y[idx] - mean) / std).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        mae = np.abs(predict('valid') - parts['valid'][BP].to_numpy()).mean()
        if mae < best:
            best, state = mae, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    pred = {k: predict(k) for k in ('valid', 'test')}
    truth = parts['test'][BP].to_numpy()
    train_rows = rows[~rows.fold.isin([test])]
    prof = lambda f: f[PROFILE[:1] + PROFILE[2:]].assign(female=(f.sex == 'F').astype(float)).to_numpy()
    ridge = lambda: RidgeCV(alphas=np.logspace(-2, 4, 13))
    profile = ridge().fit(prof(train_rows), train_rows[BP])
    # Stacking and calibration are fitted on the validation fold, which the network did not train on
    # (it chose the epoch, so these rows lean slightly optimistic).
    stacked = ridge().fit(np.c_[prof(parts['valid']), pred['valid']], parts['valid'][BP])
    out = {
        'model': pred['test'] - truth,
        'constant': np.tile(train_rows[BP].mean().to_numpy(), (len(truth), 1)) - truth,
        'age, sex, size': profile.predict(prof(parts['test'])) - truth,
        'age, sex, size + model': stacked.predict(np.c_[prof(parts['test']), pred['test']]) - truth,
        'valid MAE': best,
    }
    # Calibrated: the after-squats reading from the before reading plus the camera's estimated change.
    pv, pt = paired(parts['valid'], pred['valid']), paired(parts['test'], pred['test'])
    change = ridge().fit(pv['change'], pv['after'] - pv['before'])
    out['calibrated: + average change'] = pt['before'] + (pv['after'] - pv['before']).mean(0) - pt['after']
    out['calibrated: + model change'] = pt['before'] + change.predict(pt['change']) - pt['after']
    return out


def paired(part, pred):
    """Subjects with both steps: cuff BP before and after, and the model's estimated change."""
    frame = part[['patient_id', 'step']].assign(i=range(len(part)))
    both = frame.pivot(index='patient_id', columns='step', values='i').dropna().astype(int)
    b, a = both['before'].to_numpy(), both['after'].to_numpy()
    bp = part[BP].to_numpy()
    return {'before': bp[b], 'after': bp[a], 'change': pred[a] - pred[b]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='rgb,pos')
    ap.add_argument('--seconds', type=float, default=20)
    ap.add_argument('--stride', type=float, default=2)
    ap.add_argument('--ppg-weight', type=float, default=1.0)
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--folds', default='paper', choices=['paper', 'all'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    args.inputs = args.inputs.split(',')
    rows = table()
    signals = dataset(rows)
    print(f'{len(rows)} recordings, {rows.patient_id.nunique()} subjects')
    pairs = [(0, v) for v in range(1, 5)] if args.folds == 'paper' else [(t, (t + 1) % 5) for t in range(5)]
    results = {}
    for test, valid in pairs:
        r = run(rows, signals, test, valid, args)
        print(f'test {test} valid {valid}:', {k: np.abs(v).mean(0).round(2).tolist() if k != 'valid MAE' else round(float(v), 2)
                                             for k, v in r.items()}, flush=True)
        for k, v in r.items():
            if k != 'valid MAE':
                results.setdefault(k, []).append(v)
    # Paper protocol: four runs on the same test fold are averaged; all folds: errors are pooled.
    summary = {k: scores(np.mean(v, 0) if args.folds == 'paper' else np.concatenate(v)) for k, v in results.items()}
    table_ = pd.DataFrame(summary).T.round(2)
    print(table_.to_markdown())
    name = args.name or f"cnn_{'-'.join(args.inputs)}_{args.seconds:g}s_{args.folds}"
    (RUNS / f'{name}.json').write_text(json.dumps({'args': vars(args), 'summary': summary}, indent=1))


if __name__ == '__main__':
    main()
