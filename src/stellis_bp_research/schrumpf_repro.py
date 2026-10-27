"""Reproduces Schrumpf et al. 2021, Table 2, LSTM after camera training (13.6 / 10.3 mmHg).

Their protocol: only the final layer is trained (Adam, lr 0.001, until the error stops falling),
15 of the 17 rPPG-BP-UKL subjects train it, one held-out subject is used to stop and one to test.
Also reports the app's ridge refit and the guess-the-average baseline on the same test subjects.

Inputs in data/schrumpf (see README): lstm_ppg_nonmixed.h5, rPPG-BP-UKL_rppg_7s.h5.
Usage: uv run python -m stellis_bp_research.schrumpf_repro
"""
from pathlib import Path

import h5py
import numpy as np
import torch

from .schrumpf import features, load

DATA = Path(__file__).resolve().parents[2] / 'data' / 'schrumpf'


def head_fit(x, y, xv, yv, w, b):
    """Adam on the SBP/DBP output layers from the published weights, stopped on the validation subject."""
    x, y, xv, yv = (torch.tensor(a, dtype=torch.float32) for a in (x, y, xv, yv))
    w, b = torch.tensor(w, dtype=torch.float32, requires_grad=True), torch.tensor(b, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], 1e-3)
    best, keep, stale = np.inf, None, 0
    for _ in range(5000):
        for idx in torch.randperm(len(x)).split(32):
            loss = torch.nn.functional.mse_loss(x[idx] @ w + b, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mae = (xv @ w + b - yv).abs().mean().item()
        if mae < best - 1e-3:
            best, keep, stale = mae, (w.detach().clone(), b.detach().clone()), 0
        elif (stale := stale + 1) >= 20:
            break
    return keep


def main():
    model = load(DATA / 'lstm_ppg_nonmixed.h5')
    with h5py.File(DATA / 'rPPG-BP-UKL_rppg_7s.h5') as f:
        z = np.array(f['rppg']).T
        bp = np.array(f['label']).T
        subject = np.array(f['subject_idx']).ravel()
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-9)
    h = np.concatenate([features(model, z[i:i + 512]) for i in range(0, len(z), 512)])
    # The final layers act on the dense layer's output.
    d = h @ model['dense'][0] + model['dense'][1]
    w0 = np.c_[model['sbp'][0], model['dbp'][0]]
    b0 = np.r_[model['sbp'][1], model['dbp'][1]]
    published = np.abs(d @ w0 + b0 - bp).mean(0)
    subjects = np.unique(subject)
    err = {'fine-tuned last layer': [], 'ridge (the app)': [], 'average guess': []}
    for i, s in enumerate(subjects):
        v = subjects[(i + 1) % len(subjects)]
        test, valid = subject == s, subject == v
        train = ~(test | valid)
        w, b = head_fit(d[train], bp[train], d[valid], bp[valid], w0, b0)
        err['fine-tuned last layer'].append(np.abs(d[test] @ w.numpy() + b.numpy() - bp[test]))
        a = np.c_[h[~test], np.ones((~test).sum())]
        r = np.linalg.solve(a.T @ a + 100 * np.eye(a.shape[1]), a.T @ bp[~test])
        err['ridge (the app)'].append(np.abs(np.c_[h[test], np.ones(test.sum())] @ r - bp[test]))
        err['average guess'].append(np.abs(bp[~test].mean(0) - bp[test]))
    print(f'{len(subjects)} subjects, {len(z)} windows')
    print('| Model | SBP MAE | DBP MAE |\n| --- | --- | --- |')
    print(f'| Published weights, no camera training (paper: 33.5 / 12.4) | {published[0]:.1f} | {published[1]:.1f} |')
    print('| Paper, after camera training | 13.6 | 10.3 |')
    for k, v in err.items():
        m = np.concatenate(v).mean(0)
        print(f'| {k} | {m[0]:.1f} | {m[1]:.1f} |')


if __name__ == '__main__':
    main()
