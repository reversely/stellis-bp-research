"""Exports 30 s pulse waveform segments for the app's rhythm check (stellis-health-dev #237).

Each line of runs/rhythm_segments.jsonl is {id, dataset, subject, af, t, v}:
- mcd_camera: MCD-rPPG phone video, POS on the whole face, 30 Hz. af is False when the
  recording's 30 s ECG shows a regular rhythm (all are expected to be sinus).
- mcd_finger: the synced finger PPG of the same segments, resampled to 30 Hz.
- mimic: MIMIC PERform AF dataset (Charlton et al., ODbL-1.0), ICU finger PPG at 125 Hz,
  resampled to 30 Hz like a camera; 19 subjects in AF and 16 not.

Usage: uv run python -m stellis_bp_research.rhythm_export
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .data import ROOT
from .signals import RATE, bandpass, recording, table

SECONDS = 30
RUNS = Path(__file__).resolve().parents[2] / 'runs'
MIMIC = ROOT.parent / 'mimic_perform'


def ecg_intervals(name):
    """R-R intervals (s) from lead II of the recording's 30 s ECG."""
    subject, _, step = name.split('_')
    try:
        d = json.load(open(ROOT / 'ecg' / f'{subject}_{step}.json'))
    except json.JSONDecodeError:  # a few files hold two concatenated documents
        return np.array([])
    rate = d['frequency']
    lead = np.array(next(x['values'] for x in d['data'] if x['title'] == 'II'))
    x = bandpass(lead, 5, 25, rate)
    x = x if np.abs(x.max()) >= np.abs(x.min()) else -x
    peaks, _ = find_peaks(x, distance=int(0.3 * rate), height=0.4 * np.percentile(x, 99.5))
    return np.diff(peaks) / rate


def irregular_ecg(intervals):
    """A coarse label: more than 10% of successive intervals differ by over 15%."""
    if len(intervals) < 10:
        return None
    jumps = np.abs(np.diff(intervals)) / intervals[:-1]
    return bool((jumps > 0.15).mean() > 0.1)


def segments(x, rate):
    n = int(SECONDS * rate)
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)]


def line(out, **kw):
    t = np.arange(len(kw['v'])) / RATE
    out.write(json.dumps({**kw, 't': np.round(t, 4).tolist(), 'v': np.round(kw['v'], 5).tolist()}) + '\n')


def main():
    RUNS.mkdir(exist_ok=True)
    labels = []
    with open(RUNS / 'rhythm_segments.jsonl', 'w') as out:
        for row in table().itertuples():
            r = recording(row.name)
            ecg_irregular = irregular_ecg(ecg_intervals(row.name))
            labels.append({'name': row.name, 'ecg_irregular': ecg_irregular})
            for i, (cam, ppg) in enumerate(zip(segments(r['pos'][:, 0], RATE), segments(r['ppg'], RATE))):
                base = dict(subject=int(row.patient_id), af=False, ecg_irregular=ecg_irregular)
                line(out, id=f'{row.name}_{i}', dataset='mcd_camera', v=cam, **base)
                line(out, id=f'{row.name}_{i}_finger', dataset='mcd_finger', v=ppg, **base)
        for group, af in (('af', True), ('non_af', False)):
            for f in sorted((MIMIC / f'mimic_perform_{group}_csv' / f'mimic_perform_{group}_csv').glob('*_data.csv')):
                d = pd.read_csv(f)
                t = d.Time.to_numpy()
                grid = np.arange(t[0], t[-1], 1 / RATE)
                ppg = bandpass(np.interp(grid, t, d.PPG.to_numpy()), 0.5, 8)
                for i, seg in enumerate(segments(ppg, RATE)):
                    line(out, id=f'{f.stem}_{i}', dataset='mimic', subject=f.stem, af=af, ecg_irregular=af, v=seg)
    pd.DataFrame(labels).to_csv(RUNS / 'mcd_ecg_rhythm.csv', index=False)
    print(pd.DataFrame(labels).ecg_irregular.value_counts(dropna=False))


if __name__ == '__main__':
    main()
