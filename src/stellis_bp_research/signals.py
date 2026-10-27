"""Windows of camera signals and finger PPG, resampled from each frame's own timestamp."""
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from .data import ROOT, labels
from .preprocess import OUT

RATE = 30
THIRD_PARTY = Path(__file__).resolve().parents[2] / 'third_party'


def folds():
    """The dataset authors' subject folds (github.com/ksyegorov/mcd_rppg, mcd_rppg.csv)."""
    f = pd.read_csv(THIRD_PARTY / 'mcd_rppg' / 'mcd_rppg.csv')
    return f.groupby('patient_id').fold.first()


def bandpass(x, low, high, rate=RATE):
    return sosfiltfilt(butter(2, [low, high], 'band', fs=rate, output='sos'), x, axis=0)


def pos(rgb, rate=RATE):
    """Wang et al. 2017 plane-orthogonal-to-skin pulse from (n, 3) RGB."""
    n, w = len(rgb), int(1.6 * rate)
    h = np.zeros(n)
    proj = np.array([[0, 1, -1], [-2, 1, 1]])
    for s in range(n - w + 1):
        c = rgb[s:s + w] / rgb[s:s + w].mean(0)
        x = c @ proj.T
        p = x[:, 0] + x[:, 0].std() / (x[:, 1].std() + 1e-9) * x[:, 1]
        h[s:s + w] += p - p.mean()
    return h


def stamps(path):
    f = pd.read_csv(path, sep=r'\s+', header=None, names=['v', 'date', 'time'])
    t = pd.to_datetime(f.date + ' ' + f.time, format='ISO8601')
    return t, f.v.to_numpy(float)


def finger(name):
    """The 100 Hz finger PPG on the video's clock (both files carry wall-clock times)."""
    subject, _, step = name.split('_')
    video_t, _ = stamps(ROOT / 'meta' / f'{name}.txt')
    t, v = stamps(ROOT / 'ppg' / f'{subject}_{step}.PW')
    return (t - video_t.iloc[0]).dt.total_seconds().to_numpy(), v


@cache
def recording(name):
    """Uniform-rate signals for one video: rgb (n, regions*3) normalised colour, pos (n, regions), ppg (n,)."""
    d = np.load(OUT / f'{name}.npz')
    t, rgb, found = d['t'], d['rgb'].copy(), d['found']
    # A region can leave the frame (the far cheek in the side view); the whole face stands in.
    for r in range(1, rgb.shape[1]):
        if np.isnan(rgb[:, r, 0]).mean() > 0.5:
            rgb[:, r] = rgb[:, 0]
    rgb = rgb.reshape(len(t), -1)
    grid = np.arange(t[0], t[-1], 1 / RATE)
    rgb = np.stack([np.interp(grid, t[ok], c[ok]) for c in rgb.T for ok in [found & ~np.isnan(c)]], 1)
    ppg = np.interp(grid, *finger(name))
    regions = rgb.reshape(len(grid), -1, 3)
    pulse = np.stack([bandpass(pos(r), 0.5, 8) for r in regions.transpose(1, 0, 2)], 1)
    colour = bandpass(rgb / uniform_filter1d(rgb, RATE, axis=0) - 1, 0.5, 8)
    return {'rgb': colour, 'pos': pulse, 'ppg': bandpass(ppg, 0.5, 8), 'found': found.mean()}


def windows(x, seconds, stride):
    w, s = int(seconds * RATE), int(stride * RATE)
    return np.stack([x[i:i + w] for i in range(0, len(x) - w + 1, s)])


def table():
    """Phone recordings with signals, labels and the authors' fold."""
    rows = labels()
    rows['name'] = rows.video.map(lambda v: Path(v).stem)
    rows = rows[rows.name.map(lambda n: (OUT / f'{n}.npz').exists())]
    return rows.assign(fold=rows.patient_id.map(folds())).dropna(subset=['fold']).reset_index(drop=True)
