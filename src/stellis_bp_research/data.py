"""MCD-rPPG labels (Egorov et al. 2025, CC BY 4.0, huggingface.co/datasets/kyegorov/mcd_rppg)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / 'data' / 'mcd'
REPO = 'kyegorov/mcd_rppg'
# IriunWebcam is a phone streaming as a webcam, the closest match to the app's capture.
PHONE = 'IriunWebcam'
BP = ['upper_ap', 'lower_ap']


def labels():
    """One row per subject and step (before or after squats); BP is the same for all three cameras."""
    db = pd.read_csv(ROOT / 'db.csv')
    return db[db.camera == PHONE].reset_index(drop=True)


def fetch(patterns):
    """Downloads matching files, e.g. ['video/*_IriunWebcam_*.avi', 'ppg_sync/*_IriunWebcam_*'], into data/mcd."""
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, repo_type='dataset', local_dir=ROOT, allow_patterns=patterns)
