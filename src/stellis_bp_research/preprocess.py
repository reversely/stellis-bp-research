"""Face colour signals from the MCD-rPPG phone videos.

For every frame, MediaPipe's face landmarker finds the face and the mean RGB of four skin
regions is recorded with the frame's own timestamp (the frame rate is uneven, as in the app).
Output: data/signals/<video name>.npz with t (s), rgb (frames x regions x 3), found (bool).

Usage: uv run python -m stellis_bp_research.preprocess [workers]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import PHONE, ROOT

OUT = ROOT.parent / 'signals'
MODEL = ROOT.parent / 'models' / 'face_landmarker.task'  # storage.googleapis.com/mediapipe-models/face_landmarker
# MediaPipe face mesh indices; each region is the convex hull of its points.
REGIONS = {
    'face': None,
    'forehead': [10, 109, 67, 103, 338, 297, 332, 151, 108, 69, 337, 299, 9],
    'cheek_a': [116, 117, 118, 119, 100, 47, 126, 209, 49, 129, 203, 205, 50, 123],
    'cheek_b': [345, 346, 347, 348, 329, 277, 355, 429, 279, 358, 423, 425, 280, 352],
}


def landmarker():
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_faces=1))


def region_means(frame, points):
    out = np.full((len(REGIONS), 3), np.nan, np.float32)
    for i, idx in enumerate(REGIONS.values()):
        hull = cv2.convexHull((points if idx is None else points[idx]).astype(np.int32))
        x, y, w, h = cv2.boundingRect(hull)
        x, y = max(x, 0), max(y, 0)
        crop = frame[y:y + h, x:x + w]
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, hull - (x, y), 1)
        if mask.any():
            out[i] = cv2.mean(crop, mask)[:3]
    return out


def timestamps(name):
    meta = pd.read_csv(ROOT / 'meta' / f'{name}.txt', sep=r'\s+', header=None, names=['i', 'date', 'time'])
    t = pd.to_datetime(meta.date + ' ' + meta.time, format='ISO8601')
    return ((t - t.iloc[0]).dt.total_seconds()).to_numpy()


def process(video):
    import mediapipe as mp
    name = Path(video).stem
    out = OUT / f'{name}.npz'
    if out.exists():
        return name, 'done'
    t = timestamps(name)
    cap, detector = cv2.VideoCapture(str(video)), landmarker()
    rgb, found = [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        # Timestamps only need to increase; real times are taken from the meta file.
        result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), len(rgb) * 33)
        if result.face_landmarks:
            pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
            rgb.append(region_means(frame, pts)), found.append(True)
        else:
            rgb.append(np.full((len(REGIONS), 3), np.nan, np.float32)), found.append(False)
    n = min(len(rgb), len(t))
    np.savez_compressed(out, t=t[:n], rgb=np.array(rgb[:n]), found=np.array(found[:n]), frames=len(rgb), stamps=len(t))
    return name, f'{n} frames, face in {np.mean(found):.0%}'


def pin(cores):
    # MediaPipe starts a thread pool per landmarker; one core per worker avoids oversubscription.
    import os
    os.sched_setaffinity(0, {cores.get()})


def main(workers=16):
    from multiprocessing import Manager
    OUT.mkdir(parents=True, exist_ok=True)
    videos = sorted((ROOT / 'video').glob(f'*_{PHONE}_*.avi'))
    cores = Manager().Queue()
    for c in range(int(workers)):
        cores.put(c)
    with Pool(int(workers), pin, (cores,)) as pool:
        for name, status in pool.imap_unordered(process, videos):
            print(name, status, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
