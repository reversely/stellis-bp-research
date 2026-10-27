# Stellis camera blood pressure research

Model work for the blood pressure estimate in the Stellis app ([stellis-health-dev](https://github.com/reversely/stellis-health-dev), #231). The app runs a model on the pulse waveform its camera capture produces, and the video stays on the device. This repo trains and tests candidate models. A model goes to the app only if it beats the baselines below on subjects it never saw.

The GB10 workstation is for testing only; this repo makes no claim about where the app runs.

## Baselines to beat

MCD-rPPG, 600 subjects, 5 folds split by subject. Built from the dataset's labels alone, with no video (`uv run python -m stellis_bp_research.baselines`, output in [`results/baselines.md`](results/baselines.md)).

| Estimate | Systolic MAE | Diastolic MAE |
| --- | --- | --- |
| Training average | 13.7 | 7.4 |
| Age, sex, height and weight | 12.5 | 7.3 |
| The same, plus the cuff's pulse | 11.7 | 6.9 |
| After squats, given the reading before: unchanged | 13.1 | 6.0 |
| The same, plus the average change | 10.8 | 5.9 |
| The same, plus the pulse change, age, sex and size | 10.2 | 5.5 |

The training-average row matches the dataset paper's constant baseline (13.78 / 7.50). Their video model scored 12.82 / 8.39, which is worse than age, sex and size alone. A camera model has to add something beyond the pulse the app already measures.

## Plan

1. Run the app's waveform method (vitallens POS) on MCD-rPPG's phone videos (`IriunWebcam`), so models train on the same signal the browser produces.
2. Score candidates against the baselines, in both the absolute and calibrated settings: the app's current refitted LSTM, PaPaGei features with ridge regression, and a BBP-Net-style network on the waveform and its derivatives.
3. Fine-tune the strongest candidate. Export it in the app's `weights.bin` + `manifest.json` format.

## Data and licences

| Source | Licence | Use |
| --- | --- | --- |
| [MCD-rPPG](https://huggingface.co/datasets/kyegorov/mcd_rppg) (Egorov et al. 2025) | CC BY 4.0; its card lists commercial deployment without validation and ethical review as out of scope | Training and testing |
| [PaPaGei](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) | BSD-3-Clause | Pre-trained PPG features |
| [Schrumpf et al. 2021 LSTM](https://doi.org/10.5281/zenodo.5590603) | CC BY 4.0 (the camera refit data is GPL-3.0) | The app's current model |
| MMSE-HR, V4V (BP4D+) | Academic use only | Not used |

Datasets download into `data/` (ignored by git), e.g. `stellis_bp_research.data.fetch(['video/*_IriunWebcam_*'])`.
