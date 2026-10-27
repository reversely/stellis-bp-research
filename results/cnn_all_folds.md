# Camera CNN on MCD-rPPG phone video, 5 folds split by subject (authors' folds)

`uv run python -m stellis_bp_research.train --folds all`; per-fold lines and the pooled table.

- test 0 valid 1: {'model': [13.62, 7.35], 'constant': [14.04, 7.57], 'age, sex, size': [13.15, 7.55], 'age, sex, size + model': [13.25, 7.64], 'valid MAE': 9.2, 'calibrated: + average change': [11.87, 6.01], 'calibrated: + model change': [11.86, 6.02]}
- test 1 valid 2: {'model': [12.25, 7.07], 'constant': [13.25, 7.1], 'age, sex, size': [12.25, 7.02], 'age, sex, size + model': [12.2, 6.98], 'valid MAE': 9.48, 'calibrated: + average change': [10.7, 5.61], 'calibrated: + model change': [10.7, 5.61]}
- test 2 valid 3: {'model': [12.25, 7.12], 'constant': [13.36, 7.43], 'age, sex, size': [11.92, 7.2], 'age, sex, size + model': [12.73, 7.42], 'valid MAE': 10.56, 'calibrated: + average change': [10.66, 6.61], 'calibrated: + model change': [10.6, 6.62]}
- test 3 valid 4: {'model': [13.8, 7.7], 'constant': [14.6, 7.64], 'age, sex, size': [12.73, 7.41], 'age, sex, size + model': [12.58, 7.5], 'valid MAE': 9.54, 'calibrated: + average change': [10.69, 6.13], 'calibrated: + model change': [10.65, 6.12]}
- test 4 valid 0: {'model': [12.49, 6.7], 'constant': [13.31, 7.01], 'age, sex, size': [12.74, 7.21], 'age, sex, size + model': [12.56, 7.11], 'valid MAE': 10.3, 'calibrated: + average change': [10.08, 5.26], 'calibrated: + model change': [10.33, 5.23]}
|                              |   SBP MAE |   SBP SDE |   SBP within 10 |   DBP MAE |   DBP SDE |   DBP within 10 |
|:-----------------------------|----------:|----------:|----------------:|----------:|----------:|----------------:|
| model                        |     12.88 |     16.54 |            0.49 |      7.19 |      9.1  |            0.73 |
| constant                     |     13.71 |     17.47 |            0.44 |      7.35 |      9.26 |            0.73 |
| age, sex, size               |     12.56 |     16.2  |            0.48 |      7.28 |      9.24 |            0.73 |
| age, sex, size + model       |     12.67 |     16.42 |            0.49 |      7.33 |      9.28 |            0.73 |
| calibrated: + average change |     10.8  |     13.82 |            0.52 |      5.92 |      7.96 |            0.84 |
| calibrated: + model change   |     10.83 |     13.88 |            0.54 |      5.92 |      7.96 |            0.84 |
