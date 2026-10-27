# Paper protocol (fold 0 test, 4 validation folds averaged)

- test 0 valid 1: {'model': [13.71, 7.5], 'constant': [14.04, 7.57], 'age, sex, size': [13.15, 7.55], 'valid MAE': np.float64(9.27)}
- test 0 valid 2: {'model': [13.91, 7.54], 'constant': [14.04, 7.57], 'age, sex, size': [13.15, 7.55], 'valid MAE': np.float64(9.47)}
- test 0 valid 3: {'model': [13.48, 7.48], 'constant': [14.04, 7.57], 'age, sex, size': [13.15, 7.55], 'valid MAE': np.float64(10.4)}
- test 0 valid 4: {'model': [13.47, 7.42], 'constant': [14.04, 7.57], 'age, sex, size': [13.15, 7.55], 'valid MAE': np.float64(9.36)}
|                |   SBP MAE |   SBP SDE |   SBP within 10 |   DBP MAE |   DBP SDE |   DBP within 10 |
|:---------------|----------:|----------:|----------------:|----------:|----------:|----------------:|
| model          |     13.49 |     16.77 |            0.46 |      7.42 |      9.22 |            0.76 |
| constant       |     14.04 |     18    |            0.46 |      7.57 |      9.47 |            0.73 |
| age, sex, size |     13.15 |     16.93 |            0.46 |      7.55 |      9.56 |            0.73 |
