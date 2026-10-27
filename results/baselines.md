600 subjects, 5 folds split by subject
|                              |   SBP MAE |   SBP SDE | SBP within 10   |   DBP MAE |   DBP SDE | DBP within 10   |
|:-----------------------------|----------:|----------:|:----------------|----------:|----------:|:----------------|
| absolute: constant           |      13.7 |      17.4 | 43%             |       7.4 |       9.3 | 72%             |
| absolute: age, sex, size     |      12.5 |      16.1 | 50%             |       7.3 |       9.2 | 73%             |
| absolute: + cuff pulse       |      11.7 |      15   | 52%             |       6.9 |       8.8 | 76%             |
| calibrated: unchanged        |      13.1 |      13.7 | 50%             |       6   |       7.9 | 84%             |
| calibrated: + average change |      10.8 |      13.8 | 54%             |       5.9 |       7.9 | 84%             |
| calibrated: + pulse change   |      10.2 |      13   | 57%             |       5.5 |       7.2 | 84%             |
