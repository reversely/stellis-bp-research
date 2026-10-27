"""Baselines any camera model must beat, from the MCD-rPPG labels alone.

Absolute: estimate a subject's BP with no reading of theirs.
Calibrated: given their cuff reading before squats, estimate the one after.
Folds are split by subject, so no subject is in both training and test.

Usage: uv run python -m stellis_bp_research.baselines
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from .data import BP, labels

FOLDS = 5
PROFILE = ['age', 'female', 'bmi', 'weight', 'height']


def scores(err):
    err = np.asarray(err)
    return {f'{s} {k}': v for i, s in enumerate(('SBP', 'DBP')) for k, v in (
        ('MAE', np.abs(err[:, i]).mean()), ('SDE', err[:, i].std()), ('within 10', (np.abs(err[:, i]) <= 10).mean()))}


def cross_validate(frame, target, fit_predict):
    err = np.empty((len(frame), 2))
    for train, test in GroupKFold(FOLDS, shuffle=True, random_state=0).split(frame, groups=frame.patient_id):
        err[test] = fit_predict(frame.iloc[train], frame.iloc[test]) - target.iloc[test].to_numpy()
    return scores(err)


def constant(train, test):
    return np.tile(train[BP].mean().to_numpy(), (len(test), 1))


def ridge(columns, offset=None):
    def fit_predict(train, test):
        base = (lambda f: f[offset].to_numpy()) if offset else (lambda f: 0)
        model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(train[columns], train[BP].to_numpy() - base(train))
        return model.predict(test[columns]) + base(test)
    return fit_predict


def main():
    rows = labels().assign(female=lambda f: (f.sex == 'F').astype(float))
    results = {}
    # Absolute: every step of every subject; folds keep both steps of a subject together.
    target = rows[BP]
    results['absolute: constant'] = cross_validate(rows, target, constant)
    results['absolute: age, sex, size'] = cross_validate(rows, target, ridge(PROFILE))
    results['absolute: + cuff pulse'] = cross_validate(rows, target, ridge(PROFILE + ['pulse']))

    # Calibrated: the after step, knowing the before step's cuff reading and pulse.
    before = rows[rows.step == 'before'].set_index('patient_id')
    after = rows[rows.step == 'after'].set_index('patient_id').join(
        before[BP + ['pulse']].add_prefix('before_'), how='inner').reset_index()
    after['pulse_change'] = after.pulse - after.before_pulse
    cal = ['before_upper_ap', 'before_lower_ap']
    target = after[BP]
    results['calibrated: unchanged'] = cross_validate(after, target, lambda tr, te: te[cal].to_numpy())
    results['calibrated: + average change'] = cross_validate(
        after, target, lambda tr, te: te[cal].to_numpy() + (tr[BP].to_numpy() - tr[cal].to_numpy()).mean(0))
    results['calibrated: + pulse change'] = cross_validate(after, target, ridge(cal + ['pulse_change'] + PROFILE))

    table = pd.DataFrame(results).T
    pct = [c for c in table if c.endswith('within 10')]
    table[pct] = (table[pct] * 100).round().astype(int).astype(str) + '%'
    print(f'{rows.patient_id.nunique()} subjects, {FOLDS} folds split by subject')
    print(table.round(1).to_markdown())


if __name__ == '__main__':
    main()
