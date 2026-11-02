"""Small probability calibrators fitted only to held-out historical predictions.

The shared calibration model uses score, forecast horizon and their interaction.
It never learns ensemble weights. Exact-zero outcomes are omitted from binary
calibration and scoring, consistently with the earlier sign experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Calibrator:
    model: object | None
    kind: str
    n_calibration: int
    n_positive: int
    n_negative: int
    fallback: bool
    constant_probability: float | None = None
    fallback_reason: str | None = None


def _scores(values, kind):
    if kind not in ('regression', 'classifier'):
        raise ValueError("kind musí být 'regression' nebo 'classifier'.")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 10 or not len(values):
        raise ValueError('Skóre musí mít tvar (počet intervalů, 10).')
    if not np.isfinite(values).all():
        raise ValueError('Skóre musí obsahovat pouze konečná čísla.')
    if kind == 'classifier' and ((values < 0).any() or (values > 1).any()):
        raise ValueError('Pravděpodobnosti klasifikátoru musí být v rozsahu 0 až 1.')
    return values


def _features(scores, kind):
    values = _scores(scores, kind)
    score = values.ravel()
    if kind == 'classifier':
        probability = np.clip(score, 1e-6, 1-1e-6)
        score = np.log(probability)-np.log1p(-probability)
    horizon = np.tile((np.arange(1, 11)-5.5)/4.5, len(values))
    features = np.column_stack((score, horizon, score*horizon))
    if not np.isfinite(features).all():
        raise ValueError('Transformace skóre není numericky konečná.')
    return features


def fit_calibrator(scores, truth, kind='regression'):
    """Fit a fixed L2 logistic mapping to a past calibration day, never test data.

    The feature scaler is fitted only on calibration observations with a nonzero
    actual value. A one-class calibration set uses a Beta(1,1)-smoothed frequency
    rather than attempting an unidentified logistic fit.
    """
    scores = _scores(scores, kind)
    actual = np.asarray(truth, dtype=float)
    if actual.shape != scores.shape or not np.isfinite(actual).all():
        raise ValueError('Skutečnosti musí být konečné a mít stejný tvar jako skóre.')
    mask = actual.ravel() != 0
    count = int(mask.sum())
    if not count:
        raise ValueError('Kalibrace vyžaduje alespoň jednu nenulovou skutečnou hodnotu.')
    labels = (actual.ravel()[mask] > 0).astype(int)
    n_positive = int(labels.sum())
    n_negative = count-n_positive
    if n_positive == 0 or n_negative == 0:
        return Calibrator(None, kind, count, n_positive, n_negative, True,
                          constant_probability=(n_positive+1)/(count+2),
                          fallback_reason='Kalibrační historie obsahuje pouze jedno znaménko.')
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, max_iter=2000, solver='lbfgs', class_weight=None))
    model.fit(_features(scores, kind)[mask], labels)
    return Calibrator(model, kind, count, n_positive, n_negative, False)


def apply_calibrator(calibrator, scores):
    """Return P(positive) for each of the ten future minutes."""
    values = _scores(scores, calibrator.kind)
    if calibrator.fallback:
        p = calibrator.constant_probability
        if p is None or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Neplatná konstantní kalibrační pravděpodobnost.')
        return np.full(values.shape, p, dtype=float)
    if calibrator.model is None:
        raise ValueError('Kalibrátor neobsahuje naučený model.')
    prediction = calibrator.model.predict_proba(_features(values, calibrator.kind))[:, 1]
    prediction = prediction.reshape(values.shape)
    if not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
        raise ValueError('Kalibrátor vrátil neplatné pravděpodobnosti.')
    return prediction


def _binary_arrays(truth, p):
    actual, probability = np.asarray(truth, dtype=float), np.asarray(p, dtype=float)
    if actual.shape != probability.shape or actual.ndim not in (1, 2) or not actual.size:
        raise ValueError('Skutečnosti a pravděpodobnosti musí mít shodný neprázdný tvar.')
    if not np.isfinite(actual).all() or not np.isfinite(probability).all():
        raise ValueError('Skutečnosti a pravděpodobnosti musí být konečné.')
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError('Pravděpodobnosti musí být v rozsahu 0 až 1.')
    mask = actual.ravel() != 0
    return actual.ravel()[mask], probability.ravel()[mask]


def reliability_table(truth, p, bins=10):
    """Fixed-width reliability bins with counts, omitting exact-zero outcomes."""
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError('Počet kalibračních intervalů musí být kladné celé číslo.')
    actual, probability = _binary_arrays(truth, p)
    index = np.minimum((probability*bins).astype(int), bins-1)
    rows = []
    for number in range(bins):
        mask = index == number
        count = int(mask.sum())
        p_mean = float(probability[mask].mean()) if count else np.nan
        observed = float((actual[mask] > 0).mean()) if count else np.nan
        rows.append(dict(bin=number, lower=number/bins, upper=(number+1)/bins,
                         count=count, p_mean=p_mean, positive_rate=observed,
                         abs_gap=abs(p_mean-observed) if count else np.nan))
    return pd.DataFrame(rows)


def binary_scores(truth, p):
    """Brier, log-loss, directional accuracy and fixed-ten-bin ECE.

    A probability exactly .5 produces neutral sign 0, which is incorrect for a
    nonzero actual outcome. Clipping is used solely to keep log-loss finite;
    Brier and reliability retain the original probabilities, including 0 and 1.
    """
    actual, probability = _binary_arrays(truth, p)
    count = len(actual)
    if not count:
        return dict(count=0, brier=np.nan, log_loss=np.nan, accuracy_pct=np.nan, ece=np.nan)
    labels = (actual > 0).astype(float)
    bounded = np.clip(probability, 1e-12, 1-1e-12)
    sign = np.where(probability > .5, 1, np.where(probability < .5, -1, 0))
    bins = reliability_table(actual, probability)
    ece = float((bins['count']*bins.abs_gap.fillna(0)).sum()/count)
    return dict(count=count, brier=float(np.mean((probability-labels)**2)),
                log_loss=float(-np.mean(labels*np.log(bounded)+(1-labels)*np.log1p(-bounded))),
                accuracy_pct=float(100*np.mean(sign == np.sign(actual))), ece=ece)


def calibration_metrics(truth, p):
    """Alias for callers that need the ECE alongside the proper scoring rules."""
    return binary_scores(truth, p)
