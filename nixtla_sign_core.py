"""Real Nixtla MLForecast regressors and direct LightGBM sign classifiers.

Both models use only the five known readings at the forecast origin. Each
historical quarter is a separate series; no lag crosses a missing interval.
Regression-to-probability conversion and calibration belong to the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import pandas as pd


FEATURE_NAMES = tuple(f'minute_{i}' for i in range(5)) + (
    'mean5', 'std5', 'slope5', 'difference5', 'minimum5', 'maximum5',
)


def _observed_array(observed):
    values = np.asarray(observed, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 5 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Očekává se pět konečných známých hodnot pro každý interval.')
    return values


def make_features(observed):
    """Five levels plus summaries computed exclusively from those five levels."""
    values = _observed_array(observed)
    return np.column_stack((
        values, values.mean(axis=1), values.std(axis=1),
        values @ np.arange(-2, 3, dtype=float) / 10,
        values[:, -1] - values[:, 0], values.min(axis=1), values.max(axis=1),
    ))


def _panel(values, prefix):
    count, width = values.shape
    return pd.DataFrame({
        'unique_id': np.repeat([f'{prefix}_{i:08d}' for i in range(count)], width),
        'ds': np.tile(np.arange(width, dtype=np.int64), count),
        'y': values.ravel(),
    })


@dataclass(frozen=True)
class ConstantProbability:
    """Beta(1,1)-smoothed fallback when a horizon has fewer than two classes."""
    probability: float

    def predict_proba(self, features):
        return np.tile([1-self.probability, self.probability], (len(features), 1))


@dataclass
class BaseModels:
    mlforecast_model: object
    classifiers: list
    feature_names: tuple
    n_training_blocks: int
    regression_features: pd.DataFrame
    regression_targets: np.ndarray
    classifier_training_counts: tuple
    seed: int


def _fixed_parameters(seed):
    # Chosen before outer test evaluation. No early stopping or test-set tuning.
    return dict(n_estimators=100, learning_rate=.05, num_leaves=7, max_depth=3,
                min_child_samples=15, reg_lambda=1., verbosity=-1,
                n_jobs=1, random_state=seed, deterministic=True, force_col_wise=True)


def fit_base(blocks, seed=20261102):
    """Fit ten direct MLForecast regressors and ten binary classifiers.

    Ordinary MLForecast.fit(max_horizon=10) can train the shorter horizons at
    later origins too. We explicitly retain only ds=5 after preprocessing, so
    EVERY horizon uses exactly minutes 00–04 to predict minutes 05–14.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor
    from mlforecast import MLForecast

    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 2 or blocks.shape[1] != 15 or not len(blocks) or not np.isfinite(blocks).all():
        raise ValueError('Historie musí obsahovat úplné čtvrthodiny po 15 konečných hodnotách.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Seed musí být nezáporné celé číslo.')
    seed = int(seed)
    parameters = _fixed_parameters(seed)
    forecaster = MLForecast(models={'LGBM': LGBMRegressor(**parameters)}, freq=1,
                            lags=[1, 2, 3, 4, 5], num_threads=1)
    prepared = forecaster.preprocess(_panel(blocks, 'train'), static_features=[],
                                     max_horizon=10, return_X_y=False)
    origins = prepared.loc[prepared.ds.eq(5)].sort_values('unique_id')
    target_columns = [f'y{i}' for i in range(10)]
    features = origins[forecaster.ts.features_order_].reset_index(drop=True)
    targets = origins[target_columns].to_numpy()
    if (len(origins) != len(blocks) or targets.shape != (len(blocks), 10)
            or features.shape != (len(blocks), 5)
            or not np.array_equal(targets, blocks[:, 5:])
            or not np.array_equal(features.to_numpy(), blocks[:, 4::-1])):
        raise RuntimeError('MLForecast neposkytl očekávané zarovnání pěti vstupů a deseti cílů.')

    # Public fit_models changed from matrix targets to a generator factory in
    # recent MLForecast versions. Both branches train the actual Nixtla models.
    if 'generator_factory' in inspect.signature(forecaster.fit_models).parameters:
        def generator_factory():
            for horizon in range(10):
                yield horizon, features, targets[:, horizon]
        forecaster.fit_models(generator_factory=generator_factory)
    else:
        forecaster.fit_models(features, targets)

    known_features = pd.DataFrame(make_features(blocks[:, :5]), columns=FEATURE_NAMES)
    classifiers, counts = [], []
    for horizon in range(10):
        target = blocks[:, horizon+5]
        nonzero = target != 0
        labels = (target[nonzero] > 0).astype(int)
        counts.append(int(nonzero.sum()))
        if len(np.unique(labels)) < 2:
            classifiers.append(ConstantProbability(float((labels.sum()+1)/(len(labels)+2))))
        else:
            classifier = LGBMClassifier(**parameters)
            classifier.fit(known_features.loc[nonzero], labels)
            classifiers.append(classifier)
    return BaseModels(forecaster, classifiers, FEATURE_NAMES, len(blocks),
                      features, targets.copy(), tuple(counts), seed)


def predict_base(models, observed):
    """Return raw MW forecasts and raw sign probabilities, always shape(n,10)."""
    values = _observed_array(observed)
    known = _panel(values, 'predict')
    predicted = models.mlforecast_model.predict(h=10, new_df=known)
    predicted = predicted.sort_values(['unique_id', 'ds'])
    expected_ids = np.repeat([f'predict_{i:08d}' for i in range(len(values))], 10)
    expected_times = np.tile(np.arange(5, 15), len(values))
    if (len(predicted) != len(values)*10
            or not np.array_equal(predicted.unique_id.to_numpy(), expected_ids)
            or not np.array_equal(predicted.ds.to_numpy(), expected_times)):
        raise RuntimeError('MLForecast vrátil jiné intervaly nebo horizonty, než byly požadovány.')
    regression = predicted['LGBM'].to_numpy().reshape(len(values), 10)
    features = pd.DataFrame(make_features(values), columns=models.feature_names)
    probabilities = np.column_stack([model.predict_proba(features)[:, 1]
                                     for model in models.classifiers])
    if (not np.isfinite(regression).all() or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError('Model vrátil nekonečnou předpověď nebo neplatnou pravděpodobnost.')
    return {'regression_mw': regression, 'classifier_raw': probabilities}
