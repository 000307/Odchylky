"""Alignment, probability scoring and chronological checks on artificial data."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nixtla_sign_core import ConstantProbability, fit_base, make_features, predict_base
from sign_calibration import apply_calibrator, binary_scores, fit_calibrator, reliability_table


def synthetic_blocks(count=60, seed=71):
    rng = np.random.default_rng(seed)
    observed = rng.normal(0, 50, size=(count, 5))
    future = observed[:, -1:] + rng.normal(0, 40, size=(count, 10))
    return np.column_stack([observed, future])


class NixtlaCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = synthetic_blocks()
        # Two deliberately single-class horizons, with a true zero excluded.
        cls.blocks[:, 5] = np.arange(1, 61)
        cls.blocks[0, 5] = 0
        cls.blocks[:, 6] = -np.arange(1, 61)
        cls.models = fit_base(cls.blocks, seed=123)

    def test_real_mlforecast_uses_exact_first_five_origin_for_every_horizon(self):
        from mlforecast import MLForecast
        self.assertIsInstance(self.models.mlforecast_model, MLForecast)
        self.assertEqual(self.models.n_training_blocks, 60)
        # This detects ordinary direct-forecast training on later origins.
        self.assertEqual(self.models.regression_features.shape, (60, 5))
        np.testing.assert_array_equal(self.models.regression_features.to_numpy(), self.blocks[:, 4::-1])
        np.testing.assert_array_equal(self.models.regression_targets, self.blocks[:, 5:])

    def test_all_ten_forecasts_preserve_requested_interval_order(self):
        known = np.array([[1, 2, 3, 4, 5], [-9, -8, -7, -6, -5]], dtype=float)
        batch = predict_base(self.models, known)
        for key in ('regression_mw', 'classifier_raw'):
            self.assertEqual(batch[key].shape, (2, 10))
            self.assertTrue(np.isfinite(batch[key]).all())
            first = predict_base(self.models, known[0])[key]
            second = predict_base(self.models, known[1])[key]
            np.testing.assert_allclose(batch[key], np.concatenate([first, second]), rtol=0, atol=1e-10)
        self.assertTrue(((batch['classifier_raw'] >= 0) & (batch['classifier_raw'] <= 1)).all())

    def test_classifier_excludes_zeros_and_smooths_single_class_horizons(self):
        self.assertEqual(self.models.classifier_training_counts[:2], (59, 60))
        self.assertIsInstance(self.models.classifiers[0], ConstantProbability)
        self.assertIsInstance(self.models.classifiers[1], ConstantProbability)
        output = predict_base(self.models, self.blocks[:2, :5])['classifier_raw']
        np.testing.assert_allclose(output[:, 0], 60/61)
        np.testing.assert_allclose(output[:, 1], 1/62)

    def test_features_depend_only_on_the_five_known_values(self):
        known = np.array([[10, 20, 30, 40, 50]], dtype=float)
        features = make_features(known)
        self.assertEqual(features.shape, (1, 11))
        np.testing.assert_array_equal(features[0, :5], known[0])
        np.testing.assert_allclose(features[0, 5:], [30, np.std(known), 10, 40, 10, 50])
        for invalid in (np.ones(4), np.ones(6), np.ones((2, 15)), [1, 2, np.nan, 4, 5]):
            with self.subTest(input_shape=np.asarray(invalid).shape):
                with self.assertRaises(ValueError):
                    make_features(invalid)

    def test_invalid_training_data_and_seed_are_rejected(self):
        for invalid in (np.ones((3, 14)), np.empty((0, 15)), np.full((3, 15), np.inf)):
            with self.assertRaises(ValueError):
                fit_base(invalid)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                fit_base(self.blocks, seed=seed)


class ProbabilityCalibrationTests(unittest.TestCase):
    def test_brier_and_log_loss_match_hand_calculation_excluding_actual_zero(self):
        actual = np.array([-1, 1, 0, -1, 1])
        p = np.array([.2, .8, .99, .1, .6])
        result = binary_scores(actual, p)
        self.assertEqual(result['count'], 4)
        self.assertAlmostEqual(result['brier'], .0625)
        self.assertAlmostEqual(result['log_loss'], -np.mean(np.log([.8, .8, .9, .6])))
        self.assertEqual(result['accuracy_pct'], 100)

    def test_exact_probability_endpoints_and_bin_counts_are_preserved(self):
        actual = np.array([-1, 1, 1, 1, 0])
        p = np.array([0, .1, .999, 1, .4])
        bins = reliability_table(actual, p)
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins['count'].sum(), 4)
        self.assertEqual(bins.loc[0, 'count'], 1)
        self.assertEqual(bins.loc[1, 'count'], 1)
        self.assertEqual(bins.loc[9, 'count'], 2)
        self.assertAlmostEqual(binary_scores(actual, p)['ece'], .22525)
        bad = binary_scores([1, -1], [0, 1])
        self.assertEqual(bad['brier'], 1)
        self.assertTrue(np.isfinite(bad['log_loss']))

    def test_half_probability_is_neutral_and_not_a_correct_nonzero_sign(self):
        result = binary_scores([-1, 1], [.5, .5])
        self.assertEqual(result['accuracy_pct'], 0)
        self.assertEqual(result['brier'], .25)

    def test_single_class_calibration_uses_smoothed_constant_probability(self):
        scores = np.linspace(-10, 10, 40).reshape(4, 10)
        truth = -np.ones((4, 10))
        truth[0, 0] = 0
        fitted = fit_calibrator(scores, truth, kind='regression')
        self.assertTrue(fitted.fallback)
        self.assertEqual(fitted.n_calibration, 39)
        self.assertEqual(fitted.n_positive, 0)
        self.assertEqual(fitted.n_negative, 39)
        np.testing.assert_allclose(apply_calibrator(fitted, scores), 1/41)

    def test_zero_outcomes_cannot_influence_calibration_scaling_or_model(self):
        rng = np.random.default_rng(25)
        scores = rng.normal(size=(25, 10))
        truth = np.where(scores + rng.normal(size=scores.shape) > 0, 1., -1.)
        truth[:4, 2] = 0
        changed = scores.copy()
        changed[truth == 0] = 1e9
        first = fit_calibrator(scores, truth, kind='regression')
        second = fit_calibrator(changed, truth, kind='regression')
        np.testing.assert_allclose(apply_calibrator(first, scores), apply_calibrator(second, scores),
                                   rtol=0, atol=1e-12)
        self.assertEqual(first.n_calibration, 246)

    def test_classifier_logit_handles_zero_one_and_returns_ten_probabilities(self):
        p = np.tile(np.linspace(0, 1, 10), (20, 1))
        truth = np.where(p > .4, 1., -1.)
        truth[::2] *= -1
        fitted = fit_calibrator(p, truth, kind='classifier')
        actual = apply_calibrator(fitted, p[:3])
        self.assertEqual(actual.shape, (3, 10))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        self.assertIsNone(fitted.model.named_steps['logisticregression'].class_weight)

    def test_invalid_probabilities_calibration_shapes_and_all_zero_fit_are_rejected(self):
        for p in ([.2, 1.1], [.2, np.nan], [-.1, .2]):
            with self.assertRaises(ValueError):
                binary_scores([1, -1], p)
        for scores, truth, kind in [
                (np.ones((2, 9)), np.ones((2, 9)), 'regression'),
                (np.ones((2, 10)), np.ones((3, 10)), 'regression'),
                (np.ones((2, 10)), np.zeros((2, 10)), 'regression'),
                (np.full((2, 10), 2.), np.ones((2, 10)), 'classifier')]:
            with self.assertRaises(ValueError):
                fit_calibrator(scores, truth, kind=kind)
        result = binary_scores([0, 0], [.4, .6])
        self.assertEqual(result['count'], 0)
        self.assertTrue(np.isnan(result['brier']))


class ChronologicalWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import predikce_nixtla_znamenka
        cls.runner = predikce_nixtla_znamenka

    def setUp(self):
        self.times = pd.DatetimeIndex([
            day + pd.Timedelta(minutes=15*quarter)
            for day in pd.date_range('2026-09-23', periods=4, freq='D')
            for quarter in range(8)
        ])
        self.blocks = synthetic_blocks(len(self.times), seed=122)

    def run_mocked(self, blocks, *, live=False, restrict_history=False):
        fitted_histories, calibration_targets, observed_batches = [], [], []
        real_calibrator = self.runner.fit_calibrator

        def fit_spy(training, seed):
            fitted_histories.append(training.copy())
            return {'training_mean': training[:, 5:].mean()}

        def predict_spy(model, observed):
            observed_batches.append(observed.copy())
            values = (observed[:, -1:] + .1*model['training_mean']
                      + np.arange(10)[None, :])
            probability = 1/(1+np.exp(-values/25))
            return {'regression_mw': values, 'classifier_raw': probability}

        def calibration_spy(scores, truth, kind):
            calibration_targets.append((kind, truth.copy()))
            return real_calibrator(scores, truth, kind=kind)

        with patch.object(self.runner, 'fit_base', side_effect=fit_spy), \
                patch.object(self.runner, 'predict_base', side_effect=predict_spy), \
                patch.object(self.runner, 'fit_calibrator', side_effect=calibration_spy):
            if live:
                stop = 24 if restrict_history else len(self.times)
                result = self.runner.predict_interval(
                    self.times[:stop], blocks[:stop], self.blocks[24, :5], self.times[24], seed=7)
            else:
                result = self.runner.backtest(self.times, blocks, initial_train_days=3, seed=7)
        return result, fitted_histories, calibration_targets, observed_batches

    def test_base_calibration_and_test_days_remain_separate_without_refitting(self):
        result, histories, calibrations, observed = self.run_mocked(self.blocks)
        self.assertEqual(len(histories), 1)
        np.testing.assert_array_equal(histories[0], self.blocks[:16])
        self.assertEqual([kind for kind, _ in calibrations], ['regression', 'classifier'])
        for _, actual in calibrations:
            np.testing.assert_array_equal(actual, self.blocks[16:24, 5:])
        np.testing.assert_array_equal(observed[0], self.blocks[16:24, :5])
        np.testing.assert_array_equal(observed[1], self.blocks[24:, :5])
        fold = result['folds'].iloc[0]
        self.assertEqual(fold.base_quarters, 16)
        self.assertEqual(fold.calibration_quarters, 8)
        self.assertEqual(fold.test_quarters, 8)
        self.assertLess(fold.base_train_end, fold.calibration_day)
        self.assertLess(fold.calibration_day, fold.test_day)

    def test_modifying_outer_targets_cannot_change_any_forecast(self):
        expected = self.run_mocked(self.blocks)[0]['predictions']
        changed = self.blocks.copy()
        changed[24:, 5:] = 1000000
        actual = self.run_mocked(changed)[0]['predictions']
        self.assertFalse(actual.skutecnost_mw.equals(expected.skutecnost_mw))
        columns = [c for c in actual.columns if c != 'skutecnost_mw']
        pd.testing.assert_frame_equal(actual[columns], expected[columns])

    def test_combination_is_the_fixed_average_and_ten_timestamps_are_correct(self):
        result = self.run_mocked(self.blocks)[0]['predictions']
        wide = result.pivot(index=['start', 'krok'], columns='model', values='p_kladna')
        np.testing.assert_array_equal(wide[self.runner.MIX],
                                      .5*wide[self.runner.MLF]+.5*wide[self.runner.CLF])
        self.assertEqual(result.groupby('model').size().nunique(), 1)
        self.assertEqual(result.groupby('model').size().iloc[0], 80)
        for (start, _), group in result.groupby(['start', 'model']):
            self.assertEqual(list(group.krok), list(range(1, 11)))
            expected = pd.date_range(start+pd.Timedelta(minutes=5), periods=10, freq='min')
            pd.testing.assert_index_equal(pd.DatetimeIndex(group.cas), expected, check_names=False)

    def test_live_forecast_excludes_current_and_future_complete_intervals(self):
        expected, histories, targets, _ = self.run_mocked(self.blocks, live=True, restrict_history=True)
        contaminated = self.blocks.copy()
        contaminated[24:] = -1000000
        actual, _, _, _ = self.run_mocked(contaminated, live=True)
        pd.testing.assert_frame_equal(actual['predictions'], expected['predictions'])
        np.testing.assert_array_equal(histories[0], self.blocks[:16])
        for _, values in targets:
            np.testing.assert_array_equal(values, self.blocks[16:24, 5:])
        self.assertEqual(len(actual['predictions']), 60)

    def test_summary_reliability_and_scores_share_the_same_nonzero_denominator(self):
        blocks = self.blocks.copy()
        blocks[24, 7] = 0
        predictions = self.run_mocked(blocks)[0]['predictions']
        summary = self.runner.summarize(predictions)
        self.assertTrue(summary['metrics']['count'].eq(79).all())
        self.assertTrue(summary['metrics']['pocet_nul'].eq(1).all())
        count = summary['reliability'].groupby('model')['count'].sum()
        self.assertTrue(count.eq(79).all())
        self.assertTrue(summary['reliability'].groupby('model').size().eq(10).all())


if __name__ == '__main__':
    unittest.main()
