"""Mathematical and time-boundary checks; fixtures contain no private data."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predikce_linearni_regrese import backtest, fit_model, load_blocks, predict_interval


def synthetic_blocks(count=30, seed=43):
    """Full-rank inputs and a known affine mapping to ten future minutes."""
    rng = np.random.default_rng(seed)
    known = rng.normal(size=(count, 5))
    coefficients = rng.normal(size=(5, 10))
    intercept = np.linspace(-15, 30, 10)
    return np.column_stack((known, known @ coefficients + intercept))


def day_starts(days=4, quarters=6):
    return pd.DatetimeIndex([
        day + pd.Timedelta(minutes=15 * quarter)
        for day in pd.date_range("2026-09-23", periods=days, freq="D")
        for quarter in range(quarters)
    ])


class RegressionTests(unittest.TestCase):
    def test_recovers_ten_affine_targets_from_five_inputs(self):
        blocks = synthetic_blocks()
        model = fit_model(blocks[:24])
        self.assertEqual(model.n_features_in_, 5)
        predicted = model.predict(blocks[24:, :5])
        self.assertEqual(predicted.shape, (6, 10))
        np.testing.assert_allclose(predicted, blocks[24:, 5:], atol=1e-10)

    def test_fit_rejects_too_little_or_nonfinite_history(self):
        blocks = synthetic_blocks()
        for invalid in (blocks[:5], blocks[:, :14],
                        np.full((10, 15), np.inf),
                        np.full((10, 15), np.nan)):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    fit_model(invalid)

    def test_live_prediction_has_exact_ten_minute_timestamps(self):
        times = pd.date_range("2026-09-23", periods=31, freq="15min")
        blocks = synthetic_blocks(31)
        result = predict_interval(times[:30], blocks[:30], blocks[30, :5], times[30])
        expected = pd.date_range(times[30] + pd.Timedelta(minutes=5), periods=10, freq="min")
        pd.testing.assert_index_equal(pd.DatetimeIndex(result["cas"]), expected,
                                      check_names=False)
        np.testing.assert_allclose(result["odhad_mw"], blocks[30, 5:], atol=1e-10)

    def test_live_prediction_ignores_current_and_future_target_values(self):
        times = pd.date_range("2026-09-23", periods=36, freq="15min")
        blocks = synthetic_blocks(36)
        start = times[30]
        known = blocks[30, :5]
        expected = predict_interval(times[:30], blocks[:30], known, start)
        contaminated = blocks.copy()
        contaminated[30:, 5:] = 1e12
        actual = predict_interval(times, contaminated, known, start)
        pd.testing.assert_frame_equal(actual, expected)

    def test_live_prediction_requires_five_finite_known_values(self):
        times = pd.date_range("2026-09-23", periods=30, freq="15min")
        blocks = synthetic_blocks()
        for invalid in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 4, np.inf], [1, 2, np.nan, 4, 5]):
            with self.subTest(observed=invalid):
                with self.assertRaises(ValueError):
                    predict_interval(times, blocks, invalid,
                                     times[-1] + pd.Timedelta(minutes=15))

    def test_live_prediction_cannot_train_on_future_history(self):
        times = pd.date_range("2026-09-23", periods=30, freq="15min")
        blocks = synthetic_blocks()
        with self.assertRaises(ValueError):
            predict_interval(times, blocks, blocks[0, :5], times[0])

    def test_backtest_holds_out_whole_days_and_forecasts_all_ten_targets(self):
        times = day_starts()
        blocks = synthetic_blocks(len(times))
        result = backtest(times, blocks, initial_train_days=3)
        linear = result[result["model"] == "Lineární regrese"].sort_values("start")
        self.assertEqual(len(linear), 6)
        self.assertTrue((linear["start"].dt.normalize() == pd.Timestamp("2026-09-26")).all())
        for minute in range(5, 15):
            np.testing.assert_allclose(linear[f"skutecnost_{minute:02d}"], blocks[-6:, minute])
            np.testing.assert_allclose(linear[f"odhad_{minute:02d}"], blocks[-6:, minute], atol=1e-10)
        np.testing.assert_allclose(linear["skutecny_prumer_mw"], blocks[-6:, 5:].mean(axis=1))
        np.testing.assert_allclose(linear["odhad_prumer_mw"], blocks[-6:, 5:].mean(axis=1), atol=1e-10)

    def test_backtest_never_fits_on_held_out_day_targets(self):
        times = day_starts()
        blocks = synthetic_blocks(len(times))
        expected = backtest(times, blocks, initial_train_days=3)
        contaminated = blocks.copy()
        contaminated[-6:, 5:] += 1e9
        actual = backtest(times, contaminated, initial_train_days=3)
        estimate_columns = ["model", "start", "odhad_prumer_mw"] + [f"odhad_{m:02d}" for m in range(5, 15)]
        pd.testing.assert_frame_equal(actual[estimate_columns], expected[estimate_columns])


class CsvLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_csv(self, name, times, values, *, czech=False):
        rows = pd.DataFrame({
            "Timestamp": [t.strftime("%d.%m.%Y") if czech else t.isoformat() + "+02:00" for t in times],
            "Čas": [t.strftime("%H:%M:%S") for t in times],
            "ODCHYLKA ČR [MW]": [str(v).replace(".", ",") if czech else v for v in values],
        })
        path = self.root / name
        rows.to_csv(path, sep=";" if czech else ",", index=False, encoding="utf-8-sig")
        return path

    def test_skips_incomplete_quarters_without_interpolation(self):
        times = pd.date_range("2026-09-23", periods=30, freq="min").delete(20)
        path = self.write_csv("missing.csv", times, np.arange(len(times)))
        starts, blocks, skipped = load_blocks([path])
        self.assertEqual(list(starts), [pd.Timestamp("2026-09-23")])
        self.assertEqual(blocks.shape, (1, 15))
        self.assertEqual(skipped, 1)
        np.testing.assert_array_equal(blocks[0], np.arange(15))

    def test_identical_duplicates_are_allowed_and_sorted(self):
        times = pd.date_range("2026-09-23", periods=15, freq="min")
        first = self.write_csv("first.csv", times[::-1], np.arange(15)[::-1])
        repeated = self.write_csv("repeated.csv", times, np.arange(15))
        starts, blocks, skipped = load_blocks([first, repeated])
        self.assertEqual(len(starts), 1)
        self.assertEqual(skipped, 0)
        np.testing.assert_array_equal(blocks[0], np.arange(15))

    def test_conflicting_duplicates_are_rejected(self):
        times = pd.date_range("2026-09-23", periods=15, freq="min")
        first = self.write_csv("first.csv", times, np.arange(15))
        conflict = self.write_csv("conflict.csv", times[:1], [999])
        with self.assertRaises(ValueError):
            load_blocks([first, conflict])

    def test_nonfinite_csv_values_are_rejected(self):
        times = pd.date_range("2026-09-23", periods=15, freq="min")
        for bad in (np.inf, -np.inf, np.nan):
            with self.subTest(value=bad):
                values = np.arange(15, dtype=float)
                values[7] = bad
                path = self.write_csv("nonfinite.csv", times, values)
                with self.assertRaises(ValueError):
                    load_blocks([path])

    def test_czech_dates_and_decimal_commas(self):
        times = pd.date_range("2026-09-23", periods=15, freq="min")
        values = np.arange(15) + .25
        path = self.write_csv("czech.csv", times, values, czech=True)
        starts, blocks, skipped = load_blocks([path])
        self.assertEqual(list(starts), [pd.Timestamp("2026-09-23")])
        self.assertEqual(skipped, 0)
        np.testing.assert_array_equal(blocks[0], values)

    def test_no_complete_quarter_is_rejected(self):
        times = pd.date_range("2026-09-23", periods=14, freq="min")
        path = self.write_csv("short.csv", times, np.arange(14))
        with self.assertRaises(ValueError):
            load_blocks([path])


if __name__ == "__main__":
    unittest.main()
