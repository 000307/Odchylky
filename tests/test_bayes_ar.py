"""Checks against analytic distributions and fixed AR(1) predictive moments."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from bayes_ar_core import (
    Posterior,
    diagnostics,
    fit_ar,
    forecast,
    posterior_hyperparameters,
    stationarity_fraction,
)


def synthetic_blocks(count=80, seed=328):
    rng = np.random.default_rng(seed)
    blocks = np.empty((count, 15))
    blocks[:, 0] = rng.normal(50, 15, count)
    for minute in range(1, 15):
        blocks[:, minute] = 12 + .72 * blocks[:, minute - 1] + rng.normal(0, 8, count)
    return blocks


def fixed_posterior(phi=.7, alpha=.1, sigma=1.1, location=0, scale=1, draws=10000):
    return Posterior(
        beta=np.broadcast_to([alpha, phi], (2, draws, 2)).copy(),
        sigma=np.full((2, draws), sigma),
        location=location,
        scale=scale,
        lags=1,
        n_observations=100,
    )


class GibbsPosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = synthetic_blocks()
        cls.posterior = fit_ar(cls.blocks, lags=2, draws=2000, tune=500, chains=4, seed=397)

    def test_training_windows_stay_inside_complete_quarters(self):
        # Two quarters are independent sequences, including at the discontinuity.
        blocks = np.vstack((np.arange(15), 1000 + np.arange(15)))
        hyper = posterior_hyperparameters(blocks, lags=5)
        self.assertEqual(hyper["n_observations"], 2 * (15 - 5))
        self.assertEqual(hyper["lags"], 5)

    def test_hyperparameters_match_independent_conjugate_calculation(self):
        p = self.posterior
        z = (self.blocks - p.location) / p.scale
        # Explicit lag order: most recent observation is first.
        x = np.array([[1, block[t - 1], block[t - 2]]
                      for block in z for t in range(2, 15)])
        y = np.array([block[t] for block in z for t in range(2, 15)])
        prior_mean = np.array([0, .8, 0])
        prior_precision = np.diag([1, 4, 4])
        precision = prior_precision + x.T @ x
        covariance = np.linalg.inv(precision)
        mean = np.linalg.solve(precision, prior_precision @ prior_mean + x.T @ y)
        shape = 2 + len(y) / 2
        residual = y - x @ mean
        deviation = mean - prior_mean
        rate = 1 + .5 * (residual @ residual + deviation @ prior_precision @ deviation)
        hyper = posterior_hyperparameters(self.blocks, lags=2)
        np.testing.assert_allclose(hyper["mn"], mean, atol=1e-11)
        np.testing.assert_allclose(hyper["Vn"], covariance, atol=1e-11)
        self.assertAlmostEqual(hyper["an"], shape)
        self.assertAlmostEqual(hyper["bn"], rate, places=9)

    def test_gibbs_draws_recover_exact_posterior_moments(self):
        p = self.posterior
        hyper = posterior_hyperparameters(self.blocks, lags=2)
        beta = p.beta.reshape(-1, 3)
        sigma2 = p.sigma.reshape(-1) ** 2
        expected_sigma2 = hyper["bn"] / (hyper["an"] - 1)
        expected_beta_variance = expected_sigma2 * np.diag(hyper["Vn"])
        # Tolerances allow Monte Carlo error; these catch wrong conditional
        # shapes/rates or accidentally omitting the beta prior from sigma².
        standard_error = np.sqrt(expected_beta_variance / len(beta))
        self.assertTrue(np.all(np.abs(beta.mean(axis=0) - hyper["mn"])
                               < 7 * standard_error))
        np.testing.assert_allclose(beta.var(axis=0, ddof=1), expected_beta_variance, rtol=.10)
        self.assertAlmostEqual(sigma2.mean() / expected_sigma2, 1, delta=.03)

    def test_small_sample_variance_includes_the_coefficient_prior(self):
        # Here k=6 is large relative to n=20. Omitting the k/2 term in the
        # full conditional would materially inflate the sampled variance.
        blocks = synthetic_blocks(count=2, seed=910)
        posterior = fit_ar(blocks, lags=5, draws=2500, tune=500, chains=4, seed=383)
        hyper = posterior_hyperparameters(blocks, lags=5)
        expected = hyper["bn"] / (hyper["an"] - 1)
        self.assertAlmostEqual(np.mean(posterior.sigma ** 2) / expected, 1, delta=.03)

    def test_separate_chains_and_rank_diagnostics(self):
        p = self.posterior
        self.assertEqual(p.beta.shape, (4, 2000, 3))
        self.assertEqual(p.sigma.shape, (4, 2000))
        self.assertFalse(np.array_equal(p.beta[0], p.beta[1]))
        summary = diagnostics(p)
        for column in ("r_hat", "ess_bulk", "ess_tail", "mean", "sd"):
            self.assertIn(column, summary.columns)
            self.assertTrue(np.isfinite(summary[column]).all())
        self.assertLess(summary.r_hat.max(), 1.03)
        self.assertGreater(summary.ess_bulk.min(), 400)

    def test_seed_reproduces_mcmc_draws(self):
        kwargs = dict(lags=1, draws=200, tune=100, chains=2, seed=888)
        first = fit_ar(self.blocks[:8], **kwargs)
        second = fit_ar(self.blocks[:8], **kwargs)
        np.testing.assert_array_equal(first.beta, second.beta)
        np.testing.assert_array_equal(first.sigma, second.sigma)

    def test_invalid_history_and_sampler_configuration_are_rejected(self):
        for blocks in (np.empty((0, 15)), np.zeros((3, 14)),
                       np.full((3, 15), np.nan), np.full((3, 15), np.inf)):
            with self.subTest(shape=blocks.shape):
                with self.assertRaises(ValueError):
                    fit_ar(blocks)
        for options in ({"lags": 0}, {"lags": 6}, {"draws": 0},
                        {"tune": -1}, {"chains": 1}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    fit_ar(self.blocks, **options)


class PredictiveDistributionTests(unittest.TestCase):
    def test_ar5_uses_all_five_observations_in_the_correct_lag_order(self):
        coefficients = np.array([.1, .7, -.3, .2, -.1, .05])
        p = Posterior(
            beta=np.broadcast_to(coefficients, (2, 500, 6)).copy(),
            sigma=np.full((2, 500), .2),
            location=30., scale=10., lags=5, n_observations=100,
        )
        observed = np.array([25., 30., 20., 35., 40.])
        expected = list((observed - 30) / 10)
        for _ in range(10):
            expected.append(.1 + sum(coefficients[lag] * expected[-lag]
                                     for lag in range(1, 6)))
        result = forecast(p, observed, seed=30)
        self.assertAlmostEqual(result["mean"][0, 0], 34.25)
        np.testing.assert_allclose(result["mean"][0], 30 + 10 * np.array(expected[5:]), atol=1e-10)

    def test_ar1_matches_exact_future_means_and_sign_probabilities(self):
        p = fixed_posterior()
        observed = np.array([-.1, .3, .2, .1, .4])
        result = forecast(p, observed, seed=31)
        phi, alpha, sigma = .7, .1, 1.1
        steps = np.arange(1, 11)
        expected_mean = phi ** steps * observed[-1] + alpha * (1 - phi ** steps) / (1 - phi)
        expected_sd = sigma * np.sqrt((1 - phi ** (2 * steps)) / (1 - phi ** 2))
        self.assertEqual(result["mean"].shape, (1, 10))
        np.testing.assert_allclose(result["mean"][0], expected_mean, atol=1e-10)
        np.testing.assert_allclose(result["p_positive"][0], norm.cdf(expected_mean / expected_sd), atol=1e-10)
        np.testing.assert_allclose(result["p_negative"] + result["p_positive"], 1, atol=1e-12)
        np.testing.assert_allclose(result["lower"][0], expected_mean + norm.ppf(.05) * expected_sd,
                                   atol=.10 * expected_sd.max())
        np.testing.assert_allclose(result["upper"][0], expected_mean + norm.ppf(.95) * expected_sd,
                                   atol=.10 * expected_sd.max())

    def test_mean_of_ten_accounts_for_correlated_future_innovations(self):
        p = fixed_posterior()
        observed = np.array([-.1, .3, .2, .1, .4])
        result = forecast(p, observed, seed=32)
        # Lower triangular impulse-response matrix maps ten independent
        # innovations to ten correlated future observations.
        impulse = np.zeros((10, 10))
        for future in range(10):
            for innovation in range(future + 1):
                impulse[future, innovation] = .7 ** (future - innovation)
        covariance = 1.1 ** 2 * impulse @ impulse.T
        sd_mean10 = np.sqrt(covariance.sum()) / 10
        expected_mean10 = result["mean"][0].mean()
        self.assertAlmostEqual(result["mean10"][0], expected_mean10, places=10)
        self.assertAlmostEqual(result["p_positive10"][0], norm.cdf(expected_mean10 / sd_mean10), places=10)
        self.assertAlmostEqual(result["p_positive10"][0] + result["p_negative10"][0], 1, places=12)
        # This test would fail if average-of-ten uncertainty ignored covariance.
        independent_sd = np.sqrt(np.trace(covariance)) / 10
        self.assertGreater(abs(norm.cdf(expected_mean10 / independent_sd)
                               - result["p_positive10"][0]), .05)

    def test_zero_threshold_is_transformed_back_to_mw(self):
        p = fixed_posterior(phi=0, alpha=0, sigma=.5, location=35, scale=20)
        result = forecast(p, np.full(5, 35.), seed=33)
        np.testing.assert_allclose(result["mean"], 35, atol=1e-12)
        np.testing.assert_allclose(result["p_positive"], norm.cdf(35 / 10), atol=1e-12)
        # Comparing against zero in standardized units would incorrectly give .5.
        self.assertGreater(result["p_positive"].min(), .999)

    def test_forecasts_retain_same_parameter_draw_through_whole_trajectory(self):
        # A mixture of constant slopes is not equivalent to redrawing phi at
        # each future step. Means at horizon h must average phi**h.
        p = fixed_posterior(alpha=0, sigma=.2, draws=1000)
        p.beta[0, :, 1] = .2
        p.beta[1, :, 1] = .9
        result = forecast(p, np.array([1., 1., 1., 1., 1.]), seed=34)
        expected = (.2 ** np.arange(1, 11) + .9 ** np.arange(1, 11)) / 2
        np.testing.assert_allclose(result["mean"][0], expected, atol=1e-12)

    def test_batch_shape_reproducibility_and_invalid_observations(self):
        p = fixed_posterior(draws=200)
        observed = np.array([[1., 2., 3., 4., 5.], [-1., -2., -3., -4., -5.]])
        first = forecast(p, observed, seed=20)
        second = forecast(p, observed, seed=20)
        for key in ("mean", "lower", "upper", "p_positive", "p_negative"):
            self.assertEqual(first[key].shape, (2, 10))
            np.testing.assert_array_equal(first[key], second[key])
        for key in ("mean10", "lower10", "upper10", "p_positive10", "p_negative10"):
            self.assertEqual(first[key].shape, (2,))
        for invalid in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6],
                        [1, 2, np.nan, 4, 5], [1, 2, 3, 4, np.inf]):
            with self.subTest(observed=invalid):
                with self.assertRaises(ValueError):
                    forecast(p, invalid)

    def test_stationarity_fraction_is_reported_without_removing_draws(self):
        p = fixed_posterior(draws=30)
        p.beta[0, :, 1] = .5
        p.beta[1, :, 1] = 1.1
        self.assertAlmostEqual(stationarity_fraction(p), .5)
        result = forecast(p, np.ones(5), seed=35)
        self.assertTrue(np.isfinite(result["mean"]).all())
        expected = (.5 ** np.arange(1, 11) + 1.1 ** np.arange(1, 11)) / 2
        # alpha is .1 in this fixture; include the known recurrence offset.
        expected += .5 * .1 * ((1 - .5 ** np.arange(1, 11)) / (1 - .5)
                               + (1 - 1.1 ** np.arange(1, 11)) / (1 - 1.1))
        np.testing.assert_allclose(result["mean"][0], expected, atol=1e-10)


class ChronologicalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import predikce_bayes_ar as runner
        cls.runner = runner
        cls.times = pd.DatetimeIndex([
            day + pd.Timedelta(minutes=15 * quarter)
            for day in pd.date_range("2026-09-23", periods=4, freq="D")
            for quarter in range(6)
        ])
        cls.blocks = synthetic_blocks(len(cls.times), seed=745)
        cls.settings = dict(lags=1, draws=200, tune=100, chains=2,
                            seed=842, check_convergence=False)
        cls.fit_histories = []

        def record_fit(history, **kwargs):
            cls.fit_histories.append(history.copy())
            return fit_ar(history, **kwargs)

        with patch.object(runner, "fit_ar", side_effect=record_fit):
            cls.result = runner.backtest(cls.times, cls.blocks, **cls.settings)

    def test_backtest_holds_out_whole_days_and_has_exact_ten_horizons(self):
        self.assertEqual(len(self.fit_histories), 1)
        np.testing.assert_array_equal(self.fit_histories[0], self.blocks[:18])
        minutes = self.result["minutes"]
        bayes = minutes[minutes.model == "Bayes AR(1) MCMC"]
        self.assertEqual(len(bayes), 60)
        self.assertTrue((bayes.start.dt.normalize() == pd.Timestamp("2026-09-26")).all())
        self.assertTrue((bayes.trenink_do <= bayes.start.dt.normalize()).all())
        np.testing.assert_array_equal(bayes.krok, np.tile(np.arange(1, 11), 6))
        np.testing.assert_array_equal(bayes.skutecnost_mw, self.blocks[-6:, 5:].ravel())
        expected_times = np.concatenate([
            pd.date_range(start + pd.Timedelta(minutes=5), periods=10, freq="min").to_numpy()
            for start in self.times[-6:]
        ])
        np.testing.assert_array_equal(bayes.cas.to_numpy(), expected_times)

    def test_modifying_held_out_targets_cannot_change_forecasts(self):
        contaminated = self.blocks.copy()
        contaminated[-6:, 5:] += 1e8
        actual = self.runner.backtest(self.times, contaminated, **self.settings)
        columns = ["start", "cas", "krok", "model", "odhad_mw", "odhad_znamenka",
                   "p_kladna", "p_zaporna", "dolni_90_mw", "horni_90_mw",
                   "trenink_ctvrthodin", "trenink_do"]
        pd.testing.assert_frame_equal(actual["minutes"][columns], self.result["minutes"][columns])
        self.assertFalse(actual["minutes"].skutecnost_mw.equals(self.result["minutes"].skutecnost_mw))

    def test_live_fit_excludes_current_and_future_complete_quarters(self):
        times = pd.date_range("2026-09-23", periods=36, freq="15min")
        blocks = synthetic_blocks(36, seed=932)
        start = times[30]
        known = blocks[30, :5]
        expected = self.runner.predict_interval(times[:30], blocks[:30], known, start, **self.settings)
        contaminated = blocks.copy()
        contaminated[30:] = 1e9
        actual = self.runner.predict_interval(times, contaminated, known, start, **self.settings)
        pd.testing.assert_frame_equal(actual["minutes"], expected["minutes"])
        pd.testing.assert_frame_equal(actual["intervals"], expected["intervals"])
        self.assertTrue((actual["minutes"].trenink_ctvrthodin == 30).all())
        self.assertTrue((actual["minutes"].trenink_do == start).all())
        expected_times = pd.date_range(start + pd.Timedelta(minutes=5), periods=10, freq="min")
        np.testing.assert_array_equal(actual["minutes"].cas.to_numpy(), expected_times.to_numpy())

    def test_binary_scores_exclude_actual_zeros_and_ties_stay_zero(self):
        frame = pd.DataFrame({"skutecnost_mw": [0., 1., -1.], "p_kladna": [1., .8, .3]})
        brier, logloss = self.runner._binary_scores(frame, "skutecnost_mw")
        self.assertAlmostEqual(brier, (.2 ** 2 + .3 ** 2) / 2)
        self.assertAlmostEqual(logloss, -(np.log(.8) + np.log(.7)) / 2)
        deterministic = self.runner._point_prediction(np.zeros((1, 10)))
        np.testing.assert_array_equal(deterministic["sign"], 0)
        np.testing.assert_array_equal(deterministic["p_positive"], .5)
        np.testing.assert_array_equal(self.runner._sign_from_probability(np.array([.1, .5, .9])), [-1, 0, 1])
        frame["model"] = "Bayes AR(1) MCMC"
        self.assertEqual(self.runner.calibration(frame).pocet.sum(), 2)

    def test_failed_convergence_blocks_forecast_publication(self):
        failing = pd.DataFrame({
            "r_hat": [1.2], "ess_bulk": [100.], "ess_tail": [100.],
            "mean": [0.], "sd": [1.],
        }, index=["intercept"])
        with patch.object(self.runner, "diagnostics", return_value=failing):
            with self.assertRaises(self.runner.ConvergenceError) as raised:
                self.runner.backtest(self.times, self.blocks,
                                     **{**self.settings, "check_convergence": True})
        self.assertFalse(raised.exception.diagnostic_table.konvergence_ok.any())


if __name__ == "__main__":
    unittest.main()
