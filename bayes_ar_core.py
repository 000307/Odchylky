"""Bayesian finite-horizon Gaussian AR(p), fitted by a block Gibbs sampler.

The proper Normal-Inverse-Gamma prior and the likelihood are conditional on
the first p observations of each historical complete quarter. Stationarity is
not imposed: posterior unstable draws remain part of all predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr


@dataclass(frozen=True)
class Posterior:
    """Samples retain independent chain and draw axes; sigma is standardized SD."""

    beta: np.ndarray  # chains, retained draws, intercept followed by newest lag
    sigma: np.ndarray  # chains, retained draws
    location: float
    scale: float
    lags: int
    n_observations: int
    mn: np.ndarray | None = None
    Vn: np.ndarray | None = None
    an: float | None = None
    bn: float | None = None


def _positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} musí být celé číslo alespoň {minimum}.")
    return int(value)


def _validated_blocks(blocks, lags):
    lags = _positive_int(lags, "Počet zpoždění")
    if lags > 5:
        raise ValueError("Počet zpoždění musí být 1 až 5, protože je známo pět minut.")
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 2 or blocks.shape[1] != 15 or not len(blocks) or not np.isfinite(blocks).all():
        raise ValueError("Historie musí obsahovat alespoň jednu úplnou čtvrthodinu po 15 konečných hodnotách.")
    return blocks, lags


def posterior_hyperparameters(blocks, lags=5):
    """Exact NIG posterior, also used as an independent check of the MCMC sampler.

    Prior in training-standardized units:
      beta | sigma^2 ~ N([0, .8, 0, ...], sigma^2 diag(1, .25, ...));
      sigma^2 ~ InverseGamma(shape=2, scale=1).
    These fixed prior choices are not selected on held-out days.
    """
    blocks, lags = _validated_blocks(blocks, lags)
    location = float(blocks.mean())
    scale = float(blocks.std(ddof=0))
    if scale == 0:
        scale = 1.0
    z = (blocks - location) / scale
    windows = np.lib.stride_tricks.sliding_window_view(z, lags + 1, axis=1)
    # Every window lies within one quarter; missing quarters are never bridged.
    predictors = windows[:, :, :-1][:, :, ::-1].reshape(-1, lags)
    X = np.column_stack((np.ones(len(predictors)), predictors))
    y = windows[:, :, -1].reshape(-1)
    k = lags + 1
    prior_mean = np.zeros(k)
    prior_mean[1] = .8
    prior_precision = np.diag(np.r_[1.0, np.full(lags, 4.0)])
    precision = X.T @ X + prior_precision
    mn = np.linalg.solve(precision, X.T @ y + prior_precision @ prior_mean)
    Vn = np.linalg.solve(precision, np.eye(k))
    Vn = (Vn + Vn.T) / 2
    an = 2.0 + len(y) / 2
    # The residual form avoids cancellation for nearly deterministic series.
    residual = y - X @ mn
    displacement = mn - prior_mean
    bn = 1.0 + .5 * (residual @ residual + displacement @ prior_precision @ displacement)
    if not np.isfinite(mn).all() or not np.isfinite(Vn).all() or not np.isfinite(bn) or bn <= 0:
        raise ValueError("Posterior není numericky konečný; zkontrolujte měřítko vstupních dat.")
    return {
        "mn": mn, "Vn": Vn, "an": float(an), "bn": float(bn),
        "precision": precision, "location": location, "scale": scale,
        "lags": lags, "n_observations": len(y),
    }


def fit_ar(blocks, lags=5, draws=2000, tune=1000, chains=4, seed=20261102):
    """Run independent block-Gibbs chains with genuine conditional MCMC updates.

    beta | sigma^2, y ~ N(mn, sigma^2 Vn)
    sigma^2 | beta, y ~ IG(an + k/2,
                          bn + (beta-mn)' Vn^-1 (beta-mn)/2)
    """
    draws = _positive_int(draws, "Počet ponechaných vzorků", minimum=4)
    tune = _positive_int(tune, "Počet zahřívacích iterací", minimum=0)
    chains = _positive_int(chains, "Počet řetězců", minimum=2)
    seed = _positive_int(seed, "Seed", minimum=0)
    h = posterior_hyperparameters(blocks, lags)
    mn, Vn, an, bn, precision = (h[name] for name in ("mn", "Vn", "an", "bn", "precision"))
    k = len(mn)
    chol = np.linalg.cholesky(Vn)
    beta = np.empty((chains, draws, k))
    sigma = np.empty((chains, draws))
    # Dispersed variance starts plus independently spawned RNG streams.
    multipliers = np.geomspace(.1, 10.0, chains)
    streams = np.random.SeedSequence(seed).spawn(chains)
    for chain, stream in enumerate(streams):
        rng = np.random.default_rng(stream)
        variance = bn / (an - 1) * multipliers[chain]
        for iteration in range(tune + draws):
            current_beta = mn + np.sqrt(variance) * (chol @ rng.normal(size=k))
            difference = current_beta - mn
            rate = bn + .5 * (difference @ precision @ difference)
            variance = rate / rng.gamma(shape=an + k / 2)
            if iteration >= tune:
                # This pair is one state after a complete Gibbs sweep.
                beta[chain, iteration - tune] = current_beta
                sigma[chain, iteration - tune] = np.sqrt(variance)
    if not np.isfinite(beta).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("MCMC vytvořilo nekonečné hodnoty nebo neplatnou varianci.")
    return Posterior(beta, sigma, h["location"], h["scale"], h["lags"],
                     h["n_observations"], mn=mn, Vn=Vn, an=an, bn=bn)


def _validated_posterior(posterior):
    beta, sigma = np.asarray(posterior.beta), np.asarray(posterior.sigma)
    if (beta.ndim != 3 or beta.shape[-1] != posterior.lags + 1
            or sigma.shape != beta.shape[:2] or not beta.shape[0] or not beta.shape[1]
            or not np.isfinite(beta).all() or not np.isfinite(sigma).all()
            or (sigma <= 0).any() or not np.isfinite(posterior.location)
            or not np.isfinite(posterior.scale) or posterior.scale <= 0):
        raise ValueError("Neplatný posterior nebo jeho měřítko.")
    return beta.reshape(-1, beta.shape[-1]), sigma.reshape(-1)


def diagnostics(posterior):
    """Rank-normalized split R-hat and bulk/tail effective sample sizes (ArviZ)."""
    _validated_posterior(posterior)
    import arviz as az

    variables = {"intercept": posterior.beta[:, :, 0]}
    variables.update({f"phi_{j}": posterior.beta[:, :, j] for j in range(1, posterior.lags + 1)})
    variables["sigma"] = posterior.sigma
    inference = az.from_dict(posterior=variables)
    summary = az.summary(inference, kind="all", round_to="none")
    return summary[["r_hat", "ess_bulk", "ess_tail", "mean", "sd"]].copy()


def stationarity_fraction(posterior):
    """Fraction with companion-matrix spectral radius < 1; no draws are removed."""
    beta, _ = _validated_posterior(posterior)
    p = posterior.lags
    if p == 1:
        return float(np.mean(np.abs(beta[:, 1]) < 1))
    companion = np.zeros((len(beta), p, p))
    companion[:, 0, :] = beta[:, 1:]
    companion[:, 1:, :-1] = np.eye(p - 1)
    radius = np.max(np.abs(np.linalg.eigvals(companion)), axis=1)
    return float(np.mean(radius < 1))


def forecast(posterior, observed, seed=20261102):
    """Ten recursive forecasts, including innovation and parameter uncertainty.

    Marginal sign probabilities average the conditional Gaussian CDF over MCMC
    draws. The 90% pointwise prediction intervals use stochastic trajectories.
    Every trajectory retains the SAME parameter draw through all ten steps.
    Both single (5,) and batched (n,5) input produce arrays shaped (n,10).
    """
    beta, sigma = _validated_posterior(posterior)
    values = np.asarray(observed, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 5 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Předpověď vyžaduje pět konečných známých hodnot pro každý interval.")
    seed = _positive_int(seed, "Seed předpovědi", minimum=0)
    rng = np.random.default_rng(seed)
    phi = beta[:, 1:]
    intercept = beta[:, 0, None]
    D, p = len(beta), posterior.lags
    N = len(values)
    # Innovation impulse-response coefficients for conditional predictive variance.
    psi = np.zeros((D, 10))
    psi[:, 0] = 1
    for horizon in range(1, 10):
        count = min(p, horizon)
        psi[:, horizon] = np.sum(phi[:, :count] * psi[:, horizon - np.arange(1, count + 1)], axis=1)
    variance_factors = np.cumsum(psi ** 2, axis=1)
    variance_mean10_factor = np.sum(np.cumsum(psi, axis=1) ** 2, axis=1) / 100
    result = {name: np.empty((N, 10)) for name in ("mean", "lower", "upper", "p_positive", "p_negative")}
    result.update({name: np.empty(N) for name in ("mean10", "lower10", "upper10", "p_positive10", "p_negative10")})
    # Batching bounds memory even for many requested quarters and posterior draws.
    for begin in range(0, N, 32):
        end = min(N, begin + 32)
        normalized = (values[begin:end] - posterior.location) / posterior.scale
        state_mean = np.broadcast_to(normalized[None, :, -p:], (D, end - begin, p)).copy()
        state_draw = state_mean.copy()
        trajectories = np.empty((D, end - begin, 10))
        conditional_sum = np.zeros((D, end - begin))
        for horizon in range(10):
            conditional_mean = intercept + np.einsum("dp,dnp->dn", phi, state_mean[:, :, ::-1])
            simulated = intercept + np.einsum("dp,dnp->dn", phi, state_draw[:, :, ::-1])
            simulated += sigma[:, None] * rng.normal(size=simulated.shape)
            physical_mean = posterior.location + posterior.scale * conditional_mean
            physical_draw = posterior.location + posterior.scale * simulated
            conditional_sum += physical_mean
            sd = posterior.scale * sigma * np.sqrt(variance_factors[:, horizon])
            result["mean"][begin:end, horizon] = physical_mean.mean(axis=0)
            result["p_positive"][begin:end, horizon] = ndtr(physical_mean / sd[:, None]).mean(axis=0)
            trajectories[:, :, horizon] = physical_draw
            state_mean[:, :, :-1] = state_mean[:, :, 1:]
            state_mean[:, :, -1] = conditional_mean
            state_draw[:, :, :-1] = state_draw[:, :, 1:]
            state_draw[:, :, -1] = simulated
        limits = np.quantile(trajectories, [.05, .95], axis=0)
        result["lower"][begin:end], result["upper"][begin:end] = limits
        conditional_mean10 = conditional_sum / 10
        sd10 = posterior.scale * sigma * np.sqrt(variance_mean10_factor)
        result["mean10"][begin:end] = conditional_mean10.mean(axis=0)
        result["p_positive10"][begin:end] = ndtr(conditional_mean10 / sd10[:, None]).mean(axis=0)
        limits10 = np.quantile(trajectories.mean(axis=2), [.05, .95], axis=0)
        result["lower10"][begin:end], result["upper10"][begin:end] = limits10
    result["p_negative"] = 1 - result["p_positive"]
    result["p_negative10"] = 1 - result["p_positive10"]
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Desetiminutová predikce není konečná; posterior obsahuje numericky nestabilní průběhy. Hodnoty nebyly ořezány.")
    return result
