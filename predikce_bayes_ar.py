"""Bayesovský AR(5) s MCMC: z prvních pěti minut odhad dalších deseti.

Backtest: python predikce_bayes_ar.py 'odchylka_cr_*.csv' --out predikce_bayes
Nový interval: přidejte --first-five 10 20 30 40 50 --start '2026-09-29 12:00'
Instalace: python -m pip install -r requirements_bayes.txt
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from predikce_linearni_regrese import load_blocks, fit_model, _validated_history, _number
from bayes_ar_core import fit_ar, forecast, diagnostics, stationarity_fraction


DEFAULT_SEED = 20261102


class ConvergenceError(RuntimeError):
    def __init__(self, message, diagnostic_table):
        super().__init__(message)
        self.diagnostic_table = diagnostic_table


def _fit_checked(blocks, label, *, lags, draws, tune, chains, seed, check_convergence=True):
    posterior = fit_ar(blocks, lags=lags, draws=draws, tune=tune, chains=chains, seed=seed)
    diag = diagnostics(posterior).rename_axis('parametr').reset_index()
    diag.insert(0, 'uceni_pro', str(label))
    diag['podil_stacionarnich_vzorku'] = stationarity_fraction(posterior)
    diag['pocet_trenovacich_pozorovani'] = posterior.n_observations
    required = diag[['r_hat', 'ess_bulk', 'ess_tail']].to_numpy(dtype=float)
    good = (np.isfinite(required).all() and diag.r_hat.le(1.01).all()
            and diag.ess_bulk.ge(400).all() and diag.ess_tail.ge(400).all())
    diag['konvergence_ok'] = bool(good)
    if check_convergence and not good:
        raise ConvergenceError(
            f'MCMC pro {label} nesplnilo R-hat ≤ 1,01 a bulk/tail ESS ≥ 400. '
            'Zvyšte --draws a --tune a zkontrolujte diagnostika_mcmc.csv.', diag)
    return posterior, diag


def _point_prediction(values):
    """Pravděpodobnosti 0/1 jsou jen deterministická srovnávací předpověď."""
    values = np.asarray(values, dtype=float)
    means = values.mean(axis=1)
    p = np.where(values > 0, 1., np.where(values < 0, 0., .5))
    pm = np.where(means > 0, 1., np.where(means < 0, 0., .5))
    return dict(mean=values, lower=np.full_like(values, np.nan), upper=np.full_like(values, np.nan),
                p_positive=p, p_negative=1-p, mean10=means,
                lower10=np.full(len(values), np.nan), upper10=np.full(len(values), np.nan),
                p_positive10=pm, p_negative10=1-pm, sign=np.sign(values), sign10=np.sign(means))


def _sign_from_probability(p):
    return np.where(p > .5, 1, np.where(p < .5, -1, 0))


def _prediction_frames(starts, predictions, truth=None):
    """U znaménka rozhoduje pravděpodobnost, nikoli znaménko střední hodnoty."""
    minute_frames, interval_frames = [], []
    starts = pd.DatetimeIndex(starts)
    for name, pred in predictions.items():
        n = len(starts)
        signs = pred.get('sign', _sign_from_probability(pred['p_positive']))
        signs10 = pred.get('sign10', _sign_from_probability(pred['p_positive10']))
        minutes = pd.DataFrame({
            'start': np.repeat(starts.to_numpy(), 10),
            'cas': (np.repeat(starts.to_numpy(), 10)
                    + np.tile(np.arange(5, 15), n) * np.timedelta64(1, 'm')),
            'krok': np.tile(np.arange(1, 11), n), 'model': name,
            'odhad_mw': pred['mean'].ravel(), 'odhad_znamenka': signs.ravel().astype(int),
            'p_kladna': pred['p_positive'].ravel(), 'p_zaporna': pred['p_negative'].ravel(),
            'dolni_90_mw': pred['lower'].ravel(), 'horni_90_mw': pred['upper'].ravel(),
        })
        intervals = pd.DataFrame({
            'start': starts, 'model': name, 'odhad_prumer_mw': pred['mean10'],
            'odhad_znamenka': signs10.astype(int),
            'p_kladna': pred['p_positive10'], 'p_zaporna': pred['p_negative10'],
            'dolni_90_mw': pred['lower10'], 'horni_90_mw': pred['upper10'],
        })
        if truth is not None:
            actual_sign = np.sign(truth)
            minutes['skutecnost_mw'] = truth.ravel()
            minutes['skutecne_znamenko'] = actual_sign.ravel().astype(int)
            intervals['skutecny_prumer_mw'] = truth.mean(axis=1)
            intervals['skutecne_znamenko'] = np.sign(truth.mean(axis=1)).astype(int)
            intervals['vsech_10_znamek_spravne'] = (signs == actual_sign).all(axis=1)
        minute_frames.append(minutes)
        interval_frames.append(intervals)
    return pd.concat(minute_frames, ignore_index=True), pd.concat(interval_frames, ignore_index=True)


def _frequency_prediction(training, count):
    # Beta(1,1) vyhlazení četnosti; pouze nenulové minuty ve starších intervalech.
    values = training[:, 5:].ravel()
    nonzero = values != 0
    p = ((values > 0).sum()+1) / (nonzero.sum()+2)
    means = training[:, 5:].mean(axis=1)
    pm = ((means > 0).sum()+1) / ((means != 0).sum()+2)
    result = _point_prediction(np.full((count, 10), values.mean()))
    result.update(p_positive=np.full((count, 10), p), p_negative=np.full((count, 10), 1-p),
                  p_positive10=np.full(count, pm), p_negative10=np.full(count, 1-pm),
                  sign=np.full((count, 10), _sign_from_probability(p)),
                  sign10=np.full(count, _sign_from_probability(pm)))
    return result


def backtest(times, blocks, *, lags=5, draws=2000, tune=1000, chains=4,
             seed=DEFAULT_SEED, initial_train_days=3, check_convergence=True, verbose=False):
    times, blocks = _validated_history(times, blocks)
    days = times.normalize().unique().sort_values()
    if initial_train_days < 1 or len(days) <= initial_train_days:
        raise ValueError(f'Pro backtest je potřeba alespoň {initial_train_days+1} dostupných dnů.')
    minute_frames, interval_frames, diagnostic_frames, fits = [], [], [], []
    for fold, day in enumerate(days[initial_train_days:]):
        train, test = times.normalize() < day, times.normalize() == day
        if verbose:
            print(f'MCMC pro {day.date()}: {train.sum()} minulých čtvrthodin, '
                  f'{chains} řetězce × {draws} uchovaných vzorků.', flush=True)
        posterior, diag = _fit_checked(
            blocks[train], day.date(), lags=lags, draws=draws, tune=tune, chains=chains,
            seed=seed+2*fold, check_convergence=check_convergence)
        known = blocks[test, :5]
        name = f'Bayes AR({lags}) MCMC'
        estimates = {
            'Poslední hodnota': _point_prediction(np.repeat(known[:, -1:], 10, axis=1)),
            'Průměr prvních 5 min': _point_prediction(np.repeat(known.mean(axis=1)[:, None], 10, axis=1)),
            'Lineární regrese': _point_prediction(fit_model(blocks[train]).predict(known)),
            'Četnost znaménka v tréninku': _frequency_prediction(blocks[train], len(known)),
            name: forecast(posterior, known, seed=seed+2*fold+1),
        }
        minute, interval = _prediction_frames(times[test], estimates, blocks[test, 5:])
        for frame in (minute, interval):
            frame['trenink_ctvrthodin'] = int(train.sum())
            frame['trenink_do'] = times[train][-1] + pd.Timedelta(minutes=15)
        minute_frames.append(minute)
        interval_frames.append(interval)
        diagnostic_frames.append(diag)
        fits.append((str(day.date()), posterior))
    return dict(minutes=pd.concat(minute_frames, ignore_index=True),
                intervals=pd.concat(interval_frames, ignore_index=True),
                diagnostics=pd.concat(diagnostic_frames, ignore_index=True), fits=fits)


def predict_interval(times, blocks, observed, start, *, lags=5, draws=2000, tune=1000,
                     chains=4, seed=DEFAULT_SEED, check_convergence=True):
    times, blocks = _validated_history(times, blocks)
    start = pd.Timestamp(start)
    if pd.isna(start) or start.tz is not None or start != start.floor('15min'):
        raise ValueError('Začátek musí být místní datum a čas na :00, :15, :30 nebo :45 bez pásma.')
    observed = np.asarray(observed, dtype=float)
    if observed.shape != (5,) or not np.isfinite(observed).all():
        raise ValueError('Zadejte přesně pět konečných hodnot v časovém pořadí.')
    train = times + pd.Timedelta(minutes=15) <= start
    posterior, diag = _fit_checked(blocks[train], start, lags=lags, draws=draws, tune=tune,
                                  chains=chains, seed=seed, check_convergence=check_convergence)
    minute, interval = _prediction_frames([start], {f'Bayes AR({lags}) MCMC':
                                                   forecast(posterior, observed, seed=seed+1)})
    for frame in (minute, interval):
        frame['trenink_ctvrthodin'] = int(train.sum())
        frame['trenink_do'] = times[train][-1] + pd.Timedelta(minutes=15)
    interval['prumer_prvnich_5min_mw'] = observed.mean()
    interval['odhad_prumer_15min_mw'] = (observed.sum()+10*interval.odhad_prumer_mw)/15
    return dict(minutes=minute, intervals=interval, diagnostics=diag, fits=[('novy_interval', posterior)])


def _binary_scores(group, truth_column):
    # U nuly nelze vybrat + nebo −; hodnotí se ve správnosti znamének,
    # ale nevstupuje do binárních skóre a kalibrace na nenulových případech.
    valid = group[truth_column].ne(0)
    p = group.loc[valid, 'p_kladna'].to_numpy()
    actual = group.loc[valid, truth_column].gt(0).to_numpy(dtype=float)
    if not len(actual):
        return np.nan, np.nan
    clipped = np.clip(p, 1e-12, 1-1e-12)
    return np.mean((p-actual)**2), -np.mean(actual*np.log(clipped)+(1-actual)*np.log1p(-clipped))


def metrics(minutes, intervals):
    rows, horizon_rows = [], []
    for name, group in minutes.groupby('model', sort=False):
        quarter = intervals[intervals.model == name]
        brier, logloss = _binary_scores(group, 'skutecnost_mw')
        brier_mean, logloss_mean = _binary_scores(quarter, 'skutecny_prumer_mw')
        covered = group.dolni_90_mw.notna() & group.horni_90_mw.notna()
        rows.append({
            'model': name, 'pocet_ctvrthodin': len(quarter), 'pocet_minut': len(group),
            'spravna_znamenka_minut_pct': 100*group.odhad_znamenka.eq(group.skutecne_znamenko).mean(),
            'spravne_znamenko_prumeru_pct': 100*quarter.odhad_znamenka.eq(quarter.skutecne_znamenko).mean(),
            'vsech_10_znamek_spravne_pct': 100*quarter.vsech_10_znamek_spravne.mean(),
            'MAE_minut_MW': abs(group.odhad_mw-group.skutecnost_mw).mean(),
            'MAE_prumeru_MW': abs(quarter.odhad_prumer_mw-quarter.skutecny_prumer_mw).mean(),
            'Brier_minut': brier, 'log_loss_minut': logloss,
            'Brier_prumeru': brier_mean, 'log_loss_prumeru': logloss_mean,
            'pokryti_90_intervalu_pct': 100*group.loc[covered, 'skutecnost_mw'].between(
                group.loc[covered, 'dolni_90_mw'], group.loc[covered, 'horni_90_mw']).mean() if covered.any() else np.nan,
        })
        for horizon, part in group.groupby('krok', sort=True):
            b, ll = _binary_scores(part, 'skutecnost_mw')
            horizon_rows.append({'model': name, 'krok': horizon, 'pocet': len(part),
                                 'spravna_znamenka_pct': 100*part.odhad_znamenka.eq(part.skutecne_znamenko).mean(),
                                 'MAE_MW': abs(part.odhad_mw-part.skutecnost_mw).mean(),
                                 'Brier': b, 'log_loss': ll})
    return pd.DataFrame(rows), pd.DataFrame(horizon_rows)


def calibration(minutes):
    data = minutes[minutes.model.str.startswith('Bayes AR(') & minutes.skutecnost_mw.ne(0)].copy()
    data['bin'] = np.minimum((data.p_kladna*10).astype(int), 9)
    rows = []
    for bin_number in range(10):
        group = data[data.bin == bin_number]
        rows.append({'p_od': bin_number/10, 'p_do': (bin_number+1)/10, 'pocet': len(group),
                     'p_prumer': group.p_kladna.mean(),
                     'skutecny_podil_kladnych': group.skutecnost_mw.gt(0).mean() if len(group) else np.nan})
    return pd.DataFrame(rows)


def save_plots(result, horizons, calibration_table, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    for name, group in horizons.groupby('model', sort=False):
        axes[0, 0].plot(group.krok, group.spravna_znamenka_pct, marker='o', label=name)
    axes[0, 0].set(xlabel='Krok předpovědi [min]', ylabel='Správná znaménka [%]', xticks=range(1, 11))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot([0, 1], [0, 1], '--', color='gray', label='Ideální kalibrace')
    valid = calibration_table.pocet.gt(0)
    axes[0, 1].plot(calibration_table.loc[valid, 'p_prumer'],
                    calibration_table.loc[valid, 'skutecny_podil_kladnych'], 'o-', label='Bayes AR')
    axes[0, 1].set(xlabel='Předpovězená P(kladná)', ylabel='Skutečný podíl kladných', xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].legend()
    group = result['intervals'][result['intervals'].model.str.startswith('Bayes AR(')]
    axes[1, 0].plot(group.start, group.skutecny_prumer_mw, color='black', label='Skutečný průměr dalších 10 min')
    axes[1, 0].plot(group.start, group.odhad_prumer_mw, label='Bayes AR: střední hodnota')
    axes[1, 0].fill_between(group.start, group.dolni_90_mw, group.horni_90_mw, alpha=.2, label='90% predikční interval')
    axes[1, 0].set(ylabel='Odchylka [MW]')
    axes[1, 0].tick_params(axis='x', rotation=20)
    axes[1, 0].legend(fontsize=8)
    for name, part in horizons.groupby('model', sort=False):
        axes[1, 1].plot(part.krok, part.Brier, label=name)
    axes[1, 1].set(xlabel='Krok předpovědi [min]', ylabel='Brierovo skóre (nižší je lepší)', xticks=range(1, 11))
    axes[1, 1].legend(fontsize=8)
    fig.savefig(directory/'predikce_polarity.png', dpi=160)
    plt.close(fig)
    label, posterior = result['fits'][0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), layout='constrained')
    for chain in range(posterior.beta.shape[0]):
        for axis, values in zip(axes, [posterior.beta[chain, :, 0], posterior.beta[chain, :, 1], posterior.sigma[chain]]):
            axis.plot(values, linewidth=.6, alpha=.7, label=f'Řetězec {chain+1}')
    for axis, label_y in zip(axes, ['Konstanta α', 'Koeficient φ₁', 'Inovační σ']):
        axis.set(ylabel=label_y)
    axes[0].set_title(f'MCMC po vyřazení warm-up: první fit {label}, standardizované jednotky')
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel('Uchovaný vzorek')
    fig.savefig(directory/'mcmc_stopy.png', dpi=150)
    plt.close(fig)


def _save_posteriors(result, directory):
    for label, posterior in result['fits']:
        np.savez_compressed(directory/f'posterior_{label}.npz', beta=posterior.beta,
                            sigma=posterior.sigma, location=posterior.location,
                            scale=posterior.scale, lags=posterior.lags,
                            n_observations=posterior.n_observations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', default=['odchylka_cr_*.csv'])
    parser.add_argument('--out', type=Path, default=Path('predikce_bayes'))
    parser.add_argument('--lags', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--train-days', type=int, default=3)
    parser.add_argument('--first-five', nargs=5, type=_number, metavar='MW')
    parser.add_argument('--start', help="Začátek čtvrthodiny, např. '2026-09-29 12:00'")
    args = parser.parse_args(argv)
    if (args.first_five is None) != (args.start is None):
        parser.error('--first-five a --start je nutné zadat společně.')
    files = sorted({p for pattern in args.files for p in glob.glob(pattern, recursive=True)})
    if not files:
        parser.error('Nenalezeny vstupní CSV soubory.')
    settings = dict(lags=args.lags, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed)
    try:
        times, blocks, skipped = load_blocks(files)
        if args.first_five is not None:
            result = predict_interval(times, blocks, args.first_five, args.start, **settings)
            args.out.mkdir(parents=True, exist_ok=True)
            result['minutes'].to_csv(args.out/'predikce_noveho_intervalu.csv', index=False)
            result['intervals'].to_csv(args.out/'souhrn_predikce.csv', index=False)
            print(result['minutes'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
            print(result['intervals'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        else:
            result = backtest(times, blocks, initial_train_days=args.train_days, verbose=True, **settings)
            summary, horizons = metrics(result['minutes'], result['intervals'])
            bins = calibration(result['minutes'])
            diag = result['diagnostics']
            text = (
                f'Úplné čtvrthodiny: {len(blocks)}, vynechané neúplné: {skipped}.\n'
                f'Bayes AR({args.lags}), Gibbs MCMC: {args.chains} řetězce, {args.tune} warm-up '
                f'a {args.draws} uchovaných vzorků na řetězec. Seed: {args.seed}.\n'
                f'Počáteční učení: {args.train_days} dostupné dny; každý testovací den používá jen starší dny.\n'
                f'MCMC: max R-hat {diag.r_hat.max():.5f}, min bulk ESS {diag.ess_bulk.min():.0f}, '
                f'min tail ESS {diag.ess_tail.min():.0f}.\n'
                f'Podíl stacionárních vzorků podle fitu: '
                f'{diag.groupby("uceni_pro").podil_stacionarnich_vzorku.first().round(4).to_dict()}.\n'
                'Model neomezuje stacionaritu; nestacionární vzorky se neodstraňují.\n'
                'Znaménko Bayes modelu vychází z P(kladná)>0,5; při rovnosti je výsledek 0.\n'
                'Brier/log-loss a kalibrace vynechávají skutečné nuly. Nuly se hodnotí ve správnosti znamének.\n'
                'Pravděpodobnosti 0/1 u bodových metod jsou deterministické baseline, nikoli kalibrované odhady.\n\n'
                + summary.to_string(index=False, float_format=lambda v: f'{v:.4f}')
                + '\n\nNižší Brier/log-loss je lepší. 90% predikční interval zahrnuje nejistotu parametrů i nových inovací.\n'
                'Pravděpodobnosti platí v rámci předpokladů modelu; krátký historický test neověřuje dlouhodobou přesnost.\n'
            )
            args.out.mkdir(parents=True, exist_ok=True)
            result['minutes'].to_csv(args.out/'predikce.csv', index=False)
            result['intervals'].to_csv(args.out/'prumery.csv', index=False)
            summary.to_csv(args.out/'metriky.csv', index=False)
            horizons.to_csv(args.out/'metriky_po_horizontech.csv', index=False)
            bins.to_csv(args.out/'kalibrace.csv', index=False)
            (args.out/'vysledky.txt').write_text(text, encoding='utf-8')
            save_plots(result, horizons, bins, args.out)
            print(text)
        result['diagnostics'].to_csv(args.out/'diagnostika_mcmc.csv', index=False)
        _save_posteriors(result, args.out)
    except ConvergenceError as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        exc.diagnostic_table.to_csv(args.out/'diagnostika_mcmc.csv', index=False)
        parser.error(str(exc))
    except (ValueError, KeyError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(f'Výstupy: {args.out.resolve()}')


if __name__ == '__main__':
    main()
