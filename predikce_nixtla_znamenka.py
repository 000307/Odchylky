"""MLForecast, LightGBM a jejich kombinace: pravděpodobnosti znamének.

python predikce_nixtla_znamenka.py 'odchylka_cr_*.csv' --out predikce_nixtla
Pro nový interval přidejte --first-five 10 20 30 40 50 --start '2026-09-29 12:00'.
Instalace: python -m pip install -r requirements_nixtla.txt
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predikce_linearni_regrese import load_blocks, _validated_history, _number
from nixtla_sign_core import fit_base, predict_base
from sign_calibration import fit_calibrator, apply_calibrator, binary_scores, reliability_table


SEED = 20261102
MLF = 'MLForecast + sigmoid'
RAW = 'LightGBM bez kalibrace'
CLF = 'LightGBM + sigmoid'
MIX = 'Kombinace 50:50'
LAST = 'Poslední znaménko'
FREQUENCY = 'Četnost v minulosti'


def _partition(times, blocks):
    days = times.normalize().unique().sort_values()
    if len(days) < 2:
        raise ValueError('Pro oddělené učení a kalibraci jsou nutné alespoň dva minulé dny.')
    calibration_day = days[-1]
    base = times.normalize() < calibration_day
    calibration = times.normalize() == calibration_day
    return blocks[base], blocks[calibration], {
        'base_train_start': times[base][0],
        'base_train_end': times[base][-1]+pd.Timedelta(minutes=15),
        'calibration_day': calibration_day,
        'base_quarters': int(base.sum()),
        'calibration_quarters': int(calibration.sum()),
    }


def _fit_workflow(base_blocks, calibration_blocks, seed, test_label):
    model = fit_base(base_blocks, seed=seed)
    raw_calibration = predict_base(model, calibration_blocks[:, :5])
    regression_calibrator = fit_calibrator(raw_calibration['regression_mw'], calibration_blocks[:, 5:], kind='regression')
    classifier_calibrator = fit_calibrator(raw_calibration['classifier_raw'], calibration_blocks[:, 5:], kind='classifier')
    rows = []
    for label, calibrator in [('MLForecast', regression_calibrator), ('LightGBM', classifier_calibrator)]:
        rows.append({'test_day': str(test_label), 'model': label, 'kind': calibrator.kind,
                     'n_calibration': calibrator.n_calibration, 'n_positive': calibrator.n_positive,
                     'n_negative': calibrator.n_negative, 'fallback': calibrator.fallback,
                     'constant_probability': calibrator.constant_probability,
                     'fallback_reason': calibrator.fallback_reason})
    return (model, regression_calibrator, classifier_calibrator), pd.DataFrame(rows)


def _probabilities(workflow, observed, past_blocks):
    model, regression_calibrator, classifier_calibrator = workflow
    raw = predict_base(model, observed)
    p_regression = apply_calibrator(regression_calibrator, raw['regression_mw'])
    p_classifier = apply_calibrator(classifier_calibrator, raw['classifier_raw'])
    last = np.where(observed[:, -1:] > 0, 1., np.where(observed[:, -1:] < 0, 0., .5))
    historical = past_blocks[:, 5:].ravel()
    frequency = ((historical > 0).sum()+1)/((historical != 0).sum()+2)
    return {
        MLF: p_regression, RAW: raw['classifier_raw'], CLF: p_classifier,
        MIX: .5*p_regression+.5*p_classifier,
        LAST: np.repeat(last, 10, axis=1),
        FREQUENCY: np.full_like(p_regression, frequency),
    }, raw['regression_mw']


def _frames(starts, probabilities, raw_mw, truth=None):
    frames = []
    starts = pd.DatetimeIndex(starts)
    for name, p in probabilities.items():
        if p.shape != (len(starts), 10) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f'{name}: neplatné pravděpodobnosti nebo počet horizontů.')
        frame = pd.DataFrame({
            'start': np.repeat(starts.to_numpy(), 10),
            'cas': np.repeat(starts.to_numpy(), 10)+np.tile(np.arange(5, 15), len(starts))*np.timedelta64(1, 'm'),
            'krok': np.tile(np.arange(1, 11), len(starts)), 'model': name,
            'p_kladna': p.ravel(), 'p_zaporna': 1-p.ravel(),
            'odhad_znamenka': np.where(p.ravel() > .5, 1, np.where(p.ravel() < .5, -1, 0)),
            'mlforecast_odhad_mw': raw_mw.ravel(),
        })
        if truth is not None:
            frame['skutecnost_mw'] = truth.ravel()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def backtest(times, blocks, initial_train_days=3, seed=SEED, verbose=False):
    times, blocks = _validated_history(times, blocks)
    days = times.normalize().unique().sort_values()
    if initial_train_days < 2 or len(days) <= initial_train_days:
        raise ValueError('Backtest potřebuje alespoň dva počáteční minulé dny a další testovací den; výchozí nastavení používá 3+1.')
    predictions, folds, calibrators = [], [], []
    for fold, day in enumerate(days[initial_train_days:]):
        past = times.normalize() < day
        test = times.normalize() == day
        base_blocks, cal_blocks, info = _partition(times[past], blocks[past])
        if verbose:
            print(f'Test {day.date()}: učení {len(base_blocks)} čtvrthodin, '
                  f'kalibrace {len(cal_blocks)} ({info["calibration_day"].date()}), '
                  f'test {int(test.sum())}.', flush=True)
        workflow, diagnostics = _fit_workflow(base_blocks, cal_blocks, seed+fold, day.date())
        probabilities, raw_mw = _probabilities(workflow, blocks[test, :5], blocks[past])
        frame = _frames(times[test], probabilities, raw_mw, blocks[test, 5:])
        frame['base_train_end'] = info['base_train_end']
        frame['calibration_day'] = info['calibration_day']
        predictions.append(frame)
        folds.append({'test_day': day, **info, 'test_quarters': int(test.sum())})
        calibrators.append(diagnostics)
    return {'predictions': pd.concat(predictions, ignore_index=True),
            'folds': pd.DataFrame(folds), 'calibrators': pd.concat(calibrators, ignore_index=True)}


def predict_interval(times, blocks, observed, start, seed=SEED):
    times, blocks = _validated_history(times, blocks)
    start = pd.Timestamp(start)
    if pd.isna(start) or start.tz is not None or start != start.floor('15min'):
        raise ValueError('Začátek musí být místní datum a čas na :00, :15, :30 nebo :45 bez pásma.')
    observed = np.asarray(observed, dtype=float)
    if observed.shape != (5,) or not np.isfinite(observed).all():
        raise ValueError('Zadejte přesně pět konečných hodnot v časovém pořadí.')
    eligible = times+pd.Timedelta(minutes=15) <= start
    base_blocks, cal_blocks, info = _partition(times[eligible], blocks[eligible])
    workflow, diagnostics = _fit_workflow(base_blocks, cal_blocks, seed, start)
    probabilities, raw_mw = _probabilities(workflow, observed[None, :], blocks[eligible])
    return {'predictions': _frames([start], probabilities, raw_mw),
            'folds': pd.DataFrame([{'test_day': start, **info, 'test_quarters': 1}]),
            'calibrators': diagnostics}


def _score_group(group):
    scores = binary_scores(group.skutecnost_mw.to_numpy(), group.p_kladna.to_numpy())
    nonzero = group.skutecnost_mw.ne(0)
    scores.update(
        pocet_nul=int((~nonzero).sum()),
        p_prumer=group.loc[nonzero, 'p_kladna'].mean(),
        podil_kladnych=group.loc[nonzero, 'skutecnost_mw'].gt(0).mean(),
        spravnost_vcetne_nul_pct=100*group.odhad_znamenka.eq(np.sign(group.skutecnost_mw)).mean(),
    )
    scores['odchylka_prumerne_pravdepodobnosti'] = scores['p_prumer']-scores['podil_kladnych']
    return scores


def summarize(predictions):
    rows, horizon_rows, day_rows, reliability = [], [], [], []
    for name, group in predictions.groupby('model', sort=False):
        rows.append({'model': name, **_score_group(group)})
        for horizon, part in group.groupby('krok', sort=True):
            horizon_rows.append({'model': name, 'krok': horizon, **_score_group(part)})
        for day, part in group.groupby(group.start.dt.normalize(), sort=True):
            day_rows.append({'model': name, 'den': day, **_score_group(part)})
        bins = reliability_table(group.skutecnost_mw.to_numpy(), group.p_kladna.to_numpy())
        bins.insert(0, 'model', name)
        reliability.append(bins)
    return {'metrics': pd.DataFrame(rows), 'horizons': pd.DataFrame(horizon_rows),
            'days': pd.DataFrame(day_rows), 'reliability': pd.concat(reliability, ignore_index=True)}


def save_plots(summary, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.fonttype'] = 'none'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for name, part in summary['horizons'].groupby('model', sort=False):
        axes[0].plot(part.krok, part.brier, marker='o', label=name)
    axes[0].set(xlabel='Krok předpovědi [min]', ylabel='Brierovo skóre (nižší je lepší)', xticks=range(1, 11))
    for name, part in summary['days'].groupby('model', sort=False):
        axes[1].plot(part.den, part.brier, marker='o', label=name)
    axes[1].set_xticks(sorted(summary['days'].den.unique()))
    axes[1].set(ylabel='Brierovo skóre podle testovacího dne')
    axes[1].tick_params(axis='x', rotation=20)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(directory/'porovnani_brier.png', dpi=160)
    fig.savefig(directory/'porovnani_brier.svg', metadata={'Date': None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Ideální kalibrace')
    for name in [MLF, RAW, CLF, MIX]:
        part = summary['reliability'][summary['reliability'].model.eq(name)]
        valid = part['count'].gt(0)
        axes[0].plot(part.loc[valid, 'p_mean'], part.loc[valid, 'positive_rate'], 'o-', label=name)
        axes[1].plot((part.lower+part.upper)/2, 100*part['count']/part['count'].sum(), 'o-', label=name)
    axes[0].set(xlabel='Průměrná předpovězená P(kladná)', ylabel='Skutečný podíl kladných',
                xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel='Předpovězená P(kladná)', ylabel='Podíl předpovědí v intervalu [%]', xlim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(directory/'kalibrace.png', dpi=160)
    fig.savefig(directory/'kalibrace.svg', metadata={'Date': None})
    plt.close(fig)


def versions():
    return {name: importlib.metadata.version(name) for name in
            ['mlforecast', 'lightgbm', 'numpy', 'pandas', 'scikit-learn']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', default=['odchylka_cr_*.csv'])
    parser.add_argument('--out', type=Path, default=Path('predikce_nixtla'))
    parser.add_argument('--train-days', type=int, default=3)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--first-five', nargs=5, type=_number, metavar='MW')
    parser.add_argument('--start', help="Začátek čtvrthodiny, například '2026-09-29 12:00'")
    args = parser.parse_args(argv)
    if (args.first_five is None) != (args.start is None):
        parser.error('--first-five a --start je nutné zadat společně.')
    files = sorted({p for pattern in args.files for p in glob.glob(pattern, recursive=True)})
    if not files:
        parser.error('Nenalezeny vstupní CSV soubory.')
    try:
        times, blocks, skipped = load_blocks(files)
        if args.first_five is not None:
            result = predict_interval(times, blocks, args.first_five, args.start, seed=args.seed)
            args.out.mkdir(parents=True, exist_ok=True)
            result['predictions'].to_csv(args.out/'predikce_noveho_intervalu.csv', index=False)
            print(result['predictions'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        else:
            result = backtest(times, blocks, initial_train_days=args.train_days, seed=args.seed, verbose=True)
            summary = summarize(result['predictions'])
            text = (
                f'Úplné čtvrthodiny: {len(blocks)}, vynechané neúplné: {skipped}. Seed: {args.seed}.\n'
                'Pro každý testovací den: starší dny = učení, poslední minulý den = kalibrace.\n'
                'Po kalibraci se základní modely znovu neučí na kalibračních datech.\n'
                'MLForecast: deset přímých regresorů s přesně pěti vstupy; LightGBM: deset klasifikátorů.\n'
                'Sdílená logistická kalibrace: skóre, horizont a interakce; C=0,1.\n'
                'Kombinace je pevný průměr obou výsledných pravděpodobností 50:50; její kalibrace není zaručena.\n'
                'Skutečné nuly jsou vynechány z binárních skóre a kalibrace; jejich počet je uveden.\n'
                'Poslední znaménko má deterministické pravděpodobnosti 0/1, při nule 0,5.\n'
                'Nižší Brier, log-loss a ECE jsou lepší. ECE závisí na rozdělení do 10 pevných intervalů.\n\n'
                + summary['metrics'].sort_values('brier').to_string(index=False, float_format=lambda v: f'{v:.5f}')
                + '\n\nJde o retrospektivní srovnání. Tyto testovací dny byly prohlíženy už v předchozích experimentech.\n'
                'Pro konečné potvrzení vybrané varianty je nutná nová historie. Minuty v jedné čtvrthodině nejsou nezávislé případy.\n'
            )
            args.out.mkdir(parents=True, exist_ok=True)
            result['predictions'].to_csv(args.out/'predikce.csv', index=False)
            for key, filename in [('metrics', 'metriky.csv'), ('horizons', 'metriky_po_horizontech.csv'),
                                  ('days', 'metriky_po_dnech.csv'), ('reliability', 'kalibrace.csv')]:
                summary[key].to_csv(args.out/filename, index=False)
            (args.out/'vysledky.txt').write_text(text, encoding='utf-8')
            save_plots(summary, args.out)
            print(text)
        result['folds'].to_csv(args.out/'rozdeleni_dat.csv', index=False)
        result['calibrators'].to_csv(args.out/'kalibrace_fitu.csv', index=False)
        (args.out/'verze_knihoven.json').write_text(json.dumps(versions(), indent=2)+'\n', encoding='utf-8')
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f'Výstupy: {args.out.resolve()}')


if __name__ == '__main__':
    main()
