"""Predikce minut 05–14 z minut 00–04 každé čtvrthodiny.
Instalace: pip install pandas numpy matplotlib scikit-learn
Spuštění: python predikce_ctvrthodin.py 'odchylka_cr_*.csv' --out predikce
V Colabu použijte notebook predikce_colab.ipynb, který stáhne celý skript.
Jde o zpětné časové vyhodnocení na historických datech, nikoli živou predikci.
"""
import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def load_blocks(files):
    frames = []
    for path in files:
        raw = pd.read_csv(path, encoding='utf-8-sig', sep=None, engine='python')
        raw.columns = raw.columns.str.strip().str.casefold()
        stamp = raw['timestamp'].astype(str)
        iso = stamp.str.match(r'^\d{4}-\d{2}-\d{2}')
        date = pd.Series(index=raw.index, dtype='datetime64[ns]')
        date.loc[iso] = pd.to_datetime(stamp[iso].str[:10], format='%Y-%m-%d')
        date.loc[~iso] = pd.to_datetime(stamp[~iso].str.extract(r'^(\d{1,2}\.\d{1,2}\.\d{4})')[0], format='%d.%m.%Y')
        frames.append(pd.DataFrame({'time': date + pd.to_timedelta(raw['čas']),
                                    'mw': pd.to_numeric(raw['odchylka čr [mw]'].astype(str).str.replace(',', '.', regex=False))}))
    data = pd.concat(frames, ignore_index=True)
    if data.isna().any().any() or data.groupby('time').mw.nunique().gt(1).any():
        raise ValueError('Chybějící data nebo konfliktní duplicitní časy.')
    data = data.drop_duplicates('time').sort_values('time').set_index('time')
    blocks, starts = [], []
    skipped = 0
    for start, group in data.groupby(data.index.floor('15min')):
        expected = pd.date_range(start, periods=15, freq='min')
        if not group.index.equals(expected):
            skipped += 1
            continue
        starts.append(start)
        blocks.append(group.mw.to_numpy())
    if not blocks:
        raise ValueError('Žádná úplná čtvrthodina.')
    return pd.DatetimeIndex(starts), np.array(blocks), skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', default=['odchylka_cr_*.csv'])
    parser.add_argument('--out', type=Path, default=Path('predikce'))
    args = parser.parse_args(argv)
    files = sorted({p for pattern in args.files for p in glob.glob(pattern, recursive=True)})
    if not files:
        parser.error('Nenalezeny vstupní CSV soubory.')
    times, blocks, skipped = load_blocks(files)
    observed, target = blocks[:, :5], blocks[:, 5:]
    # Všechny vstupy jsou známé při vydání předpovědi po pátém měření.
    features = np.column_stack([observed, observed.mean(axis=1), observed.std(axis=1),
                                observed[:, -1] - observed[:, 0]])
    days = times.normalize().unique().sort_values()
    if len(days) < 4:
        parser.error('Pro toto časové vyhodnocení jsou nutné alespoň čtyři dny.')
    predictions = []
    for day in days[3:]:
        train = times.normalize() < day
        test = times.normalize() == day
        # Parametr je předem pevný, nevybírá se podle výsledků testovacích dnů.
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model.fit(features[train], target[train])
        estimates = {'Poslední hodnota': np.repeat(observed[test, -1:], 10, axis=1),
                     'Průměr prvních 5 min': np.repeat(observed[test].mean(axis=1)[:, None], 10, axis=1),
                     'Ridge model': model.predict(features[test])}
        majority = np.sign(target[train].mean(axis=1))
        majority_sign = pd.Series(majority).mode().iloc[0]
        for name, pred in estimates.items():
            for i, idx in enumerate(np.flatnonzero(test)):
                row = {'start': times[idx], 'model': name, 'skutecny_prumer_mw': target[idx].mean(),
                       'odhad_prumer_mw': pred[i].mean(),
                       'vetsinova_polarita_trenink': majority_sign}
                for minute in range(10):
                    row[f'skutecnost_{minute+5:02d}'] = target[idx, minute]
                    row[f'odhad_{minute+5:02d}'] = pred[i, minute]
                predictions.append(row)
    result = pd.DataFrame(predictions)
    rows = []
    for name, group in result.groupby('model', sort=False):
        truth = group.skutecny_prumer_mw.to_numpy()
        pred = group.odhad_prumer_mw.to_numpy()
        actual_curve = group[[f'skutecnost_{m:02d}' for m in range(5, 15)]].to_numpy()
        pred_curve = group[[f'odhad_{m:02d}' for m in range(5, 15)]].to_numpy()
        rows.append({'model': name, 'pocet_ctvrthodin': len(group),
                     'MAE_prumeru_MW': np.abs(pred-truth).mean(),
                     'RMSE_prumeru_MW': np.sqrt(np.mean((pred-truth)**2)),
                     'MAE_minut_MW': np.abs(actual_curve-pred_curve).mean(),
                     'spravna_polarita_prumeru_pct': 100*np.mean(np.sign(pred)==np.sign(truth)),
                     'vetsinova_polarita_baseline_pct': 100*np.mean(group.vetsinova_polarita_trenink==np.sign(truth))})
    metrics = pd.DataFrame(rows)
    daily = result.assign(abs_error=lambda x: abs(x.odhad_prumer_mw-x.skutecny_prumer_mw),
                          den=lambda x: x.start.dt.date).groupby(['den','model']).abs_error.mean().unstack()
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out/'predikce.csv', index=False)
    metrics.to_csv(args.out/'metriky.csv', index=False)
    daily.to_csv(args.out/'metriky_po_dnech.csv')
    text = (f'Úplné čtvrthodiny: {len(blocks)}, vynechané neúplné: {skipped}.\n'
            f'Testovací dny: {", ".join(str(d.date()) for d in days[3:])}.\n'
            'Každý testovací den se model učí pouze ze starších dnů.\n'
            'Cíl: průměrná MW a jednotlivé minuty dalších 10 minut; polarita znamená znaménko jejich průměru.\n'
            + metrics.to_string(index=False, float_format=lambda x: f'{x:.2f}')
            + '\n\nMAE průměru podle testovacího dne:\n' + daily.to_string(float_format=lambda x: f'{x:.2f}')
            + '\n\nJde o malý časový backtest, nikoli důkaz spolehlivé předpovědi. '
            'Předpověď průměrné polarity neznamená stejnou polaritu ve všech deseti minutách.\n')
    print(text)
    (args.out/'vysledky.txt').write_text(text, encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
    one = result[result.model=='Ridge model']
    axes[0].plot(one.start, one.skutecny_prumer_mw, label='Skutečný průměr dalších 10 min', color='black')
    for name, group in result.groupby('model', sort=False):
        axes[0].plot(group.start, group.odhad_prumer_mw, label=name, alpha=.7, linewidth=1)
    axes[0].axhline(0, color='gray', linewidth=.6)
    axes[0].set(ylabel='Průměrná odchylka [MW]', title='Předpovědi na dnech nezahrnutých do učení')
    axes[0].legend()
    for name, group in result.groupby('model', sort=False):
        err = np.abs(group[[f'odhad_{m:02d}' for m in range(5,15)]].to_numpy()-group[[f'skutecnost_{m:02d}' for m in range(5,15)]].to_numpy()).mean(axis=0)
        axes[1].plot(range(1,11), err, marker='o', label=name)
    axes[1].set(xlabel='Krok předpovědi po posledním známém měření [min]', ylabel='MAE [MW]', xticks=range(1,11))
    axes[1].legend()
    fig.savefig(args.out/'predikce.png', dpi=180)
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main([] if 'ipykernel' in sys.modules else None)
