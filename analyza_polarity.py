"""Použití: python analyza_polarity.py 'cesta/*.csv' --out vysledky
Závislosti: python -m pip install pandas numpy matplotlib
Časy jsou lokální bez pásma; předpokládá se období bez změny času.
"""
import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_deviation(data, out):
    # Doplnění chybějících minut jako NaN zabrání spojování čáry přes mezery.
    series = data.set_index('time').mw.asfreq('min')
    fig, ax = plt.subplots(figsize=(15, 5), layout='constrained')
    ax.plot(series.index, series.values, color='#334155', linewidth=0.65)
    ax.fill_between(series.index, series.values, 0, where=series.values >= 0,
                    interpolate=True, color='#16a34a', alpha=0.5, label='Kladná odchylka')
    ax.fill_between(series.index, series.values, 0, where=series.values <= 0,
                    interpolate=True, color='#dc2626', alpha=0.5, label='Záporná odchylka')
    ax.axhline(0, color='black', linewidth=0.8)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.\n%H:%M'))
    ax.set(title='Odchylka ČR v čase', xlabel='Datum a místní čas', ylabel='Odchylka ČR [MW]')
    ax.grid(alpha=0.2)
    ax.legend(loc='best')
    ax.margins(x=0)
    path = out / 'odchylky.png'
    fig.savefig(path, dpi=180)
    print(f'Graf uložen: {path}')
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close(fig)



def analyze_time_patterns(data, eligible, flip, out):
    # Zařazení podle VÝCHOZÍHO měření: minuta 59 znamená přechod :59 → :00.
    time = data.time.shift()
    pairs = pd.DataFrame({'hodina': time.dt.hour, 'minuta': time.dt.minute,
                          'predchozi': data.polarity.shift(), 'zmena': flip})[eligible].copy()
    pairs['petiminutovka'] = (pairs.minuta // 5) * 5
    pairs['ctvrthodina'] = (pairs.minuta // 15) * 15
    tables = {}
    for key, size in [('hodina', 24), ('minuta', 60), ('petiminutovka', 12), ('ctvrthodina', 4)]:
        records = []
        for direction, sign in [('celkem', None), ('+ → −', 1), ('− → +', -1)]:
            subset = pairs if sign is None else pairs[pairs.predchozi.eq(sign)]
            table = subset.groupby(key).zmena.agg(['count', 'sum'])
            bins = range(size) if key in ('hodina', 'minuta') else range(0, 60, 5 if key == 'petiminutovka' else 15)
            table = table.reindex(bins, fill_value=0)
            for bin_, row in table.iterrows():
                n, k = int(row['count']), int(row['sum'])
                records.append({key: bin_, 'smer': direction, 'pozorovani': n, 'zmeny': k,
                                'pravdepodobnost_pct': 100*k/n if n else np.nan})
        tables[key] = pd.DataFrame(records)
        tables[key].to_csv(out / f'trend_{key}.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout='constrained')
    for ax, key, xlabel in zip(axes, ['hodina', 'petiminutovka'],
                               ['Hodina dne (výchozí měření)', 'Minuta v hodině (začátek pětiminutového intervalu)']):
        for direction, color in [('celkem', '#334155'), ('+ → −', '#16a34a'), ('− → +', '#dc2626')]:
            t = tables[key].query('smer == @direction')
            ax.plot(t[key], t.pravdepodobnost_pct, marker='o', color=color, label=direction)
        ax.set(xlabel=xlabel, ylabel='Změna v další minutě [%]', ylim=(0, None))
        ax.set_xticks(range(24) if key == 'hodina' else range(0, 60, 5))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Četnost změn polarity podle času — popis dodaných dat')
    fig.savefig(out / 'trendy_polarity.png', dpi=180)
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close(fig)
    report = ['Časové skupiny vycházejí z času PŘED změnou; minuta 59 = :59 → :00.',
              'Jde o četnosti změn v další minutě, nikoli pravděpodobnost alespoň jedné změny za celou hodinu.',
              'Minutová pozorování jsou závislá; šest dní nestačí k potvrzení stabilního denního vzorce.',
              '\nPodle hodiny dne:', tables['hodina'].to_string(index=False),
              '\nPodle čtvrthodiny:', tables['ctvrthodina'].to_string(index=False)]
    (out / 'trendy.txt').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))


def analyze(files, out):
    frames = []
    for file in files:
        raw = pd.read_csv(file, encoding='utf-8-sig', sep=None, engine='python')
        raw.columns = raw.columns.str.strip()
        cols = {c.casefold(): c for c in raw.columns}
        stamp = raw[cols['timestamp']].astype(str)
        # ISO v přiložených CSV; alternativně dd.mm.rrrr (10 znaků, nikoli 8).
        iso = stamp.str.match(r'^\d{4}-\d{2}-\d{2}')
        date = pd.Series(index=raw.index, dtype='datetime64[ns]')
        date.loc[iso] = pd.to_datetime(stamp[iso].str[:10], format='%Y-%m-%d')
        date.loc[~iso] = pd.to_datetime(stamp[~iso].str.extract(r'^(\d{1,2}\.\d{1,2}\.\d{4})')[0], format='%d.%m.%Y')
        clock = pd.to_timedelta(raw[cols['čas']].astype(str))
        values = pd.to_numeric(raw[cols['odchylka čr [mw]']].astype(str).str.replace(',', '.', regex=False), errors='raise')
        frames.append(pd.DataFrame({'time': date + clock, 'mw': values}))
    data = pd.concat(frames, ignore_index=True)
    if data.isna().any().any():
        raise ValueError('Chybějící datum, čas nebo odchylka.')
    if data.groupby('time').mw.nunique().gt(1).any():
        raise ValueError('Stejný čas má různé hodnoty odchylky.')
    duplicates = int(data.duplicated('time').sum())
    data = data.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    delta = data.time.diff()
    step = delta.dropna().mode().iloc[0]
    if step != pd.Timedelta(minutes=1):
        raise ValueError(f'Očekávána minutová data, nalezen krok {step}.')
    contiguous = delta.eq(step)
    data['polarity'] = np.sign(data.mw).astype(int)
    prev = data.polarity.shift()
    # Nula přeruší úsek; přechod přes nulu nepočítáme jako přímou změnu +/−.
    eligible = contiguous & data.polarity.ne(0) & prev.ne(0) & prev.notna()
    flip = eligible & data.polarity.ne(prev)
    records = []
    for sign, label in [(None, 'celkem'), (1, 'kladná → záporná'), (-1, 'záporná → kladná')]:
        mask = eligible if sign is None else eligible & prev.eq(sign)
        n = int(mask.sum())
        records.append({'směr': label, 'příležitosti': n, 'změny': int((flip & mask).sum()),
                        'pravděpodobnost_pct': 100 * (flip & mask).sum() / n if n else np.nan})
    transitions = pd.DataFrame(records)
    group = (~contiguous | data.polarity.ne(prev)).cumsum()
    runs = data.groupby(group).agg(start=('time', 'first'), last=('time', 'last'), polarity=('polarity', 'first'), samples=('mw', 'size'))
    # Kompletní úsek musí mít pozorovaný začátek i konec v souvislé řadě.
    starts = data.groupby(group).head(1).index.to_numpy()
    ends = data.groupby(group).tail(1).index.to_numpy()
    left_known = np.array([i > 0 and contiguous.iloc[i] for i in starts])
    right_known = np.array([i + 1 < len(data) and contiguous.iloc[i+1] for i in ends])
    runs['complete'] = left_known & right_known
    runs['minutes'] = runs.samples * step.total_seconds() / 60
    summary = []
    for sign, label in [(1, 'kladná'), (-1, 'záporná')]:
        lengths = runs.loc[runs.complete & runs.polarity.eq(sign), 'minutes']
        summary.append({'polarita': label, 'úseky': len(lengths), 'průměr_min': lengths.mean(),
                        'medián_min': lengths.median(), 'p65_min': lengths.quantile(.65, interpolation='higher'),
                        'p95_min': lengths.quantile(.95, interpolation='higher')})
    summary = pd.DataFrame(summary)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / 'useky.csv', index=False)
    summary.to_csv(out / 'delky.csv', index=False)
    transitions.to_csv(out / 'zmeny.csv', index=False)
    text = '\n'.join([
        f'Období: {data.time.min()} až {data.time.max()}',
        f'Měření: {len(data)}, odstraněné duplicity: {duplicates}, nuly: {data.polarity.eq(0).sum()}',
        f'Mezery v řadě: {(delta.notna() & ~contiguous).sum()}, chybějící minuty: {int(((delta[delta > step] / step) - 1).sum())}',
        f'Změny polarity za hodinu souvisle pozorovaných dat: {flip.sum() / (contiguous.sum()/60):.3f}',
        '\nDélky kompletních úseků (minuty):', summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'),
        '\nPřímé změny mezi sousedními nenulovými měřeními:', transitions.to_string(index=False, float_format=lambda x: f'{x:.3f}'),
        '\nNuly přerušují úseky. Přechody přes nulu a mezery nejsou přímé změny polarity.',
        'Úseky na okrajích dat/mezer nejsou zahrnuty do statistik délek (jejich úplnou délku neznáme).',
        'Percentily jsou empirické horní kvantily; nejde o intervaly spolehlivosti.',
        'Pravděpodobnost platí pro další minutu při náhodně vybraném nenulovém měření; nejde o předpověď.'
    ])
    (out / 'vysledky.txt').write_text(text, encoding='utf-8')
    print(text)
    plot_deviation(data, out)
    analyze_time_patterns(data, eligible, flip, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', default=['*.csv'], help='CSV soubory nebo glob vzory (výchozí: *.csv)')
    parser.add_argument('--out', type=Path, default=Path('vysledky'))
    args = parser.parse_args(argv)
    files = sorted({p for pattern in args.files for p in glob.glob(pattern, recursive=True)})
    if not files:
        parser.error('Nenalezeny žádné CSV soubory.')
    analyze(files, args.out)


if __name__ == '__main__':
    # Při vložení do buňky nepřebírat argumenty běžícího Jupyter kernelu (-f).
    # Příkazové spuštění (také !python v Colabu) zachovává běžné CLI argumenty.
    main([] if 'ipykernel' in sys.modules else None)
