"""Lineární regrese: z minut 00–04 předpoví minuty 05–14 čtvrthodiny.

Instalace: python -m pip install numpy pandas matplotlib scikit-learn
Backtest: python predikce_linearni_regrese.py 'odchylka_cr_*.csv' --out predikce_lr
Nový interval: přidejte --first-five 10 20 30 40 50 --start '2026-09-29 12:00'
Časy jsou místní; vstupní období nesmí zahrnovat změnu letního/zimního času.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


MODEL_NAME = "Lineární regrese"
HORIZONS = range(5, 15)


def load_blocks(files):
    """Načte původní CSV; vrátí pouze úplné čtvrthodiny bez interpolace."""
    frames, offsets = [], set()
    for path in files:
        raw = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
        raw.columns = raw.columns.str.strip().str.casefold()
        required = {"timestamp", "čas", "odchylka čr [mw]"}
        if not required.issubset(raw.columns):
            raise ValueError(f"{path}: chybí sloupce {sorted(required - set(raw.columns))}.")
        stamp = raw["timestamp"].astype(str).str.strip()
        offsets.update(stamp.str.extract(r"(Z|[+-]\d{2}:\d{2})$", expand=False).dropna())
        iso = stamp.str.match(r"^\d{4}-\d{2}-\d{2}")
        dates = pd.Series(index=raw.index, dtype="datetime64[ns]")
        dates.loc[iso] = pd.to_datetime(stamp[iso].str[:10], format="%Y-%m-%d")
        dates.loc[~iso] = pd.to_datetime(
            stamp[~iso].str.extract(r"^(\d{1,2}\.\d{1,2}\.\d{4})")[0], format="%d.%m.%Y")
        clock = pd.to_timedelta(raw["čas"])
        if (clock < pd.Timedelta(0)).any() or (clock >= pd.Timedelta(days=1)).any():
            raise ValueError(f"{path}: čas musí být v rozsahu 00:00–23:59.")
        values = pd.to_numeric(raw["odchylka čr [mw]"].astype(str).str.replace(",", ".", regex=False))
        frames.append(pd.DataFrame({"time": dates + clock, "mw": values}))
    if len(offsets) > 1:
        raise ValueError("Vstupy zahrnují různá časová pásma nebo změnu času; rozdělte je na období se stejným UTC posunem.")
    if not frames:
        raise ValueError("Nebyly zadány vstupní CSV soubory.")
    data = pd.concat(frames, ignore_index=True)
    if data.isna().any().any() or not np.isfinite(data.mw.to_numpy(dtype=float)).all():
        raise ValueError("Vstupy obsahují chybějící nebo nekonečné hodnoty.")
    if not data.time.eq(data.time.dt.floor("min")).all():
        raise ValueError("Očekává se jedno měření na celou minutu.")
    if data.groupby("time").mw.nunique().gt(1).any():
        raise ValueError("Konfliktní duplicitní hodnoty pro stejný čas.")
    data = data.drop_duplicates("time").sort_values("time").set_index("time")
    blocks, starts, skipped = [], [], 0
    for start, group in data.groupby(data.index.floor("15min")):
        if not group.index.equals(pd.date_range(start, periods=15, freq="min")):
            skipped += 1
            continue
        starts.append(start)
        blocks.append(group.mw.to_numpy(dtype=float))
    if not blocks:
        raise ValueError("Žádná úplná čtvrthodina s 15 minutovými měřeními.")
    return pd.DatetimeIndex(starts), np.asarray(blocks), skipped


def _validated_blocks(blocks):
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 2 or blocks.shape[1] != 15 or not np.isfinite(blocks).all():
        raise ValueError("Historie musí obsahovat úplné čtvrthodiny po 15 konečných hodnotách.")
    return blocks


def _validated_history(times, blocks):
    blocks = _validated_blocks(blocks)
    times = pd.DatetimeIndex(times)
    if (len(times) != len(blocks) or times.hasnans or times.tz is not None
            or not times.is_unique or not times.is_monotonic_increasing
            or not times.equals(times.floor("15min"))):
        raise ValueError("Začátky čtvrthodin musí být místní časy seřazené bez duplicit na :00, :15, :30 nebo :45.")
    return times, blocks


def fit_model(blocks):
    """Deset OLS rovnic y_h = intercept_h + sum(b_hj * x_j), bez penalizace."""
    blocks = _validated_blocks(blocks)
    if len(blocks) < 6:
        raise ValueError("Pro učení je potřeba alespoň 6 úplných minulých čtvrthodin; doporučeno je více dnů.")
    # Pouze pět známých měření; odvozený průměr a rozdíl by byly redundantní.
    return LinearRegression().fit(blocks[:, :5], blocks[:, 5:])


def backtest(times, blocks, initial_train_days=3):
    """Každý testovací den používá model naučený jen z předchozích dnů."""
    times, blocks = _validated_history(times, blocks)
    days = times.normalize().unique().sort_values()
    if initial_train_days < 1 or len(days) <= initial_train_days:
        raise ValueError(f"Pro backtest je potřeba alespoň {initial_train_days + 1} dnů s úplnými čtvrthodinami.")
    result = []
    for day in days[initial_train_days:]:
        train, test = times.normalize() < day, times.normalize() == day
        model = fit_model(blocks[train])
        observed = blocks[test, :5]
        estimates = {
            "Poslední hodnota": np.repeat(observed[:, -1:], 10, axis=1),
            "Průměr prvních 5 min": np.repeat(observed.mean(axis=1)[:, None], 10, axis=1),
            MODEL_NAME: model.predict(observed),
        }
        majority_sign = pd.Series(np.sign(blocks[train, 5:].mean(axis=1))).mode().iloc[0]
        for name, predictions in estimates.items():
            for idx, pred in zip(np.flatnonzero(test), predictions):
                row = {
                    "start": times[idx], "model": name,
                    "skutecny_prumer_mw": blocks[idx, 5:].mean(),
                    "odhad_prumer_mw": pred.mean(),
                    "odhad_prumer_15min_mw": (blocks[idx, :5].sum() + pred.sum()) / 15,
                    "vetsinova_polarita_trenink": majority_sign,
                    "trenink_ctvrthodin": int(train.sum()),
                    "trenink_do": times[train][-1] + pd.Timedelta(minutes=15),
                }
                for column, minute in enumerate(HORIZONS):
                    row[f"skutecnost_{minute:02d}"] = blocks[idx, minute]
                    row[f"odhad_{minute:02d}"] = pred[column]
                result.append(row)
    return pd.DataFrame(result)


def predict_interval(times, blocks, observed, start):
    """Předpověď z pěti hodnot; historie končí nejpozději na začátku intervalu."""
    times, blocks = _validated_history(times, blocks)
    observed = np.asarray(observed, dtype=float)
    if observed.shape != (5,) or not np.isfinite(observed).all():
        raise ValueError("Zadejte přesně pět konečných minutových hodnot.")
    start = pd.Timestamp(start)
    if pd.isna(start) or start.tz is not None or start != start.floor("15min"):
        raise ValueError("Začátek musí být místní datum a čas na :00, :15, :30 nebo :45 bez časového pásma.")
    train = times + pd.Timedelta(minutes=15) <= start
    model = fit_model(blocks[train])
    result = pd.DataFrame({
        "cas": pd.date_range(start + pd.Timedelta(minutes=5), periods=10, freq="min"),
        "odhad_mw": model.predict(observed[None, :])[0],
    })
    result.attrs.update(trenink_ctvrthodin=int(train.sum()),
                        trenink_do=times[train][-1] + pd.Timedelta(minutes=15),
                        hodnost_modelu=int(model.rank_))
    return result


def summarize(result):
    rows = []
    for name, group in result.groupby("model", sort=False):
        error = group.odhad_prumer_mw.to_numpy() - group.skutecny_prumer_mw.to_numpy()
        actual = group[[f"skutecnost_{m:02d}" for m in HORIZONS]].to_numpy()
        pred = group[[f"odhad_{m:02d}" for m in HORIZONS]].to_numpy()
        rows.append({
            "model": name, "pocet_ctvrthodin": len(group),
            "MAE_prumeru_MW": np.abs(error).mean(),
            "RMSE_prumeru_MW": np.sqrt(np.mean(error ** 2)),
            "MAE_minut_MW": np.abs(pred - actual).mean(),
            "spravna_polarita_prumeru_pct": 100 * np.mean(
                np.sign(group.odhad_prumer_mw) == np.sign(group.skutecny_prumer_mw)),
            "vetsinova_polarita_baseline_pct": 100 * np.mean(
                group.vetsinova_polarita_trenink == np.sign(group.skutecny_prumer_mw)),
        })
    daily = result.assign(
        abs_error=lambda x: abs(x.odhad_prumer_mw - x.skutecny_prumer_mw),
        den=lambda x: x.start.dt.date,
    ).groupby(["den", "model"]).abs_error.mean().unstack()
    return pd.DataFrame(rows), daily


def save_plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    one = result[result.model == MODEL_NAME]
    axes[0].plot(one.start, one.skutecny_prumer_mw, label="Skutečný průměr dalších 10 min", color="black")
    for name, group in result.groupby("model", sort=False):
        axes[0].plot(group.start, group.odhad_prumer_mw, label=name, alpha=.7, linewidth=1)
        actual = group[[f"skutecnost_{m:02d}" for m in HORIZONS]].to_numpy()
        pred = group[[f"odhad_{m:02d}" for m in HORIZONS]].to_numpy()
        axes[1].plot(range(1, 11), np.abs(pred - actual).mean(axis=0), marker="o", label=name)
    axes[0].axhline(0, color="gray", linewidth=.6)
    axes[0].set(ylabel="Průměrná odchylka [MW]", title="Predikce na dnech mimo učení")
    axes[1].set(xlabel="Krok po posledním známém měření [min]", ylabel="MAE [MW]", xticks=range(1, 11))
    for axis in axes:
        axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _number(text):
    try:
        value = float(text.replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Očekává se číselná hodnota MW.") from exc
    if not np.isfinite(value):
        raise argparse.ArgumentTypeError("Hodnota MW musí být konečná.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", default=["odchylka_cr_*.csv"], help="CSV nebo vzory názvů CSV")
    parser.add_argument("--out", type=Path, default=Path("predikce_lr"))
    parser.add_argument("--train-days", type=int, default=3, help="Počet prvních dostupných dnů pro učení v backtestu")
    parser.add_argument("--first-five", nargs=5, type=_number, metavar="MW", help="Pět známých hodnot v časovém pořadí")
    parser.add_argument("--start", help="Začátek předpovídané čtvrthodiny, např. '2026-09-29 12:00'")
    args = parser.parse_args(argv)
    if (args.first_five is None) != (args.start is None):
        parser.error("--first-five a --start je nutné zadat společně.")
    files = sorted({p for pattern in args.files for p in glob.glob(pattern, recursive=True)})
    if not files:
        parser.error("Nenalezeny vstupní CSV soubory.")
    try:
        times, blocks, skipped = load_blocks(files)
        if args.first_five is not None:
            prediction = predict_interval(times, blocks, args.first_five, args.start)
            mean10 = prediction.odhad_mw.mean()
            summary = pd.DataFrame([{
                "start": pd.Timestamp(args.start),
                "prumer_prvnich_5min_mw": np.mean(args.first_five),
                "odhad_prumer_dalsich_10min_mw": mean10,
                "odhad_prumer_15min_mw": (sum(args.first_five) + prediction.odhad_mw.sum()) / 15,
                "odhad_polarity_10min": int(np.sign(mean10)),
                **prediction.attrs,
            }])
            args.out.mkdir(parents=True, exist_ok=True)
            prediction.to_csv(args.out / "predikce_noveho_intervalu.csv", index=False)
            summary.to_csv(args.out / "souhrn_predikce.csv", index=False)
            print(prediction.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        else:
            result = backtest(times, blocks, args.train_days)
            metrics, daily = summarize(result)
            text = (
                f"Úplné čtvrthodiny: {len(blocks)}, vynechané neúplné: {skipped}.\n"
                "Lineární regrese bez penalizace: pět vstupů a deset minutových výstupů.\n"
                f"Počáteční učení: {args.train_days} dostupné dny; další dny se testují postupně.\n"
                "Každý testovací den se model učí pouze ze starších dnů.\n"
                "Polarita je znaménko průměru dalších 10 minut.\n\n"
                + metrics.to_string(index=False, float_format=lambda x: f"{x:.2f}")
                + "\n\nMAE průměru podle dne:\n" + daily.to_string(float_format=lambda x: f"{x:.2f}")
                + "\n\nHistorický test na krátkém období neověřuje dlouhodobou přesnost.\n"
            )
            args.out.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.out / "predikce.csv", index=False)
            metrics.to_csv(args.out / "metriky.csv", index=False)
            daily.to_csv(args.out / "metriky_po_dnech.csv")
            (args.out / "vysledky.txt").write_text(text, encoding="utf-8")
            save_plot(result, args.out / "predikce.png")
            print(text)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Výstupy: {args.out.resolve()}")


if __name__ == "__main__":
    main()
