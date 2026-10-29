# Odchylka ČR

## Predikce dalších 10 minut z prvních 5 minut čtvrthodiny

[Otevřít notebook v Google Colabu](https://colab.research.google.com/github/000307/Odchylky/blob/main/predikce_colab.ipynb)

V notebooku spusťte buňky postupně shora dolů. Po výzvě vyberte všechny vstupní
CSV soubory. Notebook stáhne úplný skript z tohoto repozitáře, vypíše výsledky,
zobrazí graf a nabídne stažení výstupů v ZIP souboru. Dlouhý Python kód není
potřeba kopírovat do buňky.

Při místním spuštění:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
python predikce_ctvrthodin.py "odchylka_cr_*.csv" --out predikce
```

### Co se vyhodnocuje

- Každá čtvrthodina začíná v `:00`, `:15`, `:30` nebo `:45`.
- Prvních pět minut tvoří vstup. Například z měření `12:00–12:04` se
  odhadují jednotlivá měření `12:05–12:14` a jejich průměr.
- Porovnávají se poslední známá hodnota, průměr prvních pěti minut a Ridge model.
- První tři dostupné dny slouží k počátečnímu učení. Každý další den se
  vyhodnocuje na modelu naučeném pouze ze starších dnů; potřebujete alespoň čtyři dny.
- Používají se pouze úplné čtvrthodiny. Stejné duplicitní řádky se sloučí;
  konfliktní hodnoty stejného času se ohlásí jako chyba.
- Datum se načítá z `Timestamp`, čas ze sloupce `Čas` a hodnota z
  `ODCHYLKA ČR [MW]`. Pásmo se ignoruje; období nesmí zahrnovat změnu času.

**Správná polarita znamená správné znaménko průměru dalších deseti minut.**
Neznamená stejnou polaritu ve všech deseti minutách ani pravděpodobnost úspěchu
konkrétní budoucí předpovědi. MAE udává průměrnou absolutní chybu v MW.
Jde o historický test; šest dní nestačí k ověření dlouhodobé spolehlivosti.

Výstupy ve složce `predikce`:

| Soubor | Obsah |
| --- | --- |
| `vysledky.txt` | Shrnutí a metriky |
| `metriky.csv` | Souhrnné porovnání metod |
| `metriky_po_dnech.csv` | Chyby podle testovacího dne |
| `predikce.csv` | Skutečnosti a odhady pro každou testovací čtvrthodinu a minutu |
| `predikce.png` | Graf průměrů a chyby podle horizontu předpovědi |

### Chyba `IncompleteInputError`

Verze končící na řádku se `spravna_polarita_prumeru_pct` je neúplná: chybí
zbytek slovníku i konec programu. Použijte celý soubor
[`predikce_ctvrthodin.py`](predikce_ctvrthodin.py) nebo výše odkazovaný notebook.

## Analýza polarity

`analyza_polarity.py` analyzuje délky kladných a záporných úseků, četnosti změn
polarity a časové trendy a vytváří graf odchylky:

```bash
python analyza_polarity.py "odchylka_cr_*.csv" --out vysledky
```
