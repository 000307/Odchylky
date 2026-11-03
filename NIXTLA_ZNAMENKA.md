# Nixtla a LightGBM: pravděpodobnosti znamének dalších deseti minut

[Spustit v Google Colabu](https://colab.research.google.com/github/000307/Odchylky/blob/main/predikce_nixtla_znamenka_colab.ipynb)

Skript [`predikce_nixtla_znamenka.py`](predikce_nixtla_znamenka.py) porovnává
MLForecast s regresorem LightGBM, přímé klasifikátory znamének LightGBM a jejich
kombinaci 50:50. Z pěti známých minut odhaduje pro každou z dalších deseti minut
pravděpodobnost kladné a záporné hodnoty. Hlavními ukazateli jsou Brierovo skóre
a kalibrace pravděpodobností.

Knihovny fungují lokálně i v Colabu bez API klíče. Výsledky provedeného
srovnání jsou v [`VYSLEDKY_NIXTLA.md`](VYSLEDKY_NIXTLA.md); návaznost na starší
experimenty uvádí [`SROVNANI_MODELU.md`](SROVNANI_MODELU.md).

## Spuštění

V Colabu spusťte buňky postupně shora dolů a nahrajte všechna vstupní CSV.
Notebook stáhne skripty a závislosti, spustí vyhodnocení, zobrazí tabulky a grafy
a nabídne výstupy v ZIP souboru. Vstupní CSV se do veřejného repozitáře nenahrávají.

Pro místní spuštění použijte Python 3.10 nebo novější a tyto soubory ve stejné
složce: `predikce_nixtla_znamenka.py`, `nixtla_sign_core.py`,
`sign_calibration.py`, `predikce_linearni_regrese.py` a `requirements_nixtla.txt`.
Poslední Python soubor poskytuje společné načítání původních CSV.

```bash
python -m pip install -r requirements_nixtla.txt
python predikce_nixtla_znamenka.py "odchylka_cr_*.csv" --out predikce_nixtla
```

Vstupní sloupce jsou `Timestamp`, `Čas` a `ODCHYLKA ČR [MW]`. Datum se čte
z `Timestamp`, čas ze sloupce `Čas`. Použijte minutová data z období bez změny
letního a zimního času. Neúplné čtvrthodiny se vynechávají bez interpolace;
stejné duplicity se sloučí a konfliktní hodnoty se odmítnou.

## Porovnávané modely

**MLForecast + LightGBM.** Každá úplná historická čtvrthodina tvoří samostatnou
řadu. Skutečná knihovna Nixtla MLForecast připravuje pět zpoždění a trénuje deset
přímých regresních modelů. Trénovací počátky jsou omezené tak, aby každý model
vždy používal pouze prvních pět minut k předpovědi příslušné budoucí minuty.
Žádné zpoždění nepřechází přes mezeru mezi neúplnými intervaly.

Regresor předpovídá MW. Pravděpodobnost znaménka vzniká až samostatnou
logistickou kalibrací těchto odhadů na minulém kalibračním dni. Není odvozena
z konformního intervalu ani automaticky poskytována bodovou regresí.

**Přímý LightGBM klasifikátor.** Pro každý z deseti horizontů se trénuje
samostatný model cíle `budoucí hodnota > 0`. Vstup tvoří pět známých hodnot
a jejich průměr, variabilita, sklon, rozdíl první a poslední hodnoty, minimum
a maximum. Tyto doplňky se počítají pouze ze známých pěti minut. Porovnávají
se původní i dodatečně kalibrované pravděpodobnosti klasifikátoru.

**Kombinace 50:50.** Pro každou budoucí minutu jde o aritmetický průměr
pravděpodobností kalibrovaného MLForecast a kalibrovaného klasifikátoru.
Váha 50:50 je předem pevná; nevybírá se podle testovacích výsledků. Průměr
dvou kalibrovaných modelů nemusí být sám kalibrovaný, proto se kombinace
vyhodnocuje samostatně.

LightGBM používá předem zvolené omezené nastavení: 100 stromů, nejvýše sedm
listů, hloubku tři, rychlost učení 0,05, nejméně 15 trénovacích případů
v listu a L2 regularizaci 1. Třídy se automaticky nevyvažují. Při jediném
znaménku v tréninku se použije označená vyhlazená konstantní pravděpodobnost.

Srovnání doplňuje zachování posledního známého znaménka a vyhlazená četnost
znamének v dostupné minulosti. Pravděpodobnosti 0/1 u posledního znaménka
jsou deterministická reference, nikoli kalibrovaný pravděpodobnostní model.

Srovnání nyní obsahuje také **Poslední hodnota + sigmoid**: poslední známá hodnota
v MW se opakuje do všech deseti horizontů a stejný minulý kalibrační den
slouží k logistickému odhadu pravděpodobnosti. Kalibrátor má stejné vstupy
(skóre, horizont a interakci) a stejnou regularizaci jako u MLForecast.
Tato pravděpodobnostní reference může změnit odhad znaménka; původní
pravidlo opakování znaménka páté minuty zůstává samostatně vyhodnocené.

## Oddělení učení, kalibrace a testu

Výchozí vyhodnocení potřebuje alespoň čtyři dostupné dny. Pro každý
testovací den platí:

1. Poslední předchozí dostupný den slouží pouze ke kalibraci.
2. Starší dny slouží k učení základních modelů.
3. Následující testovací den se použije až k vyhodnocení předpovědí.

Při prvním testovacím dni tedy slouží první dva dny k učení a třetí den
ke kalibraci. Základní modely se potom znovu neučí na kalibračním dni.
S dalšími testovacími dny se celý postup posouvá dopředu. Přesné hranice
a počty intervalů uvádí `rozdeleni_dat.csv`.

Kalibrátor je malá regularizovaná logistická regrese sdílená mezi horizonty.
Používá skóre modelu, horizont a jejich interakci; u klasifikátoru je skórem
logit původní pravděpodobnosti. Standardizace i kalibrace používají pouze
kalibrační den. Regularizace `C=0.1` je předem pevná. Pokud kalibrační den
obsahuje jen jednu třídu, použije se označená vyhlazená četnost.

Nastavení lze změnit pouze před vyhodnocením, například:

```bash
python predikce_nixtla_znamenka.py "odchylka_cr_*.csv" --train-days 3 --seed 20261102 --out predikce_nixtla
```

## Jak číst výsledky

Brierovo skóre je průměr `(p_kladna − skutecnost_kladna)²`; **nižší hodnota
je lepší**, nula odpovídá dokonalé předpovědi. Logaritmická ztráta je také
lepší při nižší hodnotě. Krajní pravděpodobnosti se pro výpočet logaritmu
numericky omezují, aby chyba deterministické reference nevedla k nekonečnu.

Kalibrační diagram porovnává průměrnou předpovězenou pravděpodobnost se
skutečným podílem kladných hodnot v deseti pevných intervalech. Tabulka
uvádí také počty případů. ECE shrnuje rozdíly mezi těmito podíly, ale závisí
na zvoleném dělení a samo neříká, zda je model užitečnější než konstantní
četnostní předpověď. Posuzujte je společně s Brierovým skóre.

Přesné nulové skutečnosti se vynechávají z binárního učení, kalibrace
a pravděpodobnostního skórování. Jejich počet se vykazuje. Experiment hodnotí
znaménka **jednotlivých minut**. Průměr pravděpodobností přes deset minut
není pravděpodobností znaménka desetiminutového průměru.

| Soubor ve výstupní složce | Obsah |
| --- | --- |
| `vysledky.txt` | Shrnutí postupu a vyhodnocení |
| `predikce.csv` | Skutečné hodnoty a pravděpodobnosti pro každou minutu a metodu |
| `metriky.csv` | Brier, log-loss, ECE a správnost znamének |
| `metriky_po_horizontech.csv` | Výsledky podle kroku předpovědi |
| `metriky_po_dnech.csv` | Výsledky podle testovacího dne |
| `kalibrace.csv` | Pevné pravděpodobnostní intervaly, počty a skutečné podíly |
| `rozdeleni_dat.csv` | Oddělení základního učení, kalibrace a testu |
| `kalibrace_fitu.csv` | Velikost a případné náhradní řešení kalibračních fitů |
| `porovnani_brier.png` | Srovnání Brierova skóre |
| `kalibrace.png` | Kalibrační diagramy |

## Předpověď nového intervalu

Po získání páté hodnoty zadejte začátek čtvrthodiny a všech pět měření v MW
v časovém pořadí. Ukázkové hodnoty nahraďte skutečnými:

```bash
python predikce_nixtla_znamenka.py "odchylka_cr_*.csv" --first-five 10 20 30 40 50 --start "2026-09-29 12:00" --out nova_predikce_nixtla
```

Skript opět oddělí nejnovější dostupný historický den pro kalibraci od
staršího základního učení. Použije pouze intervaly dokončené nejpozději
na začátku předpovídané čtvrthodiny a vytvoří `predikce_noveho_intervalu.csv`.
Stejnou možnost obsahuje poslední, výchozím nastavením vypnutá buňka Colabu.

## Meze interpretace a testy

Dosavadní tři testovací dny byly prohlíženy již při předchozích experimentech.
Jde proto o retrospektivní srovnání; pro konečné potvrzení zvolené varianty
je potřeba nová, dosud nepoužitá historie. Modely ani kalibrátory se nesmějí
ladit podle těchto testovacích výsledků. Minuty uvnitř stejného intervalu
jsou závislé a několik dní neověřuje dlouhodobou stabilitu pravděpodobností.

```bash
python -m unittest discover -s tests -p "test_nixtla_signs.py" -v
```
