# Bayesovský autoregresní model s MCMC

[Spustit v Google Colabu](https://colab.research.google.com/github/000307/Odchylky/blob/main/predikce_bayes_ar_colab.ipynb)

Skript [`predikce_bayes_ar.py`](predikce_bayes_ar.py) předpovídá zbývajících
deset minut čtvrthodiny z prvních pěti známých hodnot. Pro každou budoucí
minutu vrací odhad v MW, pravděpodobnost kladného a záporného znaménka a
90% predikční interval. Samostatně odhaduje také průměr dalších deseti minut.

## Spuštění

V Colabu spusťte buňky postupně shora dolů a nahrajte všechna vstupní CSV.
Notebook stáhne potřebné skripty, spustí vyhodnocení, zobrazí tabulky a grafy
a nabídne výsledky jako ZIP. CSV zůstávají ve vašem prostředí Colabu;
do veřejného repozitáře se nenahrávají.

Pro místní spuštění použijte Python 3.10 nebo novější. Ve stejné složce musí
být soubory `predikce_bayes_ar.py`, `bayes_ar_core.py` a
`predikce_linearni_regrese.py`; poslední z nich poskytuje společné načítání CSV.

```bash
python -m pip install -r requirements_bayes.txt
python predikce_bayes_ar.py "odchylka_cr_*.csv" --out predikce_bayes
```

Vstupní sloupce jsou `Timestamp`, `Čas` a `ODCHYLKA ČR [MW]`. Datum se čte
z `Timestamp`, čas ze sloupce `Čas`. Použijte minutová data z období bez změny
letního a zimního času. Neúplné čtvrthodiny se vynechávají bez interpolace;
stejné duplicity se sloučí a konfliktní hodnoty se odmítnou.

## Model a MCMC

Výchozí model je **AR(5)**:

```text
y[t] = alpha + phi[1] × y[t−1] + ... + phi[5] × y[t−5] + chyba[t]
chyba[t] ~ Normal(0, sigma²)
```

Pět známých minut tvoří počáteční historii pro rekurzivní odhad následujících
deseti minut. Parametry se učí z minulých úplných čtvrthodin. Trénovací okna
zůstávají uvnitř jednotlivých intervalů, takže nevznikají přechody přes mezery
v datech. Standardizace používá pouze trénovací data.

Parametry a rozptyl se odhadují **Gibbsovým MCMC vzorkováním**. Výchozí nastavení
má čtyři nezávislé řetězce, 1 000 zahřívacích iterací a 2 000 uložených tahů
na řetězec. Není potřeba instalovat PyMC; diagnostiku poskytuje ArviZ.

Prior je předem pevný, nevybírá se podle výsledků testovacích dnů. Ve
standardizovaných jednotkách platí `beta | sigma² ~ Normal(m0, sigma² V0)`
a `sigma² ~ InverseGamma(shape=2, scale=1)`, kde
`m0 = [0, 0.8, 0, ...]` a `V0 = diag(1, 0.25, ...)`.
Model neomezuje tahy na stacionární AR proces; podíl stacionárních tahů se
vykazuje. Predikce používá všechny ponechané MCMC tahy včetně nestacionárních.

Pravděpodobnosti znaménka zahrnují nejistotu parametrů i náhodné budoucí
výkyvy. Získávají se průměrováním podmíněných normálních pravděpodobností přes
MCMC tahy. Predikční intervaly vycházejí ze simulovaných budoucích průběhů.
Pravděpodobnost znaménka desetiminutového průměru zahrnuje závislosti mezi
minutami; není průměrem deseti pravděpodobností kladné hodnoty.

Nastavení lze změnit například takto:

```bash
python predikce_bayes_ar.py "odchylka_cr_*.csv" --lags 5 --chains 4 --tune 1000 --draws 4000 --seed 20261102 --out predikce_bayes
```

`--lags` dovoluje řád 1 až 5. Zvolený řád a ostatní parametry je vhodné určit
před testováním; výběr nejlepší varianty podle testovacího období by zkreslil
odhad přesnosti.

## Časové vyhodnocení

Výchozí vyhodnocení potřebuje alespoň čtyři dostupné dny. První tři slouží
k počátečnímu učení. Každý další den se testuje s parametry odhadnutými jen
ze starších dnů. Budoucí hodnoty testovaného intervalu se při předpovědi
nepoužívají.

Porovnává se bayesovský model, poslední známá hodnota, průměr prvních pěti
minut, běžná lineární regrese a pravděpodobnost odvozená z četnosti znamének
v trénovací historii. Hodnotí se:

- správnost znamének jednotlivých minut a správnost podle horizontu;
- správnost všech deseti znamének současně a znaménka desetiminutového průměru;
- MAE v MW, Brierovo skóre a logaritmická ztráta;
- kalibrace pravděpodobností a pokrytí 90% predikčních intervalů.

Nižší Brierovo skóre i logaritmická ztráta jsou lepší. U deterministických
srovnávacích metod jsou pravděpodobnosti 0 nebo 1; nejde o kalibrované
pravděpodobnostní modely. Přesné nulové skutečnosti se vynechávají z binárního
pravděpodobnostního skórování. Správnost jednotlivých znamének a správnost
desetiminutového průměru jsou různé ukazatele.

Diagnostika MCMC používá pořadově normalizované R-hat a efektivní velikost
vzorku. Výchozí kontrola požaduje **R-hat ≤ 1,01** a **bulk i tail ESS ≥ 400**;
při nevyhovujícím výsledku se běh zastaví. V takovém případě zkontrolujte
diagnostiku a zvyšte počet iterací. Dobrá konvergence potvrzuje kvalitu
vzorkování za předpokladů modelu, nikoli jeho předpovědní přesnost.

| Soubor ve výstupní složce | Obsah |
| --- | --- |
| `vysledky.txt` | Shrnutí modelu, diagnostiky a vyhodnocení |
| `predikce.csv` | Minutové skutečnosti, odhady a pravděpodobnosti |
| `prumery.csv` | Předpovědi průměru dalších deseti minut |
| `metriky.csv` | Souhrnné porovnání metod |
| `metriky_po_horizontech.csv` | Výsledky pro jednotlivé kroky předpovědi |
| `kalibrace.csv` | Předpovězené pravděpodobnosti a pozorované podíly |
| `diagnostika_mcmc.csv` | Parametry, R-hat a efektivní velikosti vzorku |
| `posterior_<datum>.npz` | Uložené MCMC tahy pro jednotlivé trénovací běhy |
| `predikce_polarity.png` | Grafy předpovědí a jejich vyhodnocení |
| `mcmc_stopy.png` | Průběhy MCMC řetězců |

## Předpověď nového intervalu

Po získání páté hodnoty zadejte začátek čtvrthodiny a všech pět měření v MW
v časovém pořadí. Ukázkové hodnoty nahraďte skutečnými:

```bash
python predikce_bayes_ar.py "odchylka_cr_*.csv" --first-five 10 20 30 40 50 --start "2026-09-29 12:00" --out nova_predikce_bayes
```

Pro učení se použijí pouze historické intervaly dokončené nejpozději na
začátku předpovídané čtvrthodiny. Vznikne `predikce_noveho_intervalu.csv`
s deseti minutovými předpověďmi a `souhrn_predikce.csv` se souhrnem.
Stejnou možnost obsahuje poslední, výchozím nastavením vypnutá buňka Colabu.

## Meze interpretace

Model předpokládá lineární autoregresní vztah a normální chyby se společným
rozptylem v daném trénovacím běhu. Prudké změny režimu nebo volatility mohou
předpověď zhoršit. Udávané pravděpodobnosti nejsou automaticky kalibrované
a MCMC samo nezaručuje zlepšení proti zachování posledního znaménka.
U tohoto modelu existuje i analytický posterior; Gibbsovy tahy se vůči němu
kontrolují v testech. MCMC zde slouží k bayesovskému odhadu parametrů,
předpovědní přesnost ověřuje oddělený časový backtest.
Několik testovacích dnů nestačí k potvrzení dlouhodobé přesnosti; výsledky
je potřeba ověřit na delší oddělené historii.

## Testy

```bash
python -m unittest discover -s tests -v
```

Testy kontrolují Gibbsův sampler proti analytickému posterioru, pravděpodobnosti
a nejistotu budoucích hodnot, pořadí zpoždění a oddělení tréninku od budoucích dat.
