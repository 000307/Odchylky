# Lineární regrese: prvních 5 minut → dalších 10 minut

[Otevřít notebook v Google Colabu](https://colab.research.google.com/github/000307/Odchylky/blob/main/predikce_linearni_regrese_colab.ipynb)

Skript [`predikce_linearni_regrese.py`](predikce_linearni_regrese.py) se učí
z historických úplných čtvrthodin. Pět minutových hodnot tvoří vstup a model
`LinearRegression` odhadne deset následujících hodnot. Jde o běžnou lineární
regresi s konstantou, bez penalizace Ridge. Například z `12:00–12:04` odhadne
`12:05–12:14`. Model se učí vztah mezi začátkem a koncem historických intervalů;
nejde o pouhé prodloužení přímky proložené pěti aktuálními body.

## Vstupní data

Připravte minutové CSV se sloupci `Timestamp`, `Čas` a `ODCHYLKA ČR [MW]`, stejně
jako pro původní skript. Datum se čte z `Timestamp`, čas ze sloupce `Čas`.
Použijte soubory z období bez změny letního a zimního času. Pro učení se použijí
pouze úplné čtvrthodiny začínající v `:00`, `:15`, `:30` nebo `:45`.

CSV zůstávají u vás nebo v prostředí vašeho Colabu; není potřeba je nahrávat do
veřejného repozitáře.

## Historické vyhodnocení

V Google Colabu spusťte buňky shora dolů a nahrajte všechna vstupní CSV. Při
místním spuštění použijte Python 3.10 nebo novější:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
python predikce_linearni_regrese.py "odchylka_cr_*.csv" --out predikce_lr
```

Potřebujete alespoň čtyři dostupné dny. První tři slouží k počátečnímu učení;
každý další den se testuje na modelu naučeném pouze ze starších dnů. Výsledky
porovnávají lineární regresi s poslední známou hodnotou a s průměrem prvních
pěti minut.

Parametrem `--train-days` lze změnit počet prvních trénovacích dnů. Pro každý
fit je nutných alespoň šest úplných minulých čtvrthodin; pro praktické použití
je vhodná delší historie. Stejné duplicitní řádky se sloučí, konfliktní nebo
nekonečné hodnoty se odmítnou. Neúplné intervaly se vynechají bez interpolace.

| Soubor ve složce `predikce_lr` | Obsah |
| --- | --- |
| `vysledky.txt` | Shrnutí vyhodnocení |
| `metriky.csv` | MAE, RMSE a správnost znaménka pro jednotlivé metody |
| `metriky_po_dnech.csv` | Chyby podle testovacího dne |
| `predikce.csv` | Skutečné a odhadnuté hodnoty pro testovací intervaly |
| `predikce.png` | Graf předpovědí a chyby podle horizontu |

Správnost polarity se týká znaménka **průměru dalších deseti minut**. Neznamená
stejné znaménko v každé minutě. Výsledky na několika dnech samy o sobě
neprokazují dlouhodobou přesnost modelu.

## Předpověď nového intervalu

Zadejte začátek čtvrthodiny a pět již známých hodnot v MW v časovém pořadí.
Následující příkaz obsahuje pouze ilustrační hodnoty; nahraďte je skutečnými:

```bash
python predikce_linearni_regrese.py "odchylka_cr_*.csv" --first-five 10 20 30 40 50 --start "2026-09-29 12:00" --out nova_predikce_lr
```

Skript použije k učení jen úplné historické intervaly, které skončily nejpozději
v zadaném čase začátku. Vytvoří `predikce_noveho_intervalu.csv` s odhadem pro
`12:05–12:14` a `souhrn_predikce.csv` s průměrem prvních pěti minut, odhadem
průměru dalších deseti minut a odhadem průměru celé čtvrthodiny:

```text
odhad průměru 15 min = (5 × skutečný průměr prvních 5 min
                        + 10 × odhad průměru dalších 10 min) / 15
```

Předpověď je dostupná po získání páté hodnoty. Další skutečné hodnoty daného
intervalu do učení ani do vstupu této předpovědi nevstupují.

## Ověření kódu

Testy používají umělá data a kontrolují matematiku regrese, všech deset
časových horizontů, čtení CSV a oddělení minulosti od budoucích hodnot:

```bash
python -m unittest discover -s tests -v
```
