# Srovnání hlavních dosavadních modelů

**Pro původní cíl, správné znaménko jednotlivých budoucích minut, stále nejlépe vychází zachovat znaménko páté minuty: 81,39 %.** Nižší Brierovo skóre měří jiný aspekt: kvalitu pravděpodobnosti včetně její přiměřené jistoty.

Všechny řádky používají stejné skutečnosti ze 288 testovacích čtvrthodin (26.–28. září 2026). Při sestavení tabulky se ověřila shoda časů, horizontů i skutečných měření. Správnost znamének níže zahrnuje všech 2 880 minut; Brier, log-loss a ECE vynechávají jednu přesnou nulu a používají 2 879 případů.

| Metoda | Správná znaménka | Brier ↓ | ECE ↓ |
| --- | ---: | ---: | ---: |
| Poslední znaménko | 81,39 % | 0,18583 | 0,18583 |
| Bayes AR(5) MCMC | 79,58 % | 0,13511 | 0,06276 |
| Poslední hodnota + sigmoid | 78,54 % | 0,14737 | 0,06491 |
| Lineární regrese | 77,92 % | 0,22056 | 0,22056 |
| LightGBM + sigmoid | 77,15 % | 0,15382 | 0,03920 |
| LightGBM bez kalibrace | 77,12 % | 0,15472 | 0,06558 |
| Kombinace 50:50 | 76,88 % | 0,14904 | 0,04895 |
| MLForecast + sigmoid | 76,77 % | 0,14854 | 0,06316 |
| Četnost v minulosti | 71,08 % | 0,21272 | 0,07971 |

## Jak výsledek použít

- **Rozhodnutí pouze kladná/záporná:** nejvyšší dosavadní správnost má původní opakování pátého znaménka.
- **Pravděpodobnost a její jistota:** nejnižší dosavadní Brier má Bayes AR(5), přibližně 0,1351.
- **Jednoduchá pravděpodobnostní reference:** samotná poslední hodnota se sigmoidní kalibrací má Brier 0,1474; nové složitější Nixtla varianty ji v tomto skóre nepřekonaly.
- **Kalibrační diagram / ECE:** kalibrovaný klasifikátor LightGBM má nejnižší celkové ECE; ECE závisí na deseti zvolených intervalech a může skrýt rozdíly mezi dny.

Zlepšení Brierova skóre nemusí zvýšit počet správných rozhodnutí při hranici 0,5. Model může častěji zvolit chybné znaménko a současně dávat méně přehnaně jisté pravděpodobnosti.

## Férová interpretace jednoduchého pravidla

Opakování páté hodnoty samo neposkytuje pravděpodobnost. Dosavadní řádek **Poslední znaménko** přiřazuje předpovězenému znaménku jistotu 100 % (pravděpodobnost 0 nebo 1). Každá chyba proto znamená Brierovu ztrátu 1. Je to užitečná deterministická reference, ale její vyšší Brier nedokazuje, že poslední hodnota nese méně informace.

Proto byla dodatečně doplněna **Poslední hodnota + sigmoid**. Učí se odhad pravděpodobnosti pouze na minulém kalibračním dni, stejně jako nové Nixtla modely. Zachování posledního znaménka a kalibrovaná poslední hodnota jsou samostatné metody: kalibrace může posunout pravděpodobnost přes 0,5 a změnit i předpovězené znaménko.

## Rozdíly v protokolu a meze závěrů

Historická lineární regrese a Bayes AR(5) používaly všechny starší dny k odhadu parametrů. Nixtla / LightGBM a kalibrovaná poslední hodnota vyhrazují poslední minulý den pro kalibraci. Je shodné testovací období a dostupná minulost; způsob využití této minulosti je odlišný. Tato tabulka porovnává dosavadní pracovní postupy, nikoli pouze samotné algoritmy při totožném učení.

Brier a ECE u původní lineární regrese vznikají stejně jako u posledního znaménka převodem bodové předpovědi na 0/1, nejde o její vlastní odhad nejistoty.

Výsledky jsou retrospektivní na opakovaně prohlížených třech dnech. Dodatečně přidaný kalibrovaný jednoduchý model tuto skutečnost nemění; nebyly podle testu laděny jeho parametry. Konečné ověření vyžaduje novou historii. Minuty uvnitř čtvrthodiny nejsou nezávislé vzorky.

[Podrobný Nixtla experiment a kalibrační grafy](VYSLEDKY_NIXTLA.md) · [Agregovaná data společné tabulky](vysledky_nixtla/srovnani_vsech_modelu.csv) · [Google Colab](https://colab.research.google.com/github/000307/Odchylky/blob/main/predikce_nixtla_znamenka_colab.ipynb)
