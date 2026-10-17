# Elo: die Stärke Partie für Partie – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-elo-demo.streamlit.app/)**

Siebtes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `bradley-terry-demo`: Bradley-Terry schätzt die Stärken aus **allen Partien auf einmal**. Elo
schreibt die Wertung **nach jeder Partie** fort, mit fester Verstärkung K. Das ist ein Filter; gemessen wird, was K kostet und bringt, wenn die Stärken stillstehen
oder wandern. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel (Anfangsstärken aus dem Generator Version 1, wortgleich; danach eine Irrfahrt der Stärken
als lokales Modell); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Wie wählt man K, und wann schlägt das Fortschreiben einen Schätzer über alle Partien? **K ist ein Kompromiss zwischen Rauschen und Verzögerung, den allein die Drift
der Stärken bestimmt** (U-Form im Fehler; das beste K wächst mit der Drift wie die Verstärkung eines Kalman-Filters). **Elo gewinnt gegen Bradley-Terry über alle
Partien ab etwa σ = 0,03 Drift je Runde und verliert ohne Drift.** Der Preis ist **Pfadabhängigkeit:** dieselben Partien in anderer Reihenfolge ergeben andere
Wertungen und oft einen anderen Spitzenreiter. Ein **Anlauf-K** hilft einem Neuling, der vom Feld abweicht, und schadet einem, der ihm gleicht.

## Modell

- **Erwartung und Aktualisierung:** E = σ(θ_i − θ_j); θ_i += k (S − E), θ_j −= k (S − E) mit k = K/c, c = 400/ln 10 ≈ 173,7 Punkte je logistische Einheit (400 Punkte =
  Faktor 10 in den Gewinnchancen). Bei gleichem K bleibt die Summe der Wertungen erhalten; mit Anlauf-K nicht (der Neuling gewinnt mehr, als der Gegner verliert).
- **Anlauf-K:** K_g = K + (K0 − K)/(1 + g/g0) nach g eigenen Partien, g0 = 10.
- **Stärken:** Anfangsstärken aus `make_strengths`, danach Irrfahrt mit Schritt σ je Runde (Normalverteilung per Box-Muller aus der Gleichverteilten des Generators).
  In jeder Runde spielt jeder genau einmal, zufällig gepaart (bei ungeradem n setzt einer aus). Optional kommt ein **Neuling** (Spieler 0) in einer Runde dazu, mit
  Startwertung Feldmittel und wahrer Stärke im Abstand `gap` vom Feld.
- **Vergleich:** Bradley-Terry über alle Partien (Newton mit zentriertem Schritt, Ridge mit dem Orakel-λ = 12/(δ²(n²−1))); unabhängig von der Reihenfolge.
- **Kalman-Verstärkung (nur Vergleichswert):** k* = P mit P = (−qI + √(q²I² + 4Iq))/(2I), q = σ², I = mittlere Information p(1 − p) je Partie ≈ 0,20; in Punkten
  K* = c·k*. Nur in der Simulation bekannt.
- **Maße:** Verfolgungsfehler (RMSE der zentrierten Wertungen gegen die zentrierten wahren Stärken, ab Runde 30), Endfehler und τ gegen die wahre Endstärke,
  Log-Verlust der Vorhersage vor der Partie.
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 100 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed; Experiment 4: 40 Läufe mit je 16 Reihenfolgen), je mit einem live berechneten Verdikt;
  feste Instanzen mit 8 Spielern und 150 Runden (Experiment 5: 100 Runden, Neuling in Runde 60), unabhängig vom Regler.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit `siegtabelle-demo`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 100 feste Läufe, 8 Spieler, Stärkeabstand 0,3, 150 Runden, wenn nicht anders genannt. Fehler in logistischen Einheiten (1 Einheit = 173,7 Punkte).

| Frage | Befund | Test |
|---|---|---|
| U-Form von K (ohne Drift) | Verfolgungsfehler bei K = 2 / 5 / 10 / 20 / 40 / 80 / 160: 0,540 / 0,396 / 0,276 / **0,241** / 0,319 / 0,482 / 0,763. | `test_k_is_a_trade_off_with_a_u_shape` |
| Bestes K nach Drift | σ = 0 / 0,03 / 0,06: K = 20; σ = 0,12: K = 40 (Fehler 0,505 gegen 0,607 bei K = 20). Bei σ = 0,06 ist K = 20 (0,353) kaum besser als K = 40 (0,357). | `test_the_best_k_moves_with_the_drift` |
| Kalman-Verstärkung | K* bei σ = 0,03 / 0,06 / 0,12 / 0,24: 11,6 / 23,0 / 45,3 / 88,3 Punkte; bestes K im Raster nach Fehler 20 / 28 / 56 / 112, nach Log-Verlust 20 / 28 / 40 / 80. **Verdopplung von σ verdoppelt K\*; das beste K folgt**, liegt bei kleiner Drift aber über der Formel (Kaltstart, unsicherer Gegner). | `test_the_best_k_scales_like_the_kalman_gain` |
| Elo gegen Bradley-Terry über alle Partien (Endfehler) | σ = 0: **Elo 0,222, BT 0,148** (τ 0,880 gegen 0,950); σ = 0,015: 0,239 / 0,192; **σ = 0,03: 0,249 / 0,258 (gleich)**; σ = 0,045: 0,292 / 0,352; σ = 0,06: 0,351 / 0,455 (τ 0,772 gegen 0,701); σ = 0,12 (K = 40): **0,552 / 0,924** (τ 0,821 gegen 0,631). | `test_elo_loses_without_drift_and_wins_with_it` |
| Pfadabhängigkeit (konstante Stärken, 16 Reihenfolgen) | Streuung der Endwertung bei K = 5 / 10 / 20 / 40 / 80: 0,030 / 0,076 / **0,157** / 0,275 / 0,443 (bei K = 20 rund 27 Punkte); **anderer Spitzenreiter in 15 / 33 / 75 / 98 / 100 %** der Läufe. Bradley-Terry: Streuung 0. | `test_the_path_dependence_is_large` |
| Neuling im Feld (Abstand 1, nach 20 eigenen Partien) | Fehler bei K = 10 / 20 / 40: 0,733 / 0,531 / 0,340; **K = 20 mit Anlauf 60: 0,336**, Anlauf 120: 0,376; Mittel der ersten 40 Partien 0,569 (K = 20) gegen 0,412 (Anlauf 60). | `test_a_provisional_k_helps_a_newcomer_who_differs_from_the_field` |
| … nach Abstand vom Feld (K = 20 fest gegen Anlauf 60) | Abstand 0: **0,201 gegen 0,318** (Anlauf schadet); 0,5: 0,295 / 0,322; 1,0: **0,531 / 0,336**; 2,0: **1,250 / 0,787**. | `test_the_provisional_k_hurts_a_newcomer_who_equals_the_field` |

## Befunde und Korrekturen gegenüber dem Plan

- **Anlauf-K bricht die Nullsumme:** Der Plan führte die Nullsumme als Orakel-Eigenschaft für Elo. Mit Anlauf-K gilt sie nicht mehr: Der Neuling mit größerem K gewinnt
  mehr, als der Gegner verliert (die Summe der Wertungen driftet). Der Test fand es beim Bau der Invarianten; die App sagt es, das Orakel prüft die Nullsumme nur für
  festes K (`test_a_provisional_k_breaks_the_zero_sum_and_the_sum_follows_the_k_difference`).
- **Faktor Horizont:** Bei nur 60 Runden ist K = 40 überall am besten (Kaltstart); die Demo rechnet mit 150 Runden, das beste K ist dort 20 bis 40.
- **Neuling in der Messreihe:** Dort ein neunter Spieler, hier Spieler 0 des Turniers (n bleibt höchstens 16); die Werte (0,531 und 0,336 nach 20 Partien) weichen deshalb
  von der Vorab-Messreihe (0,458 und 0,363) ab.
- **Seed:** Der Standard-Seed 44 ist typisch (Endfehler Elo 0,35 gegen Bradley-Terry 0,51); der erste Seed (17) zeigte den untypischen Fall, in dem Bradley-Terry vorn liegt.
- **Geschwindigkeit:** Der Elo-Lauf rechnet in Skalaren (`math`) statt in `numpy` je Partie (rund zehnmal schneller); die Orakel-Tests vergleichen ihn mit der Lehrbuch-
  Formulierung in Elo-Punkten.

## Abgrenzung

- [Kalman-Filter](https://sebastianhanisch-kalman-filter-demo.streamlit.app/) (Zustandsschätzung und Filter): Filter mit Verstärkung aus der mitgeführten Unsicherheit;
  hier ist die Verstärkung fest, und die Kalman-Verstärkung dient nur als Vergleichswert.
- **Bradley-Terry** (Stück 5): der Schätzer über alle Partien; dieses Stück ersetzt ihn durch den fortlaufenden und misst, was das kostet (Pfadabhängigkeit) und bringt
  (Verfolgung).
- **Glicko und TrueSkill:** statt fester Verstärkung eine mitgeführte Unsicherheit (K je Spieler und Partie); hier nicht umgesetzt.

## Ehrliche Grenzen

- **Elo verliert gegen Bradley-Terry, wenn die Stärken stillstehen** (Fehler 0,222 gegen 0,148); das steht neben dem Gewinn bei Drift.
- **Das beste K hängt vom Horizont ab** (Kaltstart) und vom Fehlermaß; K nie als Konstante verkaufen. Der Wechsel bei σ ≈ 0,03 gilt für 150 Runden und 8 Spieler.
- **Pfadabhängigkeit ist groß:** bei K = 20 wechselt in 75 % der Läufe der Spitzenreiter; Elo sagt nichts über die Sicherheit (Thema von Glicko).
- **Irrfahrt ist eine Annahme:** reale Stärken driften nicht unbedingt so; Verletzungen, Formkurven und Sprünge sind anders.
- **Kalman-K\* ist ein Näherungswert** für einen Einzelspieler (Gegner auch unsicher) und liegt bei kleiner Drift unter dem besten K.
- **Anlauf-K hilft nur Abweichenden** und bricht die Nullsumme.
- **Alles simuliert**, ein Generator (nur die Anfangsstärken), n ≤ 16.

## Quellen

- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present.* Arco Publishing, New York (2. Auflage 1986; nach Katalogeinträgen, das Buch wurde nicht eingesehen).
- Im Schach sind K = 40 für Neulinge, K = 20 und K = 10 für Spieler ab 2400 gebräuchlich (FIDE Handbook, B.02 FIDE Rating Regulations, gültig ab 1. März 2024, Abschnitt 8.3.3,
  eingesehen 2026-10-17: K = 40 für Neue bis 30 Partien, K = 20 solange die Wertung unter 2400 bleibt, K = 10 sobald die veröffentlichte Wertung 2400 erreicht hat, auch wenn sie später sinkt;
  daneben K = 40 bis Ende des Jahres des 18. Geburtstags bei Wertung unter 2300).
- Glickman, M. E. (1999): Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society Series C (Applied Statistics)*
  48 (3), 377–394, doi:10.1111/1467-9876.00159 (ein nichtiteratives Verfahren, das den Elo-Algorithmus um die Unsicherheit der Schätzung erweitert;
  eingesehen 2026-10-17 im Volltext der Autorenseite glicko.net/research/glicko.pdf: Zusammenfassung und Abschnitt 3.3, Gleichungen (9) bis (13), Elo als Sonderfall bei bekannten Gegnerstärken).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472.

## Tests

80 Tests, rund drei Minuten (der erste Seitenaufruf rechnet die Experimente): Einheiten gegen die von Hand gerechneten Mini-Instanzen (`test_rating.py`, `test_drift.py`, `test_batch.py`), Generator (`test_generator.py`,
wortgleich mit der Wurzel), **Orakel** (`test_oracle_elo.py`: Elo-Verlauf gegen die Lehrbuch-Formulierung in Elo-Punkten (10^(ΔR/400)) mit Anlauf-K und Neuling,
Kalman-Verstärkung gegen die Riccati-Iteration und `scipy.linalg.solve_discrete_are`, Bradley-Terry gegen `scipy.optimize` und Reihenfolgen-Unabhängigkeit,
Pfadabhängigkeit), Presets, ausblendbare Regler (`test_hidden_widget_state.py`), Analyse und Verdikte (`test_evaluation.py`), alle Zahlen dieses README
(`test_claims.py`), PDF, Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, 40 Mutanten, nur gegen die Orakel-Tests): 13 getötet, 27 überlebt, Rate 32 %. Die 27 Überlebenden wurden einzeln
auf einer Kopie gegen die Einheitentests geprüft: **17 getötet** (Orakel-λ, Normalverteilung, Neuling-Abstand, Anlauf-K, Kalman-Punkte, Umrechnung in Punkte und zehn im
gemeinsamen Generator durch `test_generator.py`), **6 echte Lücken** (die Fehlerrechnung des Neulings: Zeitpunkt, Seite, beide Vorzeichen; vorher prüften die Tests nur Länge
und Vorzeichen) mit einem Handwert-Test geschlossen und danach einzeln erneut getötet, **4 äquivalent** (Zentrierung von bereits zentrierten Größen, Gleitkomma-Gleichheit,
`c ≥ 0` bei `c = 0`). Die Rate nur gegen das Orakel ist niedrig, weil das Orakel den Elo-Verlauf und die Verstärkung prüft, nicht die Handwerte und die Kennzahlen.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie; liefert die Anfangsstärken) |
| `elo_drift.py` | Irrfahrt der Stärken, Paarung, Neuling, Umordnen der Partien |
| `elo_rating.py` | Elo-Aktualisierung, Anlauf-K, Verlauf, Kalman-Verstärkung, Umrechnung in Punkte |
| `elo_batch.py` | Bradley-Terry über alle Partien (Vergleich) |
| `elo_scenario.py` | Reglerwerte (`Settings`) und die Partien |
| `elo_evaluation.py` | Analyse, fünf Experimente über die festen Läufe, Verdikte |
| `elo_presets.py`, `elo_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `elo_visualization.py`, `elo_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Unsicherheit je Spieler** (Glicko, TrueSkill): eigene Stücke der Linie.
- **Rückwärts-Glätten und Gewichten über die Zeit** (WHR, TrueSkill Through Time): eigenes Stück; hier ist der Vergleich nur der Schätzer über alle Partien.
- **Elo-Erweiterungen:** Tordifferenz-Faktor, Heimvorteil und Remis gehören zu „Tore, Remis, Heimvorteil“.
- **Zu- und Abgang vieler Spieler, Untergrenzen der Wertung, K nach Wertungsstufe:** die Inflation und Deflation der Skala über die Zeit ist hier nur im Neuling-Experiment
  und im Bruch der Nullsumme sichtbar.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
