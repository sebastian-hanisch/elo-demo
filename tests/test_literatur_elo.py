"""Literatur-Referenzwerte für Elo (ORACLE-PLAYBOOK §5.1). Nur Werte, die in einer Quelle eingesehen wurden (Stand der Einsicht: 2026-10-17).

Umrechnung: Die Demo rechnet in θ-Einheiten (1 Einheit = SCALE = 400/ln 10 ≈ 173,72 Elo-Punkte, 0 = 1500) und mit k = K/SCALE. Die Tests rechnen die Quellwerte mit
θ = (R − 1500)/SCALE in die Demo-Einheiten und zurück (R = 1500 + SCALE·θ).

Werte
-----
1. Erwartung 0,64 bei 100 Punkten Unterschied und 0,76 bei 200 Punkten.
   Quelle: FIDE Handbook, B.02 FIDE Rating Regulations (effective from 1 January 2022 till 29 February 2024), Abschnitt 8.1, Tabelle 8.1.2 („Rtg Dif / H / L“): Zeile
   „99-106 .64 .36“ und Zeile „198-206 .76 .24“; ferner Zeile „0-3 .50 .50“. URL: https://handbook.fide.com/chapter/B022022 (per WebFetch eingesehen am 2026-10-17;
   die Zeilen kamen aus der Zusammenfassung des Abrufwerkzeugs, der Wortlaut der ganzen Tabelle ließ sich nicht abrufen). Unabhängig bestätigt durch
   https://en.wikipedia.org/wiki/Elo_rating_system (am 2026-10-17): „A player whose rating is 100 points greater than their opponent's is expected to score 64%“ und
   „if the difference is 200 points, then the expected score for the stronger player is 76%“; dort auch die Formel E_A = 1/(1 + 10^((R_B − R_A)/400)).
   Bewusst NICHT getestet: die Ränder der Tafelbereiche (z. B. 106 Punkte) und 400 Punkte (Tafel: „392-411 .92“). Die FIDE-Tafel stammt aus einer Normalverteilung, die
   Demo (und Wikipedia) rechnet logistisch (400 Punkte = 0,909); die Werte stimmen bei 100 und 200 Punkten auf zwei Stellen überein, am Rand der Bereiche nicht.
2. Beispielturnier der Wikipedia: Spieler A mit 1613 spielt gegen 1609 (Niederlage), 1477 (Remis), 1388 (Sieg), 1586 (Sieg), 1720 (Niederlage); Erwartungen je Partie
   0,51 / 0,69 / 0,79 / 0,54 / 0,35, Summe der gerundeten Werte 2,88, tatsächliche Punkte 2,5, K = 32, neue Wertung 1613 + 32·(2,5 − 2,88) = 1601.
   Quelle: https://en.wikipedia.org/wiki/Elo_rating_system, Abschnitt zum Rechenbeispiel (Beispielturnier mit fünf Runden), am 2026-10-17 per WebFetch eingesehen.
   Die Quelle rechnet mit auf zwei Stellen gerundeten Erwartungen; die Demo-Rechnung mit ungerundeten Werten gibt 1601,27, also gerundet ebenfalls 1601. Verglichen wird daher
   je Partie auf zwei Stellen und die neue Wertung gerundet auf ganze Zahlen.

Nicht eingesehen (siehe Bericht „offen“): Elo (1978), The Rating of Chessplayers, Past and Present (Buch); USCF-Erwartungstabelle im Original; weitere Zeilen der FIDE-Tafel."""

import pytest

import elo_constants as C
import elo_rating as E

S = C.SCALE


def _theta(points: float) -> float:
    return (points - C.BASE) / S


def test_fide_table_8_1_2_expected_score_for_100_and_200_points_difference():
    # Tabelle 8.1.2: Unterschied 99-106 -> .64, 198-206 -> .76 für den Höherbewerteten, .36 bzw. .24 für den Niedrigeren
    assert round(E.expected_score(_theta(1600.0), _theta(1500.0)), 2) == 0.64
    assert round(E.expected_score(_theta(1500.0), _theta(1600.0)), 2) == 0.36
    assert round(E.expected_score(_theta(1700.0), _theta(1500.0)), 2) == 0.76
    assert round(E.expected_score(_theta(1500.0), _theta(1700.0)), 2) == 0.24
    assert round(E.expected_score(_theta(1500.0), _theta(1500.0)), 2) == 0.50          # Zeile 0-3 -> .50


def test_wikipedia_example_tournament_expected_scores_and_new_rating():
    r_a = 1613.0
    games = ((1609.0, 0.0, 0.51), (1477.0, 0.5, 0.69), (1388.0, 1.0, 0.79), (1586.0, 1.0, 0.54), (1720.0, 0.0, 0.35))      # Gegner, Ergebnis, Erwartung laut Quelle
    for opp, _, expected in games:
        assert round(E.expected_score(_theta(r_a), _theta(opp)), 2) == expected
    k = 32.0 / S
    # alle fünf Partien gegen die Ausgangswertung 1613 (Turnier als eine Wertungsperiode): Summe der Änderungen k·(S − E)
    change = sum(E.elo_update(_theta(r_a), _theta(opp), score, k)[0] - _theta(r_a) for opp, score, _ in games)
    assert round(r_a + S * change) == 1601
    assert r_a + S * change == pytest.approx(1601.0, abs=0.5)
