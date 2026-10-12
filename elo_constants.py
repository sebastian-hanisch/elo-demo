"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit (400 Punkte = Faktor 10 der Gewinnchancen), 173,72
BASE = 1500.0                               # Mitte der Elo-Skala in der Anzeige

N_MIN, N_MAX, DEFAULT_N = 4, 16, 8
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
ROUNDS_MIN, ROUNDS_MAX, DEFAULT_ROUNDS = 10, 200, 150
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 0.0, 0.3, 0.01, 0.06     # Schritt der Irrfahrt je Runde (logistische Einheiten)
K_MIN, K_MAX, DEFAULT_K = 1, 200, 20                                         # Elo-Punkte
K0_MIN, K0_MAX, DEFAULT_K0 = 20, 240, 120                                    # Anlauf-K für Neulinge (Punkte)
JOIN_MIN, DEFAULT_JOIN = 2, 60                                               # Runde, in der der Neuling kommt (Obergrenze: Runden − 2)
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = -3.0, 3.0, 0.5, 1.0                # Abstand des Neulings vom Feld (logistische Einheiten)
G0 = 10.0                                                                     # Anlauf: nach g0 Partien ist der Aufschlag auf K halbiert
SEED_MAX, DEFAULT_SEED = 999999, 44          # typischer Seed: Endfehler Elo 0,35 gegen Bradley-Terry 0,51 (Mittel der Experimente 0,35 und 0,46)
BURN = 30                                                                     # Runden, die beim Verfolgungsfehler nicht zählen (Kaltstart)

METHODS = ("Elo", "Bradley-Terry")
TOL = 1e-9

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100100))
EXP_N = 8
EXP_DELTA = 0.3
EXP_ROUNDS = 150
K_SWEEP = (2, 5, 10, 20, 40, 80, 160)
K_FINE = (7, 10, 14, 20, 28, 40, 56, 80, 112, 160)
SIGMA_SWEEP = (0.0, 0.03, 0.06, 0.12)
SIGMA_KALMAN = (0.03, 0.06, 0.12, 0.24)
SIGMA_BT = (0.0, 0.015, 0.03, 0.045, 0.06, 0.12)
ORDER_KS = (5, 10, 20, 40, 80)
ORDER_REPEATS = 16
ORDER_SEEDS = SWEEP_SEEDS[:40]
NEW_JOIN = 60
NEW_ROUNDS = 100
NEW_KS = (10, 20, 40)
NEW_GAPS = (0.0, 0.5, 1.0, 2.0)
NEW_SEEDS = SWEEP_SEEDS

COLOR_ELO = "#2F6B65"
COLOR_BT = "#8e6bb0"
COLOR_TRUTH = "#e67e22"
COLOR_KALMAN = "#c0392b"
PALETTE = ("#2F6B65", "#8e6bb0", "#d98c1f", "#c0392b", "#3d7ab5", "#7a8f3c", "#b0577e", "#5a5a5a", "#1f9d8a", "#a0522d", "#6c5fc7", "#b8860b",
           "#2e8b57", "#cd5c5c", "#4682b4", "#808000")

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "rounds": DEFAULT_ROUNDS, "sigma": DEFAULT_SIGMA, "k": DEFAULT_K, "provisional": False, "k0": DEFAULT_K0,
         "newcomer": False, "join": DEFAULT_JOIN, "gap": DEFAULT_GAP, "seed": DEFAULT_SEED}
PRESETS = {
    "Wandernde Stärken": dict(_BASE),
    "Konstante Stärken": {**_BASE, "sigma": 0.0},
    "K zu groß": {**_BASE, "k": 160},
    "Neuling im Feld": {**_BASE, "sigma": 0.0, "provisional": True, "newcomer": True, "join": 60, "gap": 1.0, "rounds": 100},
}
PRESET_HELP = {
    "Wandernde Stärken": "8 Spieler, 150 Runden, die Stärken wandern mit Schritt 0,06 je Runde, K = 20: Elo folgt der Drift, die Wertung hängt aber am Pfad.",
    "Konstante Stärken": "Dieselbe Runde ohne Drift: Elo schwankt um die wahre Stärke, ein Schätzer über alle Partien (Bradley-Terry) wäre genauer.",
    "K zu groß": "K = 160 Punkte: die Wertung folgt dem Zufall der letzten Partien statt der Stärke; der Fehler liegt über dem von K = 2.",
    "Neuling im Feld": "Ein Neuling mit Abstand 1 vom Feld kommt in Runde 60; mit Anlauf-K (K0 = 120) findet seine Wertung schneller zu ihm.",
}
