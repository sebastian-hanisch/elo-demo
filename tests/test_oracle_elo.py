"""Orakel-Tests elo-demo: Elo-Verlauf, Kalman-Verstärkung und Schätzer gegen ANDERE Rechenwege (DEMO-PLAYBOOK §12).

1. Elo-Verlauf gegen eine unabhängige Schleife in ELO-PUNKTEN: E = 1/(1 + 10^(−ΔR/400)), R += K (S − E) (die Formulierung der Elo-Literatur; die Demo rechnet in
   logistischen Einheiten).
2. Nullsumme und Anlauf-K in beiden Rechenwegen.
3. Kalman-Verstärkung gegen die stationäre Riccati-Iteration und gegen scipy.linalg.solve_discrete_are.
4. Bradley-Terry über alle Partien gegen scipy.optimize auf der Partienliste; Unabhängigkeit von der Reihenfolge (Elo: abhängig).
5. Pfadabhängigkeit: dieselben Partien, andere Reihenfolge, andere Elo-Endwerte bei gleicher Summe.
"""

import math
import random

import numpy as np
import pytest

import elo_batch as B
import elo_constants as C
import elo_drift as D
import elo_rating as E

optimize = pytest.importorskip("scipy.optimize")
linalg = pytest.importorskip("scipy.linalg")


def elo_in_points(m, K, K0=None, g0=10.0):
    """Unabhängige Fassung in Elo-Punkten (Anfangswert 1500), Partie für Partie; Rückgabe: Endwertungen, Log-Verlust."""
    n = m.n
    R = [1500.0] * n
    played = [0] * n
    ll = 0.0
    for t in range(m.rounds):
        if m.join_round and t == m.join_round:
            R[0] = sum(R[1:]) / (n - 1)
        for g in range(len(m.i)):
            if m.rnd[g] != t:
                continue
            i, j, s = int(m.i[g]), int(m.j[g]), float(m.win[g])
            e = 1.0 / (1.0 + 10.0 ** (-(R[i] - R[j]) / 400.0))
            ll += -math.log(e if s == 1.0 else 1.0 - e)
            ki = K if K0 is None else K + (K0 - K) / (1.0 + played[i] / g0)
            kj = K if K0 is None else K + (K0 - K) / (1.0 + played[j] / g0)
            played[i] += 1
            played[j] += 1
            R[i] += ki * (s - e)
            R[j] -= kj * (s - e)
    return np.array(R), ll / len(m.i)


# --- 1. und 2. Elo-Verlauf --------------------------------------------------------------------------------------------------------------

def test_elo_run_equals_the_textbook_formulation_in_points():
    rnd = random.Random(2026)
    checked = 0
    for seed in range(12):
        n = rnd.choice([4, 6, 7, 8])
        m = D.simulate_matches(n, 0.3, rnd.randint(10, 40), rnd.choice([0.0, 0.05, 0.1]), 0, 0.0, seed)
        for K, K0 in ((10.0, None), (20.0, None), (40.0, 120.0)):
            run = E.run_elo(m, K, K0)
            pts, ll = elo_in_points(m, K, K0)
            assert C.BASE + C.SCALE * run.final == pytest.approx(pts, abs=1e-6)
            assert run.log_loss == pytest.approx(ll, abs=1e-9)
        checked += 1
    assert checked == 12


def test_newcomer_start_and_provisional_k_match_the_textbook_formulation():
    for seed in range(6):
        m = D.simulate_matches(8, 0.3, 40, 0.05, 15, 1.0, seed)
        run = E.run_elo(m, 20.0, 100.0)
        pts, ll = elo_in_points(m, 20.0, 100.0)
        assert C.BASE + C.SCALE * run.final == pytest.approx(pts, abs=1e-6)
        assert run.log_loss == pytest.approx(ll, abs=1e-9)
        assert abs(pts.sum() - 8 * 1500.0) > 1e-3                                # Anlauf-K: die Nullsumme ist gebrochen, in beiden Rechenwegen gleich
    m = D.simulate_matches(8, 0.3, 40, 0.05, 15, 1.0, 1)
    pts, _ = elo_in_points(m, 20.0)
    assert pts.sum() == pytest.approx(8 * 1500.0, abs=1e-6)                      # festes K: Nullsumme


# --- 3. Kalman-Verstärkung ----------------------------------------------------------------------------------------------------------------

def test_kalman_gain_equals_the_riccati_iteration_and_the_discrete_are_solution():
    for sigma, info in ((0.03, 0.2), (0.06, 0.2), (0.12, 0.15), (0.24, 0.25)):
        q = sigma ** 2
        P = 1.0
        for _ in range(5000):                                                   # Vorhersage- und Messschritt bis zur Konvergenz
            Pp = P + q
            P = Pp - Pp * info * Pp / (info * Pp * Pp + info) * 1.0 if False else Pp / (1.0 + Pp * info)
        assert E.kalman_gain(sigma, info) == pytest.approx(P, rel=1e-9)
        # scipy: Vorhersage-Kovarianz Pp der dualen Riccati-Gleichung (A = 1, H = 1, Q = q, R = 1/I); posteriori P = Pp − Pp² /(Pp + R)
        Pp = float(linalg.solve_discrete_are(np.array([[1.0]]), np.array([[1.0]]), np.array([[q]]), np.array([[1.0 / info]]))[0, 0])
        assert E.kalman_gain(sigma, info) == pytest.approx(Pp - Pp * Pp / (Pp + 1.0 / info), rel=1e-6)


# --- 4. Bradley-Terry --------------------------------------------------------------------------------------------------------------------

def test_bradley_terry_matches_scipy_on_the_game_list_and_ignores_the_order():
    for seed in range(8):
        m = D.simulate_matches(6, 0.3, 25, 0.0, 0, 0.0, seed)
        lam = 0.7

        def neg(th):
            d = np.where(m.win == 1, th[m.i] - th[m.j], th[m.j] - th[m.i])
            return float(np.logaddexp(0.0, -d).sum() + 0.5 * lam * float(th @ th))

        res = optimize.minimize(neg, np.zeros(6), method="BFGS", options={"gtol": 1e-10})
        ours = B.fit_bradley_terry(B.win_matrix(m), lam)
        assert ours == pytest.approx(res.x - res.x.mean(), abs=1e-5)
        assert B.fit_bradley_terry(B.win_matrix(D.reorder_games(m, 9)), lam) == pytest.approx(ours, abs=1e-12)


# --- 5. Pfadabhängigkeit ------------------------------------------------------------------------------------------------------------------

def test_elo_depends_on_the_order_but_keeps_the_sum():
    m = D.simulate_matches(8, 0.3, 60, 0.0, 0, 0.0, 3)
    base = E.run_elo(m, 20.0).final
    other = E.run_elo(D.reorder_games(m, 4), 20.0).final
    assert base.sum() == pytest.approx(0.0, abs=1e-9) and other.sum() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(base - other).max() > 0.02                                     # θ-Einheiten: spürbar anders
    small = E.run_elo(D.reorder_games(m, 4), 2.0).final
    small0 = E.run_elo(m, 2.0).final
    assert np.abs(small - small0).max() < np.abs(base - other).max()             # je kleiner K, desto weniger Pfadabhängigkeit
