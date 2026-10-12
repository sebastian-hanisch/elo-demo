"""Elo-Aktualisierung, Verlauf, Anlauf-K und Kalman-Verstärkung gegen von Hand gerechnete Werte."""

import math

import numpy as np
import pytest

import elo_constants as C
import elo_drift as D
import elo_rating as E


def test_expected_score_hand_values():
    assert E.expected_score(0.0, 0.0) == 0.5
    assert E.expected_score(400.0 / C.SCALE, 0.0) == pytest.approx(10.0 / 11.0)          # 400 Punkte: Faktor 10
    assert E.expected_score(0.0, 400.0 / C.SCALE) == pytest.approx(1.0 / 11.0)
    assert E.expected_score(800.0, 0.0) == pytest.approx(1.0) and E.expected_score(0.0, 800.0) == pytest.approx(0.0, abs=1e-300)


def test_update_hand_values_and_zero_sum():
    k = 20.0 / C.SCALE
    a, b = E.elo_update(0.0, 0.0, 1.0, k)
    assert C.SCALE * a == pytest.approx(10.0) and C.SCALE * b == pytest.approx(-10.0) and a + b == pytest.approx(0.0)
    a2, b2 = E.elo_update(a, b, 0.0, k)              # zweite Partie, A verliert: E_A = σ(2·0,0575) = 0,5288
    assert E.expected_score(a, b) == pytest.approx(0.5288, abs=1e-4)
    assert C.SCALE * a2 == pytest.approx(10.0 - 20.0 * 0.52875, abs=1e-3) and a2 + b2 == pytest.approx(0.0)


def test_update_with_different_ks_breaks_the_zero_sum_as_documented():
    a, b = E.elo_update(0.0, 0.0, 1.0, 0.2, 0.1)
    assert a == pytest.approx(0.1) and b == pytest.approx(-0.05) and a + b == pytest.approx(0.05)


def test_effective_k_hand_values():
    assert E.effective_k(20.0, None, 0) == 20.0 and E.effective_k(20.0, None, 500) == 20.0
    assert E.effective_k(20.0, 120.0, 0) == pytest.approx(120.0)
    assert E.effective_k(20.0, 120.0, 10) == pytest.approx(70.0)              # nach g0 = 10 Partien ist der Aufschlag halbiert
    assert E.effective_k(20.0, 120.0, 1e6) == pytest.approx(20.0, abs=1e-3)


def _two_player_matches():
    z = np.zeros((2, 2))
    return D.Matches(2, z, np.array([0, 0]), np.array([0, 0]), np.array([1, 1]), np.array([1, 0]), 0, ("A", "B"))


def test_run_elo_two_games_by_hand():
    run = E.run_elo(_two_player_matches(), 20.0, burn=0)
    k = 20.0 / C.SCALE
    after1 = k * 0.5
    e2 = 1.0 / (1.0 + math.exp(-2 * after1))
    after2 = after1 + k * (0.0 - e2)
    assert run.final[0] == pytest.approx(after2) and run.final[1] == pytest.approx(-after2)
    assert run.log_loss == pytest.approx((math.log(2) + (-math.log(1 - e2))) / 2)
    assert run.history[0][0] == pytest.approx(0.0 + after2)               # eine Runde (Runden-Index 0), beide Partien in dieser Runde
    assert run.newcomer_errors == ()


def test_run_elo_conserves_the_sum_and_hides_the_newcomer_before_joining():
    m = D.simulate_matches(8, 0.3, 40, 0.05, 15, 1.0, 3)
    run = E.run_elo(m, 25.0)
    assert run.final.sum() == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(run.history[:15, 0]).all() and not np.isnan(run.history[15:, 0]).any()
    assert len(run.newcomer_errors) == 25 and all(e >= 0 for e in run.newcomer_errors)


def test_a_provisional_k_breaks_the_zero_sum_and_the_sum_follows_the_k_difference():
    m = D.simulate_matches(8, 0.3, 40, 0.0, 15, 1.0, 3)
    fixed = E.run_elo(m, 25.0)
    prov = E.run_elo(m, 25.0, 120.0)
    assert fixed.final.sum() == pytest.approx(0.0, abs=1e-9)
    assert abs(prov.final.sum()) > 1e-3                                       # Punkte entstehen oder verschwinden


def test_run_elo_matches_repeated_calls_of_the_update_function():
    m = D.simulate_matches(6, 0.3, 15, 0.05, 0, 0.0, 8)
    R = [0.0] * 6
    k = 20.0 / C.SCALE
    for g in range(len(m.i)):
        R[int(m.i[g])], R[int(m.j[g])] = E.elo_update(R[int(m.i[g])], R[int(m.j[g])], float(m.win[g]), k)
    assert E.run_elo(m, 20.0).final == pytest.approx(R, abs=1e-12)


def test_newcomer_error_by_hand():
    # 3 Spieler; Runde 0: 1 schlägt 2 (Spieler 0 fehlt noch). Runde 1 (Neuling kommt): Partie (0, 1), 0 gewinnt; Partie (2, 0), 0 gewinnt (win = 0 für i = 2).
    # So steht der Neuling einmal auf der i- und einmal auf der j-Seite; die wahren Stärken der anderen sind ungleich Null, damit Vorzeichenfehler auffallen.
    truth = np.array([[1.0, 0.2, -0.1], [1.0, 0.2, -0.1]])
    m = D.Matches(3, truth, np.array([0, 1, 1]), np.array([1, 0, 2]), np.array([2, 1, 0]), np.array([1, 1, 0]), 1, ("A", "B", "C"))
    run = E.run_elo(m, 20.0, burn=0)
    k = 20.0 / C.SCALE
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))                                  # noqa: E731
    R = [0.0, k * 0.5, -k * 0.5]                                                 # nach Runde 0
    R[0] = (R[1] + R[2]) / 2                                                     # Startwertung des Neulings: Feldmittel
    p = sig(R[0] - R[1])
    R[0] += k * (1 - p)
    R[1] -= k * (1 - p)
    err1 = abs((R[0] - (R[1] + R[2]) / 2) - (1.0 - (0.2 - 0.1) / 2))
    p = sig(R[2] - R[0])                                                         # (2, 0), 0 gewinnt: i = 2 verliert
    R[2] += k * (0 - p)
    R[0] -= k * (0 - p)
    err2 = abs((R[0] - (R[1] + R[2]) / 2) - (1.0 - (0.2 - 0.1) / 2))
    assert len(run.newcomer_errors) == 2
    assert run.newcomer_errors[0] == pytest.approx(err1, abs=1e-12) and run.newcomer_errors[1] == pytest.approx(err2, abs=1e-12)
    assert run.final == pytest.approx(R, abs=1e-12)


def test_kalman_gain_hand_values_and_fixed_point():
    info = 0.2
    q = 0.06 ** 2
    p = E.kalman_gain(0.06, info)
    assert p == pytest.approx(0.13238, abs=1e-4) and E.kalman_k_points(0.06, info) == pytest.approx(23.0, abs=0.05)
    assert p == pytest.approx((p + q) / (1.0 + (p + q) * info), abs=1e-12)        # stationäre Riccati-Gleichung
    assert E.kalman_gain(0.0, info) == 0.0
    small = E.kalman_gain(0.001, info)
    assert small == pytest.approx(math.sqrt(1e-6 / info), rel=0.01)               # k* ≈ √(q/I) für kleines q
    assert E.kalman_gain(0.12, info) / E.kalman_gain(0.06, info) == pytest.approx(2.0, rel=0.1)


def test_mean_information_and_points_conversion():
    assert E.mean_information(np.zeros(5)) == pytest.approx(0.25)
    assert E.mean_information(np.array([0.0, 100.0])) == pytest.approx(0.0, abs=1e-12)
    assert E.to_points(0.0) == 1500.0 and E.to_points(1.0) == pytest.approx(1500.0 + C.SCALE)
    assert C.SCALE == pytest.approx(173.717792761, rel=1e-9)
