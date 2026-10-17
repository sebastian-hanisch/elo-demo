"""Eigenschaftstests (metamorph) für Elo: Man verändert die Eingabe auf eine Weise, deren Wirkung auf die Ausgabe aus dem Verfahren folgt, und prüft diese
Beziehung auf Zufallsinstanzen mit festem Seed (ORACLE-PLAYBOOK §5.2). Keine Richtigkeitsprüfung gegen einen Rechenweg, sondern Aussagen, die für Elo gelten müssen:

- eine Partie verschiebt Punkte nur zwischen den Beteiligten (Nullsumme bei gleichem K; bei ungleichem K genau die dokumentierte Differenz);
- Vertauschen von Sieger und Verlierer bzw. Spiegeln aller Ergebnisse spiegelt die Wertungen;
- Umnummerieren der Spieler ändert nur die Plätze in der Ausgabe;
- nur Differenzen der Wertungen zählen (gemeinsame Verschiebung);
- K skaliert die Aktualisierung linear;
- Erwartung E(d) + E(−d) = 1, monoton und beschränkt; Aktualisierung ist 0, wenn das Ergebnis der Erwartung gleicht;
- die Partien einer Runde (disjunkte Paare) kommutieren;
- der Verfolgungsfehler hängt nur an zentrierten Größen.

Nicht geprüft, weil nicht belegt oder dokumentiert abweichend: Reihenfolge über Runden hinweg (Pfadabhängigkeit ist der Befund der Demo), Linearität in K über
mehrere Runden (die Erwartung hängt vom Zwischenstand ab)."""

import dataclasses
import math

import numpy as np
import pytest

import elo_constants as C
import elo_drift as D
import elo_rating as E

N_INSTANCES = 40
RNG_SEED = 20261017


def _rng(salt: int) -> np.random.Generator:
    return np.random.default_rng(RNG_SEED + salt)


def _random_matches(rng, newcomer: bool = False) -> D.Matches:
    n = int(rng.integers(4, 11))
    rounds = int(rng.integers(8, 25))
    sigma = float(rng.choice([0.0, 0.05, 0.1]))
    join = int(rng.integers(2, rounds - 1)) if newcomer else 0
    return D.simulate_matches(n, float(rng.choice([0.1, 0.3, 0.6])), rounds, sigma, join, float(rng.choice([-1.0, 0.0, 1.0, 2.0])), int(rng.integers(1, 10 ** 6)))


def _random_update_args(rng):
    a, b = rng.normal(0.0, 1.0, 2) * float(rng.choice([0.1, 1.0, 3.0]))
    return float(a), float(b), float(rng.choice([0.0, 0.5, 1.0])), float(rng.uniform(0.01, 0.5))


# --- 1. Eine Partie verschiebt Punkte nur zwischen den Beteiligten ------------------------------------------------------------------------------

def test_a_game_moves_points_only_between_the_two_players_with_equal_k():
    rng = _rng(1)
    for _ in range(N_INSTANCES):
        a, b, s, k = _random_update_args(rng)
        a2, b2 = E.elo_update(a, b, s, k)
        assert a2 + b2 == pytest.approx(a + b, abs=1e-12)
    for a, b in ((0.0, 0.0), (900.0, -900.0), (-900.0, 900.0), (5.0, 5.0)):           # Randfälle: Gleichstand, riesiger Abstand
        for s in (0.0, 1.0):
            a2, b2 = E.elo_update(a, b, s, 0.1)
            assert a2 + b2 == pytest.approx(a + b, abs=1e-12)


def test_a_game_with_unequal_k_changes_the_sum_by_exactly_the_documented_difference():
    rng = _rng(2)
    for _ in range(N_INSTANCES):
        a, b, s, k = _random_update_args(rng)
        kj = float(rng.uniform(0.01, 0.5))
        a2, b2 = E.elo_update(a, b, s, k, kj)
        e = E.expected_score(a, b)
        assert (a2 + b2) - (a + b) == pytest.approx((k - kj) * (s - e), abs=1e-12)


def test_a_run_with_fixed_k_conserves_the_sum_of_all_ratings_in_every_round():
    rng = _rng(3)
    for _ in range(30):
        m = _random_matches(rng)
        for K in (5.0, float(rng.choice([20.0, 80.0, 200.0]))):
            run = E.run_elo(m, K)
            assert np.abs(run.history.sum(axis=1)).max() < 1e-9
            assert abs(run.final.sum()) < 1e-9


# --- 2. Vertauschen von Sieger und Verlierer spiegelt die Aktualisierung --------------------------------------------------------------------------

def test_swapping_winner_and_loser_mirrors_the_update():
    rng = _rng(4)
    for _ in range(N_INSTANCES):
        a, b, s, k = _random_update_args(rng)
        kj = float(rng.uniform(0.01, 0.5))
        a2, b2 = E.elo_update(a, b, s, k, kj)
        # dieselbe Partie aus Sicht des anderen Spielers: Plätze getauscht, Ergebnis 1 − s
        b3, a3 = E.elo_update(b, a, 1.0 - s, kj, k)
        assert (a3, b3) == (pytest.approx(a2, abs=1e-12), pytest.approx(b2, abs=1e-12))
        # alle Ergebnisse und alle Wertungen gespiegelt: Wertungen gespiegelt
        a4, b4 = E.elo_update(-a, -b, 1.0 - s, k, kj)
        assert (a4, b4) == (pytest.approx(-a2, abs=1e-12), pytest.approx(-b2, abs=1e-12))


def test_mirroring_every_result_negates_the_ratings_and_keeps_the_log_loss():
    rng = _rng(5)
    for idx in range(30):
        m = _random_matches(rng, newcomer=False)
        flipped = dataclasses.replace(m, win=1 - m.win)
        for K, K0 in ((20.0, None), (30.0, 120.0)):
            r, rf = E.run_elo(m, K, K0), E.run_elo(flipped, K, K0)
            assert rf.final == pytest.approx(-r.final, abs=1e-9)
            assert rf.history == pytest.approx(-r.history, abs=1e-9)
            assert rf.log_loss == pytest.approx(r.log_loss, abs=1e-9)


def test_exchanging_the_two_players_of_every_game_and_flipping_the_result_changes_nothing():
    rng = _rng(6)
    for _ in range(30):
        m = _random_matches(rng, newcomer=bool(rng.integers(0, 2)))
        sw = dataclasses.replace(m, i=m.j.copy(), j=m.i.copy(), win=1 - m.win)
        r, rs = E.run_elo(m, 25.0, 100.0), E.run_elo(sw, 25.0, 100.0)
        assert rs.final == pytest.approx(r.final, abs=1e-9)
        assert rs.log_loss == pytest.approx(r.log_loss, abs=1e-9)


# --- 3. Umnummerieren der Spieler -----------------------------------------------------------------------------------------------------------------

def _renumbered(m: D.Matches, perm) -> D.Matches:
    """Spieler q heißt danach perm[q]; die wahren Stärken wandern mit."""
    inv = np.argsort(perm)
    return dataclasses.replace(m, truth=m.truth[:, inv], i=perm[m.i], j=perm[m.j], names=tuple(m.names[q] for q in inv))


def test_renumbering_the_players_only_permutes_the_output():
    rng = _rng(7)
    for _ in range(N_INSTANCES):
        m = _random_matches(rng)
        perm = rng.permutation(m.n)
        mp = _renumbered(m, perm)
        for K, K0 in ((20.0, None), (20.0, 100.0)):
            r, rp = E.run_elo(m, K, K0), E.run_elo(mp, K, K0)
            assert rp.final[perm] == pytest.approx(r.final, abs=1e-9)
            assert rp.history[:, perm] == pytest.approx(r.history, abs=1e-9)
            assert rp.log_loss == pytest.approx(r.log_loss, abs=1e-9)
            assert rp.tracking_error == pytest.approx(r.tracking_error, abs=1e-9) or (math.isnan(rp.tracking_error) and math.isnan(r.tracking_error))


def test_renumbering_all_players_except_the_newcomer_only_permutes_the_output():
    rng = _rng(8)
    for _ in range(N_INSTANCES):
        m = _random_matches(rng, newcomer=True)
        perm = np.concatenate([[0], 1 + rng.permutation(m.n - 1)])          # Spieler 0 ist der Neuling und bleibt
        mp = _renumbered(m, perm)
        r, rp = E.run_elo(m, 20.0, 120.0), E.run_elo(mp, 20.0, 120.0)
        assert rp.final[perm] == pytest.approx(r.final, abs=1e-9)
        assert rp.newcomer_errors == pytest.approx(r.newcomer_errors, abs=1e-9)


# --- 4. Gemeinsame Verschiebung aller Wertungen ---------------------------------------------------------------------------------------------------

def test_only_rating_differences_count_a_common_shift_leaves_the_updates_unchanged():
    rng = _rng(9)
    for _ in range(N_INSTANCES):
        a, b, s, k = _random_update_args(rng)
        c = float(rng.uniform(-5.0, 5.0))
        assert E.expected_score(a + c, b + c) == pytest.approx(E.expected_score(a, b), abs=1e-12)
        a2, b2 = E.elo_update(a, b, s, k)
        a3, b3 = E.elo_update(a + c, b + c, s, k)
        assert (a3 - (a + c), b3 - (b + c)) == (pytest.approx(a2 - a, abs=1e-12), pytest.approx(b2 - b, abs=1e-12))


def test_a_common_shift_of_the_start_ratings_shifts_a_whole_sequence_of_games_by_the_same_amount():
    rng = _rng(10)
    for _ in range(N_INSTANCES):
        n = int(rng.integers(3, 8))
        games = [(int(i), int(j), float(rng.integers(0, 2))) for i, j in (rng.choice(n, 2, replace=False) for _ in range(40))]
        start = rng.normal(0.0, 1.0, n)
        shift = float(rng.uniform(-3.0, 3.0))

        def play(R):
            R = list(R)
            for i, j, s in games:
                R[i], R[j] = E.elo_update(R[i], R[j], s, 20.0 / C.SCALE)
            return np.array(R)

        assert play(start + shift) == pytest.approx(play(start) + shift, abs=1e-9)


# --- 5. K skaliert die Aktualisierung linear ------------------------------------------------------------------------------------------------------

def test_the_update_is_linear_in_k():
    rng = _rng(11)
    for _ in range(N_INSTANCES):
        a, b, s, k = _random_update_args(rng)
        f = float(rng.choice([0.25, 2.0, 7.5]))
        a2, b2 = E.elo_update(a, b, s, k)
        a3, b3 = E.elo_update(a, b, s, f * k)
        assert (a3 - a, b3 - b) == (pytest.approx(f * (a2 - a), abs=1e-12), pytest.approx(f * (b2 - b), abs=1e-12))
        K, K0, g = float(rng.uniform(5, 60)), float(rng.uniform(60, 250)), float(rng.integers(0, 50))
        assert E.effective_k(f * K, f * K0, g) == pytest.approx(f * E.effective_k(K, K0, g), rel=1e-12)


def test_the_first_round_of_a_run_is_linear_in_k():
    """Alle starten bei 0, die Paare einer Runde sind disjunkt: nach einer Runde ist die Wertung k·(Ergebnis − ½), also linear in K (auch mit Anlauf-K)."""
    rng = _rng(12)
    for _ in range(N_INSTANCES):
        m = D.simulate_matches(int(rng.integers(2, 12)), 0.3, 1, 0.0, 0, 0.0, int(rng.integers(1, 10 ** 6)))
        K, f = float(rng.uniform(5, 100)), float(rng.choice([0.5, 3.0]))
        assert E.run_elo(m, f * K).final == pytest.approx(f * E.run_elo(m, K).final, abs=1e-12)
        assert E.run_elo(m, f * K, f * 3 * K).final == pytest.approx(f * E.run_elo(m, K, 3 * K).final, abs=1e-12)


# --- 6. Erwartung und Aktualisierung bei erfülltem Ergebnis --------------------------------------------------------------------------------------

def test_the_expected_scores_of_two_players_add_up_to_one_and_grow_with_the_difference():
    rng = _rng(13)
    for _ in range(N_INSTANCES):
        d = float(rng.normal(0.0, 3.0))
        assert E.expected_score(d, 0.0) + E.expected_score(0.0, d) == pytest.approx(1.0, abs=1e-12)
        assert E.expected_score(d, 0.0) + E.expected_score(-d, 0.0) == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= E.expected_score(d, 0.0) <= 1.0
    ds = np.linspace(-30.0, 30.0, 121)
    es = [E.expected_score(d, 0.0) for d in ds]
    assert all(x <= y for x, y in zip(es, es[1:])) and E.expected_score(0.0, 0.0) == 0.5
    for d in (700.0, 1e4):                                               # Randfälle: kein Überlauf
        assert E.expected_score(d, 0.0) + E.expected_score(-d, 0.0) == pytest.approx(1.0, abs=1e-12)
        assert not math.isnan(E.expected_score(-d, 0.0))


def test_the_update_is_zero_when_the_result_equals_the_expectation_and_has_the_sign_of_the_surprise():
    rng = _rng(14)
    for _ in range(N_INSTANCES):
        a, b, _, k = _random_update_args(rng)
        e = E.expected_score(a, b)
        a2, b2 = E.elo_update(a, b, e, k)                                 # Ergebnis = Erwartung
        assert (a2, b2) == (pytest.approx(a, abs=1e-12), pytest.approx(b, abs=1e-12))
        win, loss = E.elo_update(a, b, 1.0, k), E.elo_update(a, b, 0.0, k)
        assert win[0] >= loss[0] and win[1] <= loss[1]                     # ein Sieg senkt die Wertung des Siegers nie unter die einer Niederlage
        assert win[0] - a == pytest.approx(k * (1.0 - e), abs=1e-12) and loss[0] - a == pytest.approx(-k * e, abs=1e-12)
        assert abs(win[0] - a) <= k and abs(loss[0] - a) <= k


# --- 7. Die Partien einer Runde kommutieren; der Verfolgungsfehler hängt nur an zentrierten Größen -----------------------------------------------

def test_the_order_of_the_games_inside_a_round_does_not_matter():
    rng = _rng(15)
    for _ in range(N_INSTANCES):
        m = _random_matches(rng, newcomer=bool(rng.integers(0, 2)))
        perm = np.arange(len(m.i))
        for t in range(m.rounds):                                          # jede Runde für sich umordnen (die Runden bleiben sortiert)
            idx = np.flatnonzero(m.rnd == t)
            perm[idx] = rng.permutation(idx)
        shuffled = dataclasses.replace(m, i=m.i[perm], j=m.j[perm], win=m.win[perm])
        for K, K0 in ((20.0, None), (20.0, 100.0)):
            r, rs = E.run_elo(m, K, K0), E.run_elo(shuffled, K, K0)
            assert rs.final == pytest.approx(r.final, abs=1e-12)
            assert rs.log_loss == pytest.approx(r.log_loss, abs=1e-12)


def test_the_tracking_error_ignores_a_common_shift_of_the_true_strengths_in_every_round():
    rng = _rng(16)
    for _ in range(N_INSTANCES):
        m = _random_matches(rng)
        shift = rng.uniform(-2.0, 2.0, size=(m.rounds, 1))                  # je Runde ein anderer gemeinsamer Betrag
        shifted = dataclasses.replace(m, truth=m.truth + shift)
        r, rs = E.run_elo(m, 20.0, burn=3), E.run_elo(shifted, 20.0, burn=3)
        assert rs.tracking_error == pytest.approx(r.tracking_error, abs=1e-9)
        assert rs.final == pytest.approx(r.final, abs=1e-12)


def test_the_same_inputs_give_the_same_matches_and_the_same_ratings():
    rng = _rng(17)
    for _ in range(10):
        args = (int(rng.integers(4, 9)), 0.3, 20, 0.05, 0, 0.0, int(rng.integers(1, 10 ** 6)))
        m1, m2 = D.simulate_matches(*args), D.simulate_matches(*args)
        assert np.array_equal(m1.win, m2.win) and np.array_equal(E.run_elo(m1, 20.0).final, E.run_elo(m2, 20.0).final)
