"""Spiele mit wandernden Stärken: Paarung, Neuling, Irrfahrt, Normalverteilung und Umordnen gegen Handwerte und Statistik."""

import numpy as np
import pytest

import elo_drift as D
import turnier_generator as T


def test_every_active_player_plays_exactly_once_per_round_for_even_n():
    m = D.simulate_matches(8, 0.3, 12, 0.05, 0, 0.0, 3)
    assert len(m.i) == 12 * 4 and m.truth.shape == (12, 8)
    for t in range(12):
        players = np.r_[m.i[m.rnd == t], m.j[m.rnd == t]]
        assert sorted(players.tolist()) == list(range(8))


def test_odd_n_leaves_exactly_one_player_out_per_round():
    m = D.simulate_matches(7, 0.3, 10, 0.0, 0, 0.0, 4)
    for t in range(10):
        players = np.r_[m.i[m.rnd == t], m.j[m.rnd == t]]
        assert len(players) == 6 and len(set(players.tolist())) == 6


def test_newcomer_is_absent_before_and_present_after_the_join_round():
    m = D.simulate_matches(8, 0.3, 30, 0.0, 12, 1.5, 5)
    before = m.rnd < 12
    assert 0 not in np.r_[m.i[before], m.j[before]]
    after = m.rnd >= 12
    assert 0 in np.r_[m.i[after], m.j[after]]
    # wahre Stärke des Neulings: Mittel der anderen plus Abstand
    assert m.truth[0][0] == pytest.approx(m.truth[0][1:].mean() + 1.5)


def test_static_strengths_do_not_move_and_the_random_walk_has_the_right_variance():
    m = D.simulate_matches(8, 0.3, 20, 0.0, 0, 0.0, 6)
    assert np.allclose(m.truth, m.truth[0])
    sigma, rounds = 0.1, 41
    steps = []
    for seed in range(400):
        mm = D.simulate_matches(8, 0.3, rounds, sigma, 0, 0.0, seed)
        steps.append(mm.truth[-1] - mm.truth[0])
    steps = np.array(steps)
    assert steps.mean() == pytest.approx(0.0, abs=0.02)
    assert steps.var() == pytest.approx(sigma ** 2 * (rounds - 1), rel=0.08)


def test_box_muller_normals_have_mean_zero_and_unit_variance():
    z = D.normals(T.SplitMix64(9), 200000)
    assert z.mean() == pytest.approx(0.0, abs=0.01) and z.std() == pytest.approx(1.0, abs=0.01)
    assert np.isfinite(z).all()


def test_outcomes_follow_the_logistic_probability():
    s = T.make_strengths(8, 0.3)
    s = s - s.mean()
    wins, games = {}, {}
    for seed in range(150):
        m = D.simulate_matches(8, 0.3, 40, 0.0, 0, 0.0, seed)
        for a, b, w in zip(m.i.tolist(), m.j.tolist(), m.win.tolist()):
            key = (min(a, b), max(a, b))
            wins[key] = wins.get(key, 0) + (w if a < b else 1 - w)
            games[key] = games.get(key, 0) + 1
    checked = 0
    for (a, b), g in games.items():
        if g > 800:
            assert wins[(a, b)] / g == pytest.approx(float(D.sigmoid(s[a] - s[b])), abs=0.05)
            checked += 1
    assert checked > 5


def test_deterministic_and_seed_sensitive_and_invalid_join_raises():
    a = D.simulate_matches(8, 0.3, 10, 0.05, 0, 0.0, 7)
    b = D.simulate_matches(8, 0.3, 10, 0.05, 0, 0.0, 7)
    c = D.simulate_matches(8, 0.3, 10, 0.05, 0, 0.0, 8)
    assert np.array_equal(a.i, b.i) and np.array_equal(a.win, b.win) and np.array_equal(a.truth, b.truth)
    assert not np.array_equal(a.win, c.win) or not np.array_equal(a.i, c.i)
    with pytest.raises(ValueError, match="Neuling"):
        D.simulate_matches(8, 0.3, 10, 0.0, 10, 0.0, 1)


def test_reordering_keeps_the_games_and_the_round_slots():
    m = D.simulate_matches(8, 0.3, 20, 0.0, 0, 0.0, 2)
    r = D.reorder_games(m, 5)
    assert np.array_equal(r.rnd, m.rnd) and np.array_equal(r.truth, m.truth)
    key = lambda x: sorted(zip(x.i.tolist(), x.j.tolist(), x.win.tolist()))      # noqa: E731
    assert key(r) == key(m)
    assert not np.array_equal(r.i, m.i)
    assert np.array_equal(D.reorder_games(m, 5).i, r.i) and not np.array_equal(D.reorder_games(m, 6).i, r.i)
