"""Analyse, Szenario und Verdikte; Randfälle der Regler."""

import itertools
import math

import numpy as np
import pytest

import elo_constants as C
import elo_drift as D
import elo_evaluation as ev
from elo_scenario import Settings, make_matches

CASES = [Settings(n=n, rounds=rounds, sigma=sigma, k=k, provisional=prov, newcomer=new, join=7, gap=1.0, seed=3)
         for n, rounds, sigma, k, prov, new in itertools.product((4, 7, 9), (12, 30), (0.0, 0.1), (5, 80), (False, True), (False, True))]


def test_analysis_invariants_over_many_settings():
    for s in CASES:
        a = ev.analyse(s)
        n = s.n
        if not s.provisional:
            assert a.run.final.sum() == pytest.approx(0.0, abs=1e-9)          # Nullsumme nur bei gleichem K für alle
        assert a.bt.sum() == pytest.approx(0.0, abs=1e-9) and a.truth_end.sum() == pytest.approx(0.0, abs=1e-9)
        assert a.games.sum() == 2 * len(a.matches.i)
        assert a.run.history.shape == (s.rounds, n)
        assert set(a.rmse) == {"Elo", "Bradley-Terry"} and all(v >= 0 for v in a.rmse.values())
        assert -1.0 <= a.tau["Elo"] <= 1.0 and -1.0 <= a.tau["Bradley-Terry"] <= 1.0
        assert a.run.log_loss > 0 and a.info > 0
        assert (a.k_star == 0.0) == (s.sigma == 0.0)
        assert (s.join_round > 0) == (a.matches.join_round > 0) == (len(a.run.newcomer_errors) > 0)


def test_analysis_is_cached_and_deterministic():
    s = Settings(n=6, rounds=20, seed=11)
    assert ev.analyse(s) is ev.analyse(s)
    assert np.array_equal(make_matches(s).win, make_matches(Settings(n=6, rounds=20, seed=11)).win)


def test_helpers_centering_rmse_and_tau():
    assert ev.centered([1.0, 2.0, 3.0]).tolist() == [-1.0, 0.0, 1.0]
    assert ev.rmse_centered([0.0, 2.0], [10.0, 12.0]) == pytest.approx(0.0)
    assert ev.rmse_centered([1.0, -1.0], [-1.0, 1.0]) == pytest.approx(2.0)
    assert ev.kendall_tau([3.0, 2.0, 1.0], [30.0, 20.0, 10.0]) == 1.0 and ev.kendall_tau([1.0, 2.0], [2.0, 1.0]) == -1.0
    assert math.isnan(ev.kendall_tau([1.0, 1.0], [1.0, 2.0]))


def test_verdict_cases():
    level, text = ev.verdict_analysis(ev.analyse(Settings(sigma=0.0, seed=2)))
    assert level == "info" and "Ohne Drift" in text and "Kalman" not in text
    wins = [ev.analyse(Settings(sigma=0.12, k=40, seed=s)) for s in range(100000, 100030)]
    win = next(a for a in wins if a.rmse["Elo"] < a.rmse["Bradley-Terry"])
    level, text = ev.verdict_analysis(win)
    assert level == "success" and "Kalman-Verstärkung" in text and "mittelt die Vergangenheit" in text


def test_sweeps_have_consistent_shapes():
    seeds = C.SWEEP_SEEDS[:8]
    rows = ev.sweep_k(sigmas=(0.0, 0.06), ks=(5, 20, 80), seeds=seeds)
    assert len(rows) == 6 and sum(r["best"] for r in rows) == 2
    kal = ev.sweep_kalman(sigmas=(0.06,), ks=(10, 20, 40), seeds=seeds)
    assert kal[0]["k_star"] == pytest.approx(23.0, abs=0.1) and kal[0]["best_error"] in (10, 20, 40)
    bt = ev.sweep_vs_bt(sigmas=(0.0,), ks=(10, 20), seeds=seeds)
    assert bt[0]["k"] in (10, 20) and bt[0]["bt_rmse"] > 0
    od = ev.sweep_order(ks=(10, 40), repeats=4, seeds=seeds[:3])
    assert od[1]["sd"] > od[0]["sd"] > 0 and 0.0 <= od[0]["flip"] <= 1.0
    nc = ev.sweep_newcomer(seeds=seeds)
    assert [r["label"] for r in nc][0] == "K = 10" and nc[0]["g20"] > nc[2]["g20"]
    gp = ev.sweep_newcomer_gap(gaps=(0.0, 2.0), seeds=seeds)
    assert gp[1]["fixed"] > gp[0]["fixed"]


def test_decimal_comma_helpers():
    assert ev.de(0.5) == "0,50" and ev.de(3.14159, 3) == "3,142" and ev.pct(0.405) in ("40", "41")


def test_simulation_with_odd_player_count_still_runs_with_a_newcomer():
    a = ev.analyse(Settings(n=5, rounds=20, newcomer=True, join=8, seed=1))
    assert a.games.sum() == 2 * len(a.matches.i) and D.Matches is type(a.matches)
