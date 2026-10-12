"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import elo_evaluation as ev


@pytest.fixture(scope="module")
def k_rows():
    return {(r["sigma"], r["k"]): r for r in ev.sweep_k()}


@pytest.fixture(scope="module")
def kalman():
    return {r["sigma"]: r for r in ev.sweep_kalman()}


@pytest.fixture(scope="module")
def vs_bt():
    return {r["sigma"]: r for r in ev.sweep_vs_bt()}


@pytest.fixture(scope="module")
def order():
    return {r["k"]: r for r in ev.sweep_order()}


@pytest.fixture(scope="module")
def newcomer():
    return {r["label"]: r for r in ev.sweep_newcomer()}


@pytest.fixture(scope="module")
def gaps():
    return {r["gap"]: r for r in ev.sweep_newcomer_gap()}


# --- Experiment 1 ------------------------------------------------------------------------------------------------------------------------

def test_k_is_a_trade_off_with_a_u_shape(k_rows):
    expected = {2: 0.540, 5: 0.396, 10: 0.276, 20: 0.2405, 40: 0.319, 80: 0.482, 160: 0.763}
    for k, err in expected.items():
        assert k_rows[(0.0, k)]["error"] == pytest.approx(err, abs=0.006), k
    assert k_rows[(0.0, 20)]["best"] and not any(k_rows[(0.0, k)]["best"] for k in (2, 5, 10, 40, 80, 160))
    assert k_rows[(0.0, 2)]["error"] > 2 * k_rows[(0.0, 20)]["error"] and k_rows[(0.0, 160)]["error"] > 3 * k_rows[(0.0, 20)]["error"]


def test_the_best_k_moves_with_the_drift(k_rows):
    best = {sg: next(k for (s, k), r in k_rows.items() if s == sg and r["best"]) for sg in (0.0, 0.03, 0.06, 0.12)}
    assert best == {0.0: 20, 0.03: 20, 0.06: 20, 0.12: 40}
    assert k_rows[(0.06, 20)]["error"] == pytest.approx(0.353, abs=0.006) and k_rows[(0.06, 40)]["error"] == pytest.approx(0.357, abs=0.006)
    assert k_rows[(0.12, 40)]["error"] == pytest.approx(0.505, abs=0.006) and k_rows[(0.12, 20)]["error"] == pytest.approx(0.607, abs=0.006)
    assert k_rows[(0.06, 20)]["log_loss"] == pytest.approx(0.5946, abs=0.004)


# --- Experiment 2 ------------------------------------------------------------------------------------------------------------------------

def test_the_best_k_scales_like_the_kalman_gain(kalman):
    expected = {0.03: (11.6, 20, 20), 0.06: (23.0, 28, 28), 0.12: (45.3, 56, 40), 0.24: (88.3, 112, 80)}
    for sg, (kstar, be, bl) in expected.items():
        assert kalman[sg]["k_star"] == pytest.approx(kstar, abs=0.1), sg
        assert kalman[sg]["best_error"] == be and kalman[sg]["best_log_loss"] == bl, sg
    assert kalman[0.12]["k_star"] / kalman[0.06]["k_star"] == pytest.approx(2.0, rel=0.05)
    assert kalman[0.24]["best_error"] / kalman[0.12]["best_error"] == pytest.approx(2.0, rel=0.1)


# --- Experiment 3 ------------------------------------------------------------------------------------------------------------------------

def test_elo_loses_without_drift_and_wins_with_it(vs_bt):
    expected = {0.0: (0.222, 0.148, 0.880, 0.950), 0.015: (0.239, 0.192, 0.846, 0.883), 0.045: (0.292, 0.352, 0.779, 0.746),
                0.06: (0.351, 0.455, 0.772, 0.701), 0.12: (0.552, 0.924, 0.821, 0.631)}
    for sg, (er, br, et, bt) in expected.items():
        r = vs_bt[sg]
        assert r["elo_rmse"] == pytest.approx(er, abs=0.007) and r["bt_rmse"] == pytest.approx(br, abs=0.007), sg
        assert r["elo_tau"] == pytest.approx(et, abs=0.01) and r["bt_tau"] == pytest.approx(bt, abs=0.01), sg
    assert vs_bt[0.0]["bt_rmse"] < vs_bt[0.0]["elo_rmse"] and vs_bt[0.12]["elo_rmse"] < vs_bt[0.12]["bt_rmse"]
    assert abs(vs_bt[0.03]["elo_rmse"] - vs_bt[0.03]["bt_rmse"]) < 0.015          # Gleichstand bei etwa 0,03 je Runde
    assert vs_bt[0.12]["k"] == 40 and vs_bt[0.06]["k"] == 20


# --- Experiment 4 ------------------------------------------------------------------------------------------------------------------------

def test_the_path_dependence_is_large(order):
    expected = {5: (0.030, 0.15), 10: (0.076, 0.325), 20: (0.157, 0.75), 40: (0.275, 0.975), 80: (0.443, 1.0)}
    for k, (sd, flip) in expected.items():
        assert order[k]["sd"] == pytest.approx(sd, abs=0.006), k
        assert order[k]["flip"] == pytest.approx(flip, abs=0.03), k
    assert order[20]["sd_points"] == pytest.approx(27.3, abs=1.0)


# --- Experiment 5 ------------------------------------------------------------------------------------------------------------------------

def test_a_provisional_k_helps_a_newcomer_who_differs_from_the_field(newcomer, gaps):
    assert newcomer["K = 10"]["g20"] == pytest.approx(0.733, abs=0.01) and newcomer["K = 20"]["g20"] == pytest.approx(0.531, abs=0.01)
    assert newcomer["K = 40"]["g20"] == pytest.approx(0.340, abs=0.01) and newcomer["K = 20, Anlauf 60"]["g20"] == pytest.approx(0.336, abs=0.01)
    assert newcomer["K = 20, Anlauf 60"]["mean"] == pytest.approx(0.412, abs=0.01) and newcomer["K = 20"]["mean"] == pytest.approx(0.569, abs=0.01)
    assert newcomer["K = 20, Anlauf 60"]["g20"] < newcomer["K = 20"]["g20"] - 0.15
    assert gaps[1.0]["fixed"] == pytest.approx(0.531, abs=0.01) and gaps[1.0]["provisional"] == pytest.approx(0.336, abs=0.01)
    assert gaps[2.0]["fixed"] == pytest.approx(1.250, abs=0.02) and gaps[2.0]["provisional"] == pytest.approx(0.787, abs=0.02)


def test_the_provisional_k_hurts_a_newcomer_who_equals_the_field(gaps):
    assert gaps[0.0]["fixed"] == pytest.approx(0.201, abs=0.01) and gaps[0.0]["provisional"] == pytest.approx(0.318, abs=0.01)
    assert gaps[0.0]["provisional"] > gaps[0.0]["fixed"] + 0.08
    assert gaps[0.5]["provisional"] - gaps[0.5]["fixed"] < 0.05                # etwa gleich bei kleinem Abstand


# --- Texte -------------------------------------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_values(k_rows, kalman, vs_bt, order, newcomer, gaps):
    level, text = ev.verdict_k(list(k_rows.values()))
    assert level == "info" and "K = 20" in text and ev.de(k_rows[(0.0, 2)]["error"], 3) in text and ev.de(k_rows[(0.0, 160)]["error"], 3) in text
    level, text = ev.verdict_kalman(list(kalman.values()))
    assert "12 Punkte" in text and "88 bei" in text
    level, text = ev.verdict_bt(list(vs_bt.values()))
    assert level == "warning" and ev.de(vs_bt[0.0]["elo_rmse"], 3) in text and ev.de(vs_bt[0.12]["bt_rmse"], 3) in text
    level, text = ev.verdict_order(list(order.values()))
    assert level == "warning" and "75 %" in text and "100 %" in text
    level, text = ev.verdict_newcomer(list(newcomer.values()), list(gaps.values()))
    assert ev.de(gaps[1.0]["provisional"], 3) in text and ev.de(gaps[0.0]["fixed"], 3) in text
