"""Bradley-Terry über alle Partien: Handwerte, Reihenfolgen-Unabhängigkeit, Eingabeprüfung."""

import math

import numpy as np
import pytest

import elo_batch as B
import elo_drift as D


def test_pair_hand_value_and_centering():
    W = np.array([[0, 3], [1, 0]])
    th = B.fit_bradley_terry(W, 1e-9)
    assert th == pytest.approx([0.5 * math.log(3), -0.5 * math.log(3)], abs=1e-6)
    assert th.sum() == pytest.approx(0.0, abs=1e-12)


def test_prior_keeps_the_estimate_finite_for_an_undefeated_player():
    W = np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]])
    th = B.fit_bradley_terry(W, 0.5)
    assert np.isfinite(th).all() and th[0] > th[1] > th[2]
    with pytest.raises(ValueError, match="lam muss positiv sein"):
        B.fit_bradley_terry(W, 0.0)


def test_win_matrix_hand_value_and_order_independence():
    # Partien: 0 schlägt 1; 2 schlägt 1 (i = 1, j = 2, i verliert); 0 schlägt 2
    m = D.Matches(3, np.zeros((2, 3)), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([1, 0, 1]), 0, ("A", "B", "C"))
    assert B.win_matrix(m).tolist() == [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    big = D.simulate_matches(8, 0.3, 30, 0.0, 0, 0.0, 4)
    shuffled = D.reorder_games(big, 3)
    assert np.array_equal(B.win_matrix(big), B.win_matrix(shuffled))
    lam = B.oracle_lambda(8, 0.3)
    assert B.fit_bradley_terry(B.win_matrix(big), lam) == pytest.approx(B.fit_bradley_terry(B.win_matrix(shuffled), lam))


def test_oracle_lambda_formula():
    assert B.oracle_lambda(8, 0.3) == pytest.approx(12 / (0.09 * 63))
