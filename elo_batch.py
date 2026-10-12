"""Bradley-Terry über alle Partien (Vergleich zu Elo): penalisierte Maximum Likelihood (Ridge auf θ), Newton-Verfahren mit zentriertem Schritt (Rauschen in
der Richtung „alle gleich“ würde sonst mit 1/λ verstärkt, die Schrittzahl wäre plattformabhängig). Nur numpy."""

import numpy as np

from elo_drift import Matches, sigmoid


def oracle_lambda(n: int, delta: float) -> float:
    """1/Varianz der Anfangsstärken: Var(s) = δ²(n² − 1)/12 (nur in der Simulation bekannt)."""
    return 12.0 / (delta ** 2 * (n ** 2 - 1))


def win_matrix(m: Matches) -> np.ndarray:
    """W[i, j] = Siege von i gegen j."""
    W = np.zeros((m.n, m.n))
    np.add.at(W, (m.i, m.j), m.win.astype(float))
    np.add.at(W, (m.j, m.i), 1.0 - m.win.astype(float))
    return W


def fit_bradley_terry(W, lam: float, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Stärken θ mit Mittelwert 0; λ > 0 hält die Schätzung endlich (auch für Spieler ohne Partie oder ohne Sieg)."""
    if lam <= 0:
        raise ValueError("lam muss positiv sein")
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    G = W + W.T
    n = len(W)
    th = np.zeros(n)
    for _ in range(max_iter):
        P = sigmoid(th[:, None] - th[None, :])
        grad = W.sum(axis=1) - (G * P).sum(axis=1) - lam * th
        V = G * P * (1.0 - P)
        np.fill_diagonal(V, 0.0)
        info = np.diag(V.sum(axis=1)) - V + lam * np.eye(n)
        step = np.linalg.solve(info, grad)
        step -= step.mean()
        th = th + step
        if np.abs(step).max() < tol:
            break
    return th - th.mean()
