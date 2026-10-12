"""Elo-Wertung (nur numpy): Aktualisierung Partie für Partie, Anlauf-K für Neulinge, Verfolgungsfehler, Vorhersage vor der Partie und die Kalman-Verstärkung
als Vergleichswert. Alle Wertungen intern in logistischen Einheiten θ; die Anzeige rechnet mit SCALE = 400/ln 10 in Punkte um (1500 + SCALE θ).

Update: θ_i += k (S − E), θ_j −= k (S − E), E = σ(θ_i − θ_j), k = K/SCALE; bei gleichem k bleibt die Summe der Wertungen erhalten (Nullsumme), mit Anlauf-K
nicht."""

import math
from dataclasses import dataclass

import numpy as np

import elo_constants as C
from elo_drift import Matches, sigmoid


def expected_score(theta_i: float, theta_j: float) -> float:
    """E = σ(θ_i − θ_j) in Skalarrechnung, ohne Überlauf."""
    x = float(theta_i) - float(theta_j)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def elo_update(theta_i: float, theta_j: float, score_i: float, k_i: float, k_j: float | None = None, expected: float | None = None) -> tuple:
    """Neue Wertungen nach einer Partie; score_i = 1 (i gewinnt) oder 0. Mit k_j = None ist die Aktualisierung symmetrisch (Nullsumme); ungleiche k
    (Anlauf-K) brechen die Nullsumme: wer das größere k hat, gewinnt mehr, als der andere verliert. `expected` spart die zweite Berechnung der Erwartung."""
    e = expected_score(theta_i, theta_j) if expected is None else expected
    k_j = k_i if k_j is None else k_j
    return theta_i + k_i * (score_i - e), theta_j - k_j * (score_i - e)


def effective_k(K: float, K0: float | None, played: float, g0: float = C.G0) -> float:
    """K in Punkten: fest, oder mit Anlauf K0 für Neulinge, nach g0 Partien halbiert, strebt gegen K."""
    if K0 is None:
        return K
    return K + (K0 - K) / (1.0 + played / g0)


@dataclass(frozen=True)
class EloRun:
    history: np.ndarray        # (Runden, n): Wertung nach der Runde (θ), NaN solange der Neuling fehlt
    final: np.ndarray          # Wertung nach der letzten Partie
    log_loss: float            # mittlerer Log-Verlust der Vorhersage vor jeder Partie
    tracking_error: float      # mittlerer RMSE der zentrierten Wertungen gegen die zentrierten wahren Stärken ab Runde BURN
    newcomer_errors: tuple     # |Wertung − wahre Stärke| des Neulings (zentriert gegen die anderen) nach jeder seiner Partien


def run_elo(m: Matches, K: float, K0: float | None = None, g0: float = C.G0, burn: int = C.BURN) -> EloRun:
    n, rounds = m.n, m.rounds
    k_fixed = K / C.SCALE
    R = [0.0] * n                                       # Skalarrechnung (math statt numpy je Partie): rund zehnmal schneller
    played = [0] * n
    hist = np.full((rounds, n), np.nan)
    errs, new_errs = [], []
    ll = 0.0
    cnt = 0
    g = 0
    gi, gj, gw, gr = m.i.tolist(), m.j.tolist(), m.win.tolist(), m.rnd.tolist()
    G_ = len(gi)
    for t in range(rounds):
        if m.join_round and t == m.join_round:
            R[0] = sum(R[1:]) / (n - 1)                 # Startwertung des Neulings: Mittel des Feldes
        while g < G_ and gr[g] == t:
            i, j, w = gi[g], gj[g], float(gw[g])
            p = expected_score(R[i], R[j])
            ll += -math.log(max(p if w else 1.0 - p, 1e-12))
            cnt += 1
            if K0 is None:
                ki = kj = k_fixed
            else:
                ki, kj = effective_k(K, K0, played[i], g0) / C.SCALE, effective_k(K, K0, played[j], g0) / C.SCALE
            played[i] += 1
            played[j] += 1
            R[i], R[j] = elo_update(R[i], R[j], w, ki, kj, p)
            if m.join_round and (i == 0 or j == 0) and t >= m.join_round:
                est = R[0] - sum(R[1:]) / (n - 1)
                tru = m.truth[t][0] - m.truth[t][1:].mean()
                new_errs.append(abs(est - tru))
            g += 1
        Ra = np.array(R)
        active = np.ones(n, dtype=bool)
        if m.join_round and t < m.join_round:
            active[0] = False
        hist[t, active] = Ra[active]
        if t >= burn:
            c = Ra[active] - Ra[active].mean()
            tr = m.truth[t][active] - m.truth[t][active].mean()
            errs.append(float(np.sqrt(np.mean((c - tr) ** 2))))
    return EloRun(hist, np.array(R), ll / max(cnt, 1), float(np.mean(errs)) if errs else float("nan"), tuple(new_errs))


def mean_information(strengths) -> float:
    """Mittlere Information p(1 − p) einer Partie bei den Anfangsstärken (alle Paare): Beobachtungsrauschen im Kalman-Vergleich."""
    s = np.asarray(strengths, dtype=float)
    p = sigmoid(s[:, None] - s[None, :])
    mask = ~np.eye(len(s), dtype=bool)
    return float((p * (1.0 - p))[mask].mean())


def kalman_gain(sigma: float, info: float) -> float:
    """Stationäre Verstärkung k* eines Filters für eine Irrfahrt mit Schrittvarianz q = σ² und Information I je Partie: k* = P mit
    P = (−qI + √(q²I² + 4Iq)) / (2I), die Lösung der stationären Riccati-Gleichung P = (P + q)/(1 + (P + q) I)."""
    q = sigma * sigma
    if q <= 0 or info <= 0:
        return 0.0
    return (-q * info + np.sqrt(q * q * info * info + 4.0 * info * q)) / (2.0 * info)


def kalman_k_points(sigma: float, info: float) -> float:
    """Dieselbe Verstärkung als K in Elo-Punkten."""
    return C.SCALE * kalman_gain(sigma, info)


def to_points(theta) -> np.ndarray:
    """θ (logistische Einheiten, Mittel 0) in Elo-Punkte um 1500."""
    return C.BASE + C.SCALE * np.asarray(theta, dtype=float)
