"""Spiele mit wandernden Stärken (lokal; der gemeinsame Generator der Linie liefert nur die Anfangsstärken und bleibt unverändert).

Die Stärken starten bei turnier_generator.make_strengths (Mittel 0) und folgen einer Irrfahrt mit Schritt σ je Runde. In jeder Runde werden die aktiven Spieler
zufällig gepaart (bei ungerader Zahl setzt einer aus); P(i schlägt j) = σ(θ_i(t) − θ_j(t)). Optional kommt ein Neuling (Spieler 0) in einer Runde dazu, mit
einer wahren Stärke im Abstand `gap` vom Mittel der anderen. Zufall nur über SplitMix64; Normalverteilung per Box-Muller aus zwei Gleichverteilten."""

from dataclasses import dataclass

import numpy as np

import turnier_generator as T

_SALT = 0xE10


@dataclass(frozen=True)
class Matches:
    n: int
    truth: np.ndarray          # (Runden, n): wahre Stärken während der Runde
    rnd: np.ndarray            # Runde je Partie
    i: np.ndarray              # Spieler i
    j: np.ndarray              # Spieler j
    win: np.ndarray            # 1, wenn i gewinnt
    join_round: int            # 0 = kein Neuling; sonst kommt Spieler 0 in dieser Runde
    names: tuple

    @property
    def rounds(self) -> int:
        return len(self.truth)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def normals(rng: T.SplitMix64, size: int) -> np.ndarray:
    """Standardnormalverteilte Zahlen per Box-Muller aus der Gleichverteilten des Generators."""
    u1 = 1.0 - rng.uniform(size)            # in (0, 1]
    u2 = rng.uniform(size)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def simulate_matches(n: int, delta: float, rounds: int, sigma: float, join_round: int, gap: float, seed: int) -> Matches:
    if join_round and not (0 < join_round < rounds):
        raise ValueError("Der Neuling muss vor dem Ende der Runden kommen")
    s = T.make_strengths(n, delta)
    theta = s - s.mean()
    if join_round:
        theta[0] = theta[1:].mean() + gap
    rng = T.SplitMix64(seed ^ _SALT)
    perm = rng.permutation(n)
    names = tuple(T.LABELS[int(p)] for p in perm)
    truth = np.zeros((rounds, n))
    R, I, J, W = [], [], [], []
    for t in range(rounds):
        truth[t] = theta
        active = np.arange(n) if (not join_round or t >= join_round) else np.arange(1, n)
        order = active[rng.permutation(len(active))]
        m = len(order) // 2
        a, b = order[:2 * m:2], order[1:2 * m:2]
        win = rng.uniform(m) < sigmoid(theta[a] - theta[b])
        R.extend([t] * m)
        I.extend(a.tolist())
        J.extend(b.tolist())
        W.extend(win.astype(int).tolist())
        if sigma > 0:
            theta = theta + sigma * normals(rng, n)
    return Matches(n, truth, np.array(R, dtype=int), np.array(I, dtype=int), np.array(J, dtype=int), np.array(W, dtype=int), int(join_round), names)


def reorder_games(m: Matches, order_seed: int) -> Matches:
    """Dieselben Partien in zufälliger anderer Reihenfolge; die Runden-Zuordnung der Plätze bleibt, es ändert sich nur, wer in welcher Position spielt."""
    perm = T.SplitMix64(order_seed ^ 0x0DE5).permutation(len(m.i))
    return Matches(m.n, m.truth, m.rnd, m.i[perm], m.j[perm], m.win[perm], m.join_round, m.names)
