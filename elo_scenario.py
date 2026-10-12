"""Szenario: Reglerwerte (Settings) und die daraus erzeugten Spiele."""

from dataclasses import dataclass

import elo_constants as C
import elo_drift as D


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    rounds: int = C.DEFAULT_ROUNDS
    sigma: float = C.DEFAULT_SIGMA
    k: int = C.DEFAULT_K
    provisional: bool = False      # Anlauf-K für Neulinge
    k0: int = C.DEFAULT_K0
    newcomer: bool = False         # ein Spieler kommt später dazu
    join: int = C.DEFAULT_JOIN
    gap: float = C.DEFAULT_GAP
    seed: int = C.DEFAULT_SEED

    @property
    def join_round(self) -> int:
        """Runde, in der der Neuling kommt (0 = keiner); begrenzt auf [2, Runden − 2]."""
        return min(max(self.join, C.JOIN_MIN), self.rounds - 2) if self.newcomer else 0

    @property
    def k_start(self):
        """K0 für Neulinge oder None (festes K)."""
        return float(self.k0) if self.provisional else None


def make_matches(s: Settings) -> D.Matches:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiche Partien."""
    return D.simulate_matches(s.n, s.delta, s.rounds, s.sigma, s.join_round, s.gap, s.seed)
