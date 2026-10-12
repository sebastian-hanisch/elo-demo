"""Kennzahlen, Verdikt-Logik und Experimente (über die festen Instanzen SWEEP_SEEDS)."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import elo_batch as B
import elo_constants as C
import elo_drift as D
import elo_rating as E
import turnier_generator as T
from elo_scenario import Settings, make_matches


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    matches: D.Matches
    run: E.EloRun
    bt: np.ndarray             # Bradley-Terry über alle Partien (θ, Mittel 0)
    truth_end: np.ndarray      # wahre Stärken in der letzten Runde (zentriert)
    games: np.ndarray          # Partien je Spieler
    rmse: dict                 # Verfahren -> RMSE der zentrierten Endwertung gegen die wahre Endstärke (θ-Einheiten)
    tau: dict                  # Verfahren -> τ der Endwertung gegen die wahre Endstärke
    info: float                # mittlere Information je Partie bei den Anfangsstärken
    k_star: float              # Kalman-Verstärkung als K in Punkten (nur in der Simulation bekannt)


def centered(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def rmse_centered(a, b) -> float:
    return float(np.sqrt(np.mean((centered(a) - centered(b)) ** 2)))


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    iu = np.triu_indices(len(a), 1)
    sa, sb = np.sign(a[iu[0]] - a[iu[1]]), np.sign(b[iu[0]] - b[iu[1]])
    counted = (sa != 0) & (sb != 0)
    if not counted.any():
        return float("nan")
    return float((sa[counted] * sb[counted]).mean())


@functools.lru_cache(maxsize=4096)
def analyse(settings: Settings) -> Analysis:
    """Gleiche Reglerwerte, gleiche Analyse (Settings ist unveränderlich)."""
    s = settings
    m = make_matches(s)
    run = E.run_elo(m, s.k, s.k_start)
    bt = B.fit_bradley_terry(B.win_matrix(m), B.oracle_lambda(s.n, s.delta))
    end = centered(m.truth[-1])
    games = np.zeros(s.n, dtype=int)
    np.add.at(games, m.i, 1)
    np.add.at(games, m.j, 1)
    info = E.mean_information(T.make_strengths(s.n, s.delta))
    return Analysis(s, m, run, bt, end, games,
                    {"Elo": rmse_centered(run.final, end), "Bradley-Terry": rmse_centered(bt, end)},
                    {"Elo": kendall_tau(run.final, end), "Bradley-Terry": kendall_tau(bt, end)},
                    info, E.kalman_k_points(s.sigma, info))


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else float("nan")


@functools.lru_cache(maxsize=4096)
def _matches(sigma: float, seed: int) -> D.Matches:
    """Die feste Instanz der Experimente (8 Spieler, 150 Runden, kein Neuling); gleiche (Drift, Seed) teilen sich die Partien über alle Experimente."""
    return D.simulate_matches(C.EXP_N, C.EXP_DELTA, C.EXP_ROUNDS, sigma, 0, 0.0, seed)


@functools.lru_cache(maxsize=32768)
def _elo(sigma: float, seed: int, k: float) -> tuple:
    """(Verfolgungsfehler, Log-Verlust, Endwertungen) eines Elo-Laufs; die Experimente 1 bis 3 teilen sich gleiche Läufe."""
    r = E.run_elo(_matches(sigma, seed), k)
    return r.tracking_error, r.log_loss, r.final


# --- Experiment 1: K ist ein Kompromiss -------------------------------------------------------------------------------------------------------

def sweep_k(sigmas=C.SIGMA_SWEEP, ks=C.K_SWEEP, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Verfolgungsfehler und Log-Verlust vor der Partie je K und Drift (8 Spieler, 150 Runden, kein Neuling)."""
    rows = []
    for sg in sigmas:
        acc = {k: ([], []) for k in ks}
        for sd in seeds:
            for k in ks:
                err, ll, _ = _elo(sg, sd, k)
                acc[k][0].append(err)
                acc[k][1].append(ll)
        best = min(ks, key=lambda k: _mean(acc[k][0]))
        for k in ks:
            rows.append({"sigma": sg, "k": k, "error": _mean(acc[k][0]), "log_loss": _mean(acc[k][1]), "best": k == best})
    return rows


# --- Experiment 2: das beste K wächst mit der Drift, wie die Kalman-Verstärkung ----------------------------------------------------------------

def sweep_kalman(sigmas=C.SIGMA_KALMAN, ks=C.K_FINE, seeds=C.SWEEP_SEEDS) -> list[dict]:
    info = E.mean_information(T.make_strengths(C.EXP_N, C.EXP_DELTA))
    rows = []
    for sg in sigmas:
        err = {k: [] for k in ks}
        ll = {k: [] for k in ks}
        for sd in seeds:
            for k in ks:
                e_, l_, _ = _elo(sg, sd, k)
                err[k].append(e_)
                ll[k].append(l_)
        rows.append({"sigma": sg, "q": sg * sg, "k_star": E.kalman_k_points(sg, info),
                     "best_error": min(ks, key=lambda k: _mean(err[k])), "best_log_loss": min(ks, key=lambda k: _mean(ll[k]))})
    return rows


# --- Experiment 3: Elo gegen Bradley-Terry über alle Partien -----------------------------------------------------------------------------------

def sweep_vs_bt(sigmas=C.SIGMA_BT, ks=C.K_SWEEP, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Je Drift: das K mit dem kleinsten Verfolgungsfehler, damit Endfehler und τ von Elo gegen Bradley-Terry über alle Partien."""
    lam = B.oracle_lambda(C.EXP_N, C.EXP_DELTA)
    rows = []
    for sg in sigmas:
        track = {k: [] for k in ks}
        final = {k: [] for k in ks}
        bt_err, bt_tau = [], []
        for sd in seeds:
            m = _matches(sg, sd)
            end = centered(m.truth[-1])
            th = B.fit_bradley_terry(B.win_matrix(m), lam)
            bt_err.append(rmse_centered(th, end))
            bt_tau.append(kendall_tau(th, end))
            for k in ks:
                e_, _, fin = _elo(sg, sd, k)
                track[k].append(e_)
                final[k].append((rmse_centered(fin, end), kendall_tau(fin, end)))
        best = min(ks, key=lambda k: _mean(track[k]))
        rows.append({"sigma": sg, "k": best, "elo_rmse": _mean([f[0] for f in final[best]]), "bt_rmse": _mean(bt_err),
                     "elo_tau": _mean([f[1] for f in final[best]]), "bt_tau": _mean(bt_tau)})
    return rows


# --- Experiment 4: dieselben Partien, andere Reihenfolge -----------------------------------------------------------------------------------------

def sweep_order(ks=C.ORDER_KS, repeats=C.ORDER_REPEATS, seeds=C.ORDER_SEEDS) -> list[dict]:
    """Konstante Stärken: dieselben Partien in `repeats` zufälligen Reihenfolgen; Streuung der Endwertung je Spieler und Anteil der Läufe, in denen der Spitzenreiter
    wechselt. Bradley-Terry über alle Partien ist unabhängig von der Reihenfolge (Streuung 0)."""
    rows = []
    for k in ks:
        sds, flips = [], []
        for sd in seeds:
            m = _matches(0.0, sd)
            finals = np.array([centered(E.run_elo(D.reorder_games(m, 1000 * sd + r), k).final) for r in range(repeats)])
            sds.append(float(finals.std(axis=0).mean()))
            flips.append(float(len({int(np.argmax(f)) for f in finals}) > 1))
        rows.append({"k": k, "sd": _mean(sds), "sd_points": C.SCALE * _mean(sds), "flip": _mean(flips)})
    return rows


# --- Experiment 5: ein Neuling im eingespielten Feld ---------------------------------------------------------------------------------------------

def _newcomer_run(sd, gap, k, k0):
    m = D.simulate_matches(C.EXP_N, C.EXP_DELTA, C.NEW_ROUNDS, 0.0, C.NEW_JOIN, gap, sd)
    return E.run_elo(m, k, k0).newcomer_errors


def sweep_newcomer(seeds=C.NEW_SEEDS) -> list[dict]:
    """Fehler der Neuling-Wertung (zentriert gegen das Feld) nach g eigenen Partien: festes K gegen Anlauf-K, Abstand 1."""
    variants = [("K = 10", 10, None), ("K = 20", 20, None), ("K = 40", 40, None), ("K = 20, Anlauf 60", 20, 60.0), ("K = 20, Anlauf 120", 20, 120.0)]
    rows = []
    for label, k, k0 in variants:
        E_ = np.array([_newcomer_run(sd, 1.0, k, k0)[:40] for sd in seeds])
        rows.append({"label": label, "k": k, "k0": k0, "g1": float(E_[:, 0].mean()), "g5": float(E_[:, 4].mean()), "g10": float(E_[:, 9].mean()),
                     "g20": float(E_[:, 19].mean()), "mean": float(E_.mean())})
    return rows


def sweep_newcomer_gap(gaps=C.NEW_GAPS, seeds=C.NEW_SEEDS) -> list[dict]:
    """Derselbe Fehler nach 20 Partien nach Abstand des Neulings vom Feld: K = 20 fest gegen Anlauf-K 60."""
    rows = []
    for gap in gaps:
        fixed = np.mean([_newcomer_run(sd, gap, 20, None)[19] for sd in seeds])
        prov = np.mean([_newcomer_run(sd, gap, 20, 60.0)[19] for sd in seeds])
        rows.append({"gap": gap, "fixed": float(fixed), "provisional": float(prov)})
    return rows


# --- Verdikte (live aus den Messwerten) --------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Primäransicht: Folgt Elo der Stärke besser als ein Schätzer über alle Partien, und was kostet das?"""
    s = a.settings
    r, t = a.rmse, a.tau
    text = (f"Endfehler gegen die wahre Stärke (θ-Einheiten): Elo mit K = {s.k} **{de(r['Elo'], 3)}**, Bradley-Terry über alle Partien **{de(r['Bradley-Terry'], 3)}**; "
            f"τ {de(t['Elo'], 2)} gegen {de(t['Bradley-Terry'], 2)}. Verfolgungsfehler im Lauf {de(a.run.tracking_error, 3)}, Log-Verlust vor der Partie "
            f"{de(a.run.log_loss, 3)}.")
    if s.sigma > 0:
        text += f" Die Kalman-Verstärkung für diese Drift läge bei K ≈ {de(a.k_star, 0)} Punkten."
    if r["Elo"] < r["Bradley-Terry"]:
        return "success", text + " Die Stärken wandern: Elo folgt, der Schätzer über alle Partien mittelt die Vergangenheit."
    if s.sigma == 0:
        return "info", text + " Ohne Drift ist das Mitteln über alle Partien im Vorteil; Elo zahlt für das Fortschreiben."
    return "info", text + " Bei dieser Drift und diesem K gewinnt Elo nicht gegen den Schätzer über alle Partien."


def verdict_k(rows) -> tuple[str, str]:
    st = {r["k"]: r for r in rows if r["sigma"] == 0.0}
    dr = {r["k"]: r for r in rows if r["sigma"] == 0.06}
    best0 = next(r["k"] for r in rows if r["sigma"] == 0.0 and r["best"])
    best1 = next(r["k"] for r in rows if r["sigma"] == 0.06 and r["best"])
    best2 = next(r["k"] for r in rows if r["sigma"] == 0.12 and r["best"])
    return "info", (f"Ohne Drift ist der Verfolgungsfehler bei K = 2 {de(st[2]['error'], 3)}, bei K = {best0} **{de(st[best0]['error'], 3)}** und bei K = 160 "
                    f"{de(st[160]['error'], 3)}: eine U-Form. Mit Drift 0,06 liegt das Optimum bei K = {best1} ({de(dr[best1]['error'], 3)}), mit 0,12 bei K = {best2}. Zu "
                    "kleines K hinkt hinterher, zu großes folgt dem Zufall der letzten Partien.")


def verdict_kalman(rows) -> tuple[str, str]:
    lo, hi = rows[0], rows[-1]
    return "info", (f"Die Kalman-Verstärkung für eine Irrfahrt wächst mit σ: {de(lo['k_star'], 0)} Punkte bei σ = {de(lo['sigma'])}, {de(hi['k_star'], 0)} bei "
                    f"σ = {de(hi['sigma'])}. Das beste K im Raster: {lo['best_error']} gegen {hi['best_error']} (nach Fehler), {lo['best_log_loss']} gegen "
                    f"{hi['best_log_loss']} (nach Log-Verlust). Die Skalierung stimmt; der Wert liegt bei kleiner Drift unter dem besten K, weil der Kaltstart und "
                    "der unsichere Gegner mitspielen. Die Verstärkung ist nur in der Simulation bekannt (σ, Information).")


def verdict_bt(rows) -> tuple[str, str]:
    r0 = next(r for r in rows if r["sigma"] == 0.0)
    r3 = next(r for r in rows if r["sigma"] == 0.03)
    r12 = next(r for r in rows if r["sigma"] == 0.12)
    return "warning", (f"Ohne Drift verliert Elo gegen Bradley-Terry über alle Partien (Fehler {de(r0['elo_rmse'], 3)} gegen {de(r0['bt_rmse'], 3)}, τ "
                       f"{de(r0['elo_tau'], 2)} gegen {de(r0['bt_tau'], 2)}). Bei σ = 0,03 steht es gleich ({de(r3['elo_rmse'], 3)} gegen {de(r3['bt_rmse'], 3)}); "
                       f"bei σ = 0,12 gewinnt Elo klar ({de(r12['elo_rmse'], 3)} gegen {de(r12['bt_rmse'], 3)}). Der Wechsel liegt hier bei etwa 0,03 je Runde "
                       "(150 Runden, 8 Spieler), nicht allgemein.")


def verdict_order(rows) -> tuple[str, str]:
    r = {x["k"]: x for x in rows}
    return "warning", (f"Dieselben Partien in anderer Reihenfolge ändern die Endwertung: bei K = 20 um {de(r[20]['sd'], 3)} (θ-Einheiten, rund "
                       f"{de(r[20]['sd_points'], 0)} Punkte) je Spieler, und in {pct(r[20]['flip'])} % der Läufe wechselt der Spitzenreiter; bei K = 80 in "
                       f"{pct(r[80]['flip'])} %. Bradley-Terry über alle Partien hängt nicht an der Reihenfolge (Streuung 0).")


def verdict_newcomer(rows, gaps) -> tuple[str, str]:
    r = {x["label"]: x for x in rows}
    g = {x["gap"]: x for x in gaps}
    return "info", (f"Ein Neuling mit Abstand 1 vom Feld: Fehler nach 20 eigenen Partien bei K = 20 {de(r['K = 20']['g20'], 3)}, mit Anlauf-K 60 "
                    f"**{de(r['K = 20, Anlauf 60']['g20'], 3)}**, bei festem K = 40 {de(r['K = 40']['g20'], 3)}. Der Anlauf-K ist auf lange Sicht nicht lauter. Er "
                    f"hilft nur Abweichenden: bei Abstand 1 {de(g[1.0]['provisional'], 3)} statt {de(g[1.0]['fixed'], 3)}, bei Abstand 0 aber "
                    f"{de(g[0.0]['provisional'], 3)} statt {de(g[0.0]['fixed'], 3)}.")
