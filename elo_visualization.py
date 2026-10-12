"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import elo_constants as C
import elo_rating as E


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_trajectories(a):
    """Wertung nach jeder Runde (durchgezogen) gegen die wahre Stärke (gestrichelt), je Spieler, in Elo-Punkten um 1500."""
    m, run = a.matches, a.run
    rounds = np.arange(1, m.rounds + 1)
    fig = go.Figure()
    center = np.nanmean(run.history, axis=1)                             # Mittel der aktiven Spieler je Runde
    for p in range(m.n):
        color = C.PALETTE[p % len(C.PALETTE)]
        fig.add_trace(go.Scatter(x=rounds, y=E.to_points(run.history[:, p] - center), mode="lines", name=m.names[p], line=dict(color=color, width=2),
                                 connectgaps=False, hovertemplate=m.names[p] + ", Runde %{x}: %{y:.0f} Punkte<extra></extra>"))
        truth = m.truth[:, p] - m.truth.mean(axis=1)
        fig.add_trace(go.Scatter(x=rounds, y=E.to_points(truth), mode="lines", showlegend=False, line=dict(color=color, width=1, dash="dot"),
                                 hovertemplate=m.names[p] + " wahr, Runde %{x}: %{y:.0f}<extra></extra>"))
    if m.join_round:
        fig.add_vline(x=m.join_round + 1, line_dash="dash", line_color="rgba(150,150,150,0.7)", annotation_text="Neuling kommt", annotation_position="top left")
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Wertung in Elo-Punkten (gepunktet: wahre Stärke)")
    return _base(fig, height=430)


def build_k(rows, sigmas=C.SIGMA_SWEEP):
    fig = go.Figure()
    shades = ("#9cc5bf", "#5f9d95", "#2F6B65", "#1f4a45")
    for idx, sg in enumerate(sigmas):
        sel = sorted((r for r in rows if r["sigma"] == sg), key=lambda r: r["k"])
        fig.add_trace(go.Scatter(x=[str(r["k"]) for r in sel], y=[r["error"] for r in sel], mode="lines+markers", name=f"Drift σ = {str(sg).replace('.', ',')}",
                                 line=dict(color=shades[idx % 4], width=3), marker=dict(size=[13 if r["best"] else 7 for r in sel], symbol="diamond"),
                                 hovertemplate=f"σ = {sg}, K = %{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title_text="K in Elo-Punkten (Raute: bestes K)", type="category")
    fig.update_yaxes(title_text="Verfolgungsfehler (θ-Einheiten)")
    return _base(fig)


def build_kalman(rows):
    xs = [f"σ = {str(r['sigma']).replace('.', ',')}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["k_star"] for r in rows], name="Kalman-Verstärkung K*", marker_color=C.COLOR_KALMAN,
                         hovertemplate="%{x}: K* = %{y:.0f} Punkte<extra></extra>"))
    fig.add_trace(go.Bar(x=xs, y=[r["best_error"] for r in rows], name="bestes K nach Fehler", marker_color=C.COLOR_ELO,
                         hovertemplate="%{x}: %{y} Punkte<extra></extra>"))
    fig.add_trace(go.Bar(x=xs, y=[r["best_log_loss"] for r in rows], name="bestes K nach Log-Verlust", marker_color="#9cc5bf",
                         hovertemplate="%{x}: %{y} Punkte<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="K in Elo-Punkten")
    return _base(fig)


def build_vs_bt(rows):
    xs = [str(r["sigma"]).replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["elo_rmse"] for r in rows], name="Elo (bestes K)", marker_color=C.COLOR_ELO,
                         hovertemplate="σ = %{x}: Elo %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=xs, y=[r["bt_rmse"] for r in rows], name="Bradley-Terry über alle Partien", marker_color=C.COLOR_BT,
                         hovertemplate="σ = %{x}: Bradley-Terry %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Drift σ je Runde", type="category")
    fig.update_yaxes(title_text="Endfehler gegen die wahre Stärke (θ-Einheiten)")
    return _base(fig)


def build_order(rows):
    xs = [str(r["k"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["sd_points"] for r in rows], name="Streuung der Endwertung (Punkte)", marker_color=C.COLOR_ELO,
                         hovertemplate="K = %{x}: %{y:.0f} Punkte<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["flip"] for r in rows], name="Läufe mit anderem Spitzenreiter", mode="lines+markers", yaxis="y2",
                             line=dict(color=C.COLOR_KALMAN, width=3), hovertemplate="K = %{x}: %{y:.0%}<extra></extra>"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], tickformat=".0%", title="anderer Spitzenreiter", fixedrange=True))
    fig.update_xaxes(title_text="K in Elo-Punkten", type="category")
    fig.update_yaxes(title_text="Streuung der Endwertung (Punkte)", rangemode="tozero")
    return _base(fig)


def build_newcomer(rows):
    gs = ("g1", "g5", "g10", "g20")
    xs = ["1", "5", "10", "20"]
    colors = ("#9cc5bf", "#5f9d95", "#8e6bb0", "#c0392b", "#d98c1f")
    fig = go.Figure()
    for idx, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=xs, y=[r[g] for g in gs], mode="lines+markers", name=r["label"], line=dict(color=colors[idx % len(colors)], width=3),
                                 hovertemplate=r["label"] + ", %{x} Partien: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="eigene Partien des Neulings", type="category")
    fig.update_yaxes(title_text="Fehler der Wertung (θ-Einheiten)")
    return _base(fig)
