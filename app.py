"""Elo: die Stärke Partie für Partie - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Ranking-Linie: Elo schreibt die Wertung nach jeder Partie fort, mit fester Verstärkung K. Fünf Experimente messen, was K kostet und bringt: der
Kompromiss zwischen Rauschen und Verzögerung, der Vergleich mit einem Schätzer über alle Partien, die Pfadabhängigkeit und der Neuling. Aufbau und Regeln:
DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import elo_constants as C
import elo_drift as D
import elo_evaluation as ev
import elo_rating as E
import turnier_generator as T
from elo_pdf_export import build_pdf
from elo_presets import (
    apply_preset,
    bounds,
    clamp_kept,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from elo_scenario import Settings
from elo_visualization import build_k, build_kalman, build_newcomer, build_order, build_trajectories, build_vs_bt

st.set_page_config(page_title="Elo – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 100 feste Läufe …")
def _k():
    return ev.sweep_k()


@st.cache_data(show_spinner="Experiment rechnet über 100 feste Läufe …")
def _kalman():
    return ev.sweep_kalman()


@st.cache_data(show_spinner="Experiment rechnet über 100 feste Läufe …")
def _vs_bt():
    return ev.sweep_vs_bt()


@st.cache_data(show_spinner="Experiment rechnet über 50 feste Läufe mit je 16 Reihenfolgen …")
def _order():
    return ev.sweep_order()


@st.cache_data(show_spinner="Experiment rechnet über 100 feste Läufe …")
def _newcomer():
    return ev.sweep_newcomer(), ev.sweep_newcomer_gap()


def verdict(level, text):
    SHOW[level](text)


st.title("♟️ Elo: die Stärke Partie für Partie")
st.markdown(
    """
Bradley-Terry schätzt die Stärken aus **allen Partien auf einmal**. Diese Demo ist das **siebte Stück der Ranking-Linie** und fragt, was passiert, wenn man die
Wertung stattdessen **nach jeder Partie fortschreibt**: Elo. Das ist ein **Filter mit fester Verstärkung K**. Gemessen wird, dass K ein Kompromiss zwischen Rauschen
und Verzögerung ist, den die **Drift der Stärken** bestimmt; dass Elo bei wandernden Stärken einen Schätzer über alle Partien schlägt und bei konstanten verliert;
dass die Wertung am **Pfad** hängt; und dass ein **Neuling** ein größeres K braucht, wenn er vom Feld abweicht.

Weiter unten: fünf **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So rechnet Elo", expanded=True):
    st.markdown(
        """
1. **Die Erwartung:** Zwei Spieler mit den Wertungen R_i und R_j. Die erwartete Punktzahl von i ist E = 1/(1 + 10^(−(R_i − R_j)/400)): **400 Punkte Unterschied heißen
   Faktor 10 in den Gewinnchancen**. In dieser Demo rechnet das Modell in logistischen Einheiten θ; 1 Einheit sind 173,7 Punkte.
2. **Die Aktualisierung:** Nach der Partie wandert die Wertung um K · (Ergebnis − Erwartung). Gewinnt der Favorit, passiert wenig; gewinnt der Außenseiter, viel. Was
   der eine gewinnt, verliert der andere: Bei gleichem K bleibt die **Summe der Wertungen gleich**; mit Anlauf-K gewinnt der Neuling mehr, als der Gegner verliert.
3. **K ist die Verstärkung:** Groß heißt, die Wertung folgt schnell, aber auch dem Zufall; klein heißt ruhig, aber verspätet. Es gibt kein richtiges K ohne eine
   Annahme darüber, wie schnell die Stärken wandern.
4. **Was fehlt:** Elo führt keine Unsicherheit mit und sieht die Zukunft nicht. Beides ist Thema der Folgestücke (Glicko, TrueSkill, WHR).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Spieler und Runden**")
    n = st.slider("Spieler n", *bounds("n_slider"), key="n_slider", help="Wie viele Spieler im Turnier antreten (jede Runde zufällig gepaart).")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Spieler zu Beginn (logistische Einheiten).")
    rounds = st.slider("Runden", *bounds("rounds_slider"), key="rounds_slider", help="Jeder Spieler spielt je Runde einmal (bei ungeradem n setzt einer aus).")
    st.markdown("**Drift der Stärken**")
    sigma = st.slider("Schritt der Irrfahrt σ je Runde", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider",
                      help="Wie weit die wahren Stärken je Runde wandern (logistische Einheiten). 0 = konstant.")
    st.markdown("**Elo**")
    k = st.slider("K in Elo-Punkten", *bounds("k_slider"), key="k_slider", help="Verstärkung: Wertungsänderung je Partie ist K mal (Ergebnis minus Erwartung).")
    info = E.mean_information(T.make_strengths(int(n), float(delta)))
    if sigma > 0:
        st.caption(f"Kalman-Verstärkung für diese Drift: K* ≈ {ev.de(E.kalman_k_points(float(sigma), info), 0)} Punkte (nur in der Simulation bekannt).")
    provisional = st.checkbox("Anlauf-K für Neulinge", key="provisional_toggle", help="Höheres K in den ersten Partien eines Spielers, strebt gegen K.")
    if provisional:
        seed_widget("k0_slider")
        k0 = st.slider("Anlauf-K0 in Punkten", *bounds("k0_slider"), key="k0_slider", help="K für die allererste Partie; nach 10 Partien halbiert sich der Aufschlag.")
        remember("k0_slider", k0)
    st.markdown("**Neuling**")
    newcomer = st.checkbox("Ein Neuling kommt später dazu", key="newcomer_toggle", help="Spieler 0 beginnt erst in der gewählten Runde; seine Startwertung ist das Feldmittel.")
    if newcomer:
        clamp_kept("join_slider", C.JOIN_MIN, int(rounds) - 2)
        seed_widget("join_slider")
        join = st.slider("Neuling kommt in Runde", C.JOIN_MIN, int(rounds) - 2, key="join_slider")
        remember("join_slider", join)
        seed_widget("gap_slider")
        gap = st.slider("Abstand des Neulings vom Feld", *bounds("gap_slider"), step=C.GAP_STEP, key="gap_slider",
                        help="Wahre Stärke des Neulings minus Mittel des Feldes (logistische Einheiten).")
        remember("gap_slider", gap)
    seed = st.number_input("Zufalls-Seed des Turniers", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Turnier würfeln", width="stretch", on_click=randomize_seed)

k0_value = int(current_value("k0_slider"))
join_value = int(current_value("join_slider"))
gap_value = round(float(current_value("gap_slider")), 1)
sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "rounds_slider": int(rounds), "sigma_slider": round(float(sigma), 2), "k_slider": int(k),
                   "provisional_toggle": bool(provisional), "k0_slider": k0_value, "newcomer_toggle": bool(newcomer), "join_slider": join_value,
                   "gap_slider": gap_value, "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(rounds), round(float(sigma), 2), int(k), bool(provisional), k0_value, bool(newcomer), join_value, gap_value, int(seed))
a = _analysis(settings)
m = a.matches

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Wertungen")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Endfehler: Elo", ev.de(a.rmse["Elo"], 3), delta=f"Bradley-Terry {ev.de(a.rmse['Bradley-Terry'], 3)}", delta_color="off",
          help="RMSE der zentrierten Endwertung gegen die wahre Endstärke, in logistischen Einheiten. Kleiner ist besser.")
m2.metric("τ der Endwertung: Elo", ev.de(a.tau["Elo"]), delta=f"Bradley-Terry {ev.de(a.tau['Bradley-Terry'])}", delta_color="off")
m3.metric("Verfolgungsfehler im Lauf", ev.de(a.run.tracking_error, 3), delta=f"Log-Verlust vor der Partie {ev.de(a.run.log_loss, 3)}", delta_color="off",
          help="Mittlerer RMSE der zentrierten Wertungen gegen die wahren Stärken ab Runde 30 (der Kaltstart zählt nicht).")
m4.metric("Kalman-K*", "–" if settings.sigma == 0 else f"{ev.de(a.k_star, 0)} Punkte", delta=f"gewählt: K = {settings.k}", delta_color="off",
          help="Stationäre Kalman-Verstärkung für diese Drift; nur in der Simulation bekannt.")

st.plotly_chart(build_trajectories(a), width="stretch", key="trajectories_chart")
st.caption("Wertung nach jeder Runde (durchgezogen) und wahre Stärke (gepunktet) je Spieler, in Elo-Punkten um das Feldmittel 1500. Die wahre Stärke ist hier bekannt, "
           "weil das Turnier simuliert ist.")

final_pts = E.to_points(a.run.final - a.run.final.mean())
true_pts = E.to_points(a.truth_end)
bt_pts = E.to_points(a.bt)
order = sorted(range(settings.n), key=lambda p: -a.truth_end[p])


def places(values):
    """Platz je Spieler (1 = bester) nach absteigender Wertung."""
    return {p: r + 1 for r, p in enumerate(sorted(range(settings.n), key=lambda q: -values[q]))}


r_elo, r_bt, r_true = places(a.run.final), places(a.bt), places(a.truth_end)
df = pd.DataFrame({
    "Spieler": [m.names[p] for p in order],
    "Partien": [int(a.games[p]) for p in order],
    "Wertung Elo": [round(float(final_pts[p])) for p in order],
    "wahre Stärke (Punkte)": [round(float(true_pts[p])) for p in order],
    "Bradley-Terry (Punkte)": [round(float(bt_pts[p])) for p in order],
    "Platz wahr": [r_true[p] for p in order],
    "Platz Elo": [r_elo[p] for p in order],
    "Platz Bradley-Terry": [r_bt[p] for p in order],
})
st.dataframe(df, hide_index=True, width="stretch")

st.markdown("**Dieselben Partien, andere Reihenfolge**")
st.session_state.setdefault("order_seed", 0)
if st.button("🔀 Partien neu ordnen", key="reorder_button", help="Die gleichen Partien mit den gleichen Ausgängen in einer zufälligen anderen Reihenfolge."):
    st.session_state["order_seed"] += 1
if st.session_state["order_seed"] > 0:
    m2_ = D.reorder_games(m, st.session_state["order_seed"])
    run2 = E.run_elo(m2_, settings.k, settings.k_start)
    new_pts = E.to_points(run2.final - run2.final.mean())
    lead_old, lead_new = max(range(settings.n), key=lambda p: a.run.final[p]), max(range(settings.n), key=lambda p: run2.final[p])
    st.dataframe(pd.DataFrame({
        "Spieler": [m.names[p] for p in order],
        "Wertung Elo": [round(float(final_pts[p])) for p in order],
        "neu geordnet": [round(float(new_pts[p])) for p in order],
        "Änderung": [round(float(new_pts[p] - final_pts[p])) for p in order],
        "Bradley-Terry": [round(float(bt_pts[p])) for p in order],
    }), hide_index=True, width="stretch")
    st.caption(f"Spitzenreiter nach Elo: {m.names[lead_old]} → {m.names[lead_new]}. Bradley-Terry über alle Partien ändert sich nicht, weil es die Reihenfolge nicht "
               "sieht." + (" Mit Anlauf-K ist die Summe der Wertungen nicht erhalten (die Anzeige ist auf das Mittel zentriert)." if settings.provisional else ""))
st.download_button("📄 PDF der aktuellen Wertungen", data=build_pdf(a), file_name="elo.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Folgt Elo der Stärke, und was kostet das?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: K ist ein Kompromiss")
st.caption("8 Spieler (feste Instanz, unabhängig vom Regler), 150 Runden, vier Driften; Verfolgungsfehler der zentrierten Wertungen gegen die wahre Stärke (ab Runde "
           "30), Mittel über 100 feste Läufe.")
rows1 = _k()
st.plotly_chart(build_k(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_k(rows1))

st.subheader("🔬 Experiment 2: Das beste K wächst mit der Drift")
st.caption("8 Spieler (feste Instanz), 150 Runden; feines K-Raster; die Kalman-Verstärkung k* = P der stationären Riccati-Gleichung mit der mittleren Information "
           "I ≈ 0,20 je Partie; 100 feste Läufe.")
rows2 = _kalman()
st.plotly_chart(build_kalman(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_kalman(rows2))
st.dataframe(pd.DataFrame([{"Drift σ": ev.de(r["sigma"]), "Schrittvarianz q = σ²": round(r["q"], 4), "Kalman-K* (Punkte)": round(r["k_star"], 1),
                            "bestes K nach Fehler": r["best_error"], "bestes K nach Log-Verlust": r["best_log_loss"]} for r in rows2]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Elo gegen Bradley-Terry über alle Partien")
st.caption("8 Spieler (feste Instanz), 150 Runden; je Drift das K mit dem kleinsten Verfolgungsfehler; Endfehler der zentrierten Wertungen gegen die wahre Endstärke; "
           "100 feste Läufe.")
rows3 = _vs_bt()
st.plotly_chart(build_vs_bt(rows3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_bt(rows3))
st.dataframe(pd.DataFrame([{"Drift σ": ev.de(r["sigma"], 3), "bestes K": r["k"], "Endfehler Elo": round(r["elo_rmse"], 3), "Endfehler Bradley-Terry": round(r["bt_rmse"], 3),
                            "τ Elo": round(r["elo_tau"], 3), "τ Bradley-Terry": round(r["bt_tau"], 3)} for r in rows3]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Dieselben Partien, andere Reihenfolge")
st.caption("8 Spieler (feste Instanz), 150 Runden, konstante Stärken; dieselben Partien in 16 zufälligen Reihenfolgen; 50 feste Läufe.")
rows4 = _order()
st.plotly_chart(build_order(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_order(rows4))

st.subheader("🔬 Experiment 5: Ein Neuling im eingespielten Feld")
st.caption("8 Spieler, konstante Stärken, der Neuling kommt nach 60 von 100 Runden mit Startwertung Feldmittel; Fehler seiner Wertung gegen die wahre Stärke nach g eigenen "
           "Partien; 100 feste Läufe.")
rows5, gaps5 = _newcomer()
st.plotly_chart(build_newcomer(rows5), width="stretch", key="exp5_chart")
verdict(*ev.verdict_newcomer(rows5, gaps5))
st.dataframe(pd.DataFrame([{"Abstand des Neulings vom Feld": ev.de(g["gap"], 1), "Fehler nach 20 Partien, K = 20 fest": round(g["fixed"], 3),
                            "mit Anlauf-K 60": round(g["provisional"], 3)} for g in gaps5]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Stärke ist konstant oder wandert langsam** | Springt sie oder driftet schnell, folgt eine feste Verstärkung nur mit Verzögerung oder Rauschen; das beste K hängt an der Drift (Experimente 1 und 2). | Glicko und TrueSkill |
| **Ein K passt für alle** | Spieler mit wenigen Partien sind unsicherer als solche mit vielen; Elo führt keine Unsicherheit mit (Experiment 5). | Glicko und TrueSkill |
| **Die Reihenfolge der Partien ist egal** | Elo ist pfadabhängig: dieselben Partien in anderer Reihenfolge geben andere Wertungen und oft einen anderen Spitzenreiter (Experiment 4). | WHR und TrueSkill Through Time |
| **Vergangene Partien bleiben, wie sie sind** | Eine Wertung wird nie rückwirkend verbessert; neue Information über frühere Gegner fließt nicht zurück. | WHR und TrueSkill Through Time |
| **Es spielen Einzelne gegeneinander** | Teamergebnisse und Mehrpersonenspiele brauchen eine eigene Zerlegung. | Von der Mannschaft zum Einzelnen |
| **Ein Spiel ist Sieg oder Niederlage** | Remis, Tore und Heimvorteil haben ein eigenes Modell. | Tore, Remis, Heimvorteil |
| **Alles ist simuliert** | Die Stärken, die Irrfahrt und die Paarung sind Konstruktionen des Generators; die Kalman-Verstärkung ist nur hier bekannt. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Erwartung.** Mit Wertungen $\theta_i, \theta_j$ in logistischen Einheiten ist $E_{ij} = \sigma(\theta_i - \theta_j)$. In Elo-Punkten $R = 1500 + c\,\theta$ mit
$c = 400/\ln 10 \approx 173{,}7$; 400 Punkte Unterschied sind der Faktor 10 in den Gewinnchancen.

**Aktualisierung.** Nach einer Partie mit Ergebnis $S_i \in \{0, 1\}$: $\theta_i \leftarrow \theta_i + k\,(S_i - E_{ij})$, $\theta_j \leftarrow \theta_j - k\,(S_i - E_{ij})$ mit
$k = K/c$. Die Summe der Wertungen ist erhalten. Mit Anlauf: $K_g = K + (K_0 - K)/(1 + g/g_0)$ nach $g$ eigenen Partien.

**Filter-Sicht.** Für eine Irrfahrt $\theta_i(t+1) = \theta_i(t) + \sigma\varepsilon$ mit Beobachtungsinformation $I = p(1-p)$ je Partie ist die stationäre
Verstärkung $k^\ast = P$ mit $P = \frac{-qI + \sqrt{q^2 I^2 + 4 I q}}{2I}$, $q = \sigma^2$ (die Lösung von $P = (P + q)/(1 + (P + q) I)$); für kleines $q$ gilt $k^\ast \approx \sqrt{q/I}$.
Elo ist dieser Filter mit fest gewähltem $k$. Der Wert ist eine Näherung (der Gegner ist ebenfalls unsicher).

**Verfolgungsfehler.** $\sqrt{\tfrac1n\sum_i\bigl((\theta_i - \bar\theta) - (s_i - \bar s)\bigr)^2}$, gemittelt über die Runden ab 30. **Log-Verlust** vor der Partie:
$-\log E_{ij}$ für den Sieger.

**Bradley-Terry über alle Partien.** Maximum Likelihood mit Ridge $\lambda_0 = 12/(\delta^2(n^2-1))$, Newton mit zentriertem Schritt; unabhängig von der Reihenfolge.

**Zufall.** Irrfahrt mit Normalverteilung per Box-Muller aus der Gleichverteilten des Generators (SplitMix64); Paarung durch eine Zufallspermutation je Runde.

**Literatur.**
- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present.* Arco Publishing, New York (2. Auflage 1986; nach Katalogeinträgen, das Buch wurde nicht
  eingesehen).
- Im Schach sind K = 40 für Neulinge, K = 20 und K = 10 für Spieler ab 2400 gebräuchlich (nach einem Blog zu den FIDE-Regeln ab 2024; das FIDE-Handbuch wurde nicht
  eingesehen).
- Glickman, M. E. (1999): Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society Series C (Applied
  Statistics)* 48 (3), 377–394, doi:10.1111/1467-9876.00159 (nach dem Abstract: ein nichtiteratives Verfahren, das den Elo-Algorithmus um die Unsicherheit der
  Schätzung erweitert; das Original wurde nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
