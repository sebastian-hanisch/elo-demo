"""PDF-Export der aktuellen Konfiguration und der Wertungen (fpdf2, Kernschrift: nur Zeichen aus Latin-1, daher „Tau“ statt τ und „theta“ statt θ)."""

from fpdf import FPDF

import elo_evaluation as ev
import elo_rating as E


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="), ("→", "->"), ("**", ""),
                     ("€", "EUR"), ("−", "-")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s, m = a.settings, a.matches
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Elo: die Stärke Partie für Partie"))
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    extra = f", Anlauf-K0: {s.k0}" if s.provisional else ""
    extra += f", Neuling ab Runde {s.join_round} (Abstand {ev.de(s.gap, 1)})" if s.newcomer else ""
    pdf.multi_cell(0, 5.5, _latin(f"Spieler: {s.n}, Runden: {s.rounds}, Drift sigma: {ev.de(s.sigma)}, K: {s.k}{extra}, Seed: {s.seed}."))
    pdf.ln(2)
    level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text))
    pdf.ln(3)
    headers = ("Spieler", "Partien", "Elo-Wertung", "wahre Staerke", "Bradley-Terry")
    widths = (28, 24, 34, 40, 40)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    final_pts = E.to_points(a.run.final - a.run.final.mean())
    true_pts = E.to_points(a.truth_end)
    bt_pts = E.to_points(a.bt)
    for p in sorted(range(s.n), key=lambda q: -a.run.final[q]):
        row = (m.names[p], str(int(a.games[p])), f"{final_pts[p]:.0f}", f"{true_pts[p]:.0f}", f"{bt_pts[p]:.0f}")
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Anfangsstärken aus dem Generator der Ranking-Linie, danach Irrfahrt der Stärken); die Stärke ist eine Konstruktion der "
        "Simulation. Elo-Punkte: 400 Punkte entsprechen dem Faktor 10 der Gewinnchancen. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."))
    return bytes(pdf.output())
