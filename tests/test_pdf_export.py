"""PDF-Export: erzeugt für jede Konfiguration gültige PDF-Bytes; der Latin-1-Filter ersetzt alle Zeichen außerhalb der Kernschrift."""

import elo_evaluation as ev
import elo_pdf_export as X
from elo_scenario import Settings


def test_pdf_is_built_for_drift_static_provisional_and_newcomer():
    for s in (Settings(), Settings(sigma=0.0, k=5), Settings(n=16, rounds=40, provisional=True, k0=200, seed=4),
              Settings(n=4, rounds=12, newcomer=True, join=5, gap=-2.0, provisional=True, seed=7)):
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("τ ≈ 0,5; σ → θ − K **fett** €")
    assert out == "Tau ca. 0,5; sigma -> theta - K fett EUR"
    assert out.encode("latin-1")
