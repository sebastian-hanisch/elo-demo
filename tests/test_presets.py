"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt, was seine Beschreibung verspricht."""

import dataclasses

import elo_constants as C
import elo_evaluation as ev
import elo_presets as P
from elo_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.DELTA_MIN <= p["delta"] <= C.DELTA_MAX and C.ROUNDS_MIN <= p["rounds"] <= C.ROUNDS_MAX, name
        assert C.SIGMA_MIN <= p["sigma"] <= C.SIGMA_MAX and C.K_MIN <= p["k"] <= C.K_MAX and C.K0_MIN <= p["k0"] <= C.K0_MAX, name
        assert C.JOIN_MIN <= p["join"] <= p["rounds"] - 2 and C.GAP_MIN <= p["gap"] <= C.GAP_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert isinstance(p["provisional"], bool) and isinstance(p["newcomer"], bool), name


def test_first_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Wandernde Stärken"]) == Settings()


def test_setting_specs_cover_every_preset_key_and_settings_fields_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(Settings)}
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_kept_keys_are_exactly_the_hideable_widgets():
    assert set(P.KEPT_KEYS) == {"k0_slider", "join_slider", "gap_slider"}


def test_join_round_is_clamped_into_the_valid_range():
    assert Settings(newcomer=True, join=500, rounds=50).join_round == 48
    assert Settings(newcomer=True, join=0, rounds=50).join_round == 2
    assert Settings(newcomer=False, join=30, rounds=50).join_round == 0
    assert Settings(provisional=True, k0=90).k_start == 90.0 and Settings(provisional=False).k_start is None


def _mean(preset, attr, seeds=range(100000, 100040)):
    vals = [attr(ev.analyse(Settings(**{**C.PRESETS[preset], "seed": s}))) for s in seeds]
    return sum(vals) / len(vals)


def test_each_preset_tells_its_story():
    drift = "Wandernde Stärken"
    assert _mean(drift, lambda a: a.rmse["Elo"]) < _mean(drift, lambda a: a.rmse["Bradley-Terry"])        # Elo folgt der Drift
    static = "Konstante Stärken"
    assert _mean(static, lambda a: a.rmse["Bradley-Terry"]) < _mean(static, lambda a: a.rmse["Elo"])      # ohne Drift gewinnt der Schätzer über alle Partien
    big = "K zu groß"
    assert _mean(big, lambda a: a.run.tracking_error) > _mean(drift, lambda a: a.run.tracking_error) + 0.2
    new = "Neuling im Feld"
    assert _mean(new, lambda a: float(len(a.run.newcomer_errors))) > 20
    assert Settings(**C.PRESETS[new]).provisional and Settings(**C.PRESETS[new]).newcomer
