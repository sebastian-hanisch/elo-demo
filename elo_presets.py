"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Ausblendbare Regler (Anlauf-K0 nur bei Anlauf, Neuling-Runde und Abstand nur bei Neuling) folgen dem
KEPT-Muster (DEMO-PLAYBOOK §3): der Wert eines nicht gezeichneten Reglers wird nie in sein Widget geschrieben, sondern in KEPT abgelegt und unmittelbar vor dem
Zeichnen per `seed_widget` eingesetzt. Die Obergrenze der Neuling-Runde hängt von den Runden ab: `clamp_kept` begrenzt den gemerkten Wert vorher."""

import random
from dataclasses import dataclass

import streamlit as st

import elo_constants as C


def _to_bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "ja", "yes")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "delta_slider": SettingSpec("d", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX),
    "rounds_slider": SettingSpec("r", int, C.DEFAULT_ROUNDS, C.ROUNDS_MIN, C.ROUNDS_MAX),
    "sigma_slider": SettingSpec("s", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "provisional_toggle": SettingSpec("prov", _to_bool, False, choices=(True, False)),
    "k0_slider": SettingSpec("k0", int, C.DEFAULT_K0, C.K0_MIN, C.K0_MAX),
    "newcomer_toggle": SettingSpec("new", _to_bool, False, choices=(True, False)),
    "join_slider": SettingSpec("join", int, C.DEFAULT_JOIN, C.JOIN_MIN, C.ROUNDS_MAX),
    "gap_slider": SettingSpec("gap", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "delta": "delta_slider", "rounds": "rounds_slider", "sigma": "sigma_slider", "k": "k_slider", "provisional": "provisional_toggle",
               "k0": "k0_slider", "newcomer": "newcomer_toggle", "join": "join_slider", "gap": "gap_slider", "seed": "seed_input"}
KEPT_KEYS = ("k0_slider", "join_slider", "gap_slider")


def _kept() -> dict:
    return st.session_state.setdefault("kept_values", {})


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key in KEPT_KEYS:
            _kept().setdefault(state_key, spec.default)
        elif state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines ausblendbaren Reglers: seinen zuletzt gewählten Wert einsetzen."""
    if state_key not in st.session_state:
        st.session_state[state_key] = _kept().get(state_key, SETTING_SPECS[state_key].default)


def clamp_kept(state_key, lo, hi):
    """Begrenzt den gemerkten (und, falls gezeichnet, den Widget-)Wert auf [lo, hi]; nötig, wenn die Grenzen von anderen Reglern abhängen."""
    kept = _kept()
    kept[state_key] = min(max(kept.get(state_key, SETTING_SPECS[state_key].default), lo), hi)
    if state_key in st.session_state:
        st.session_state[state_key] = min(max(st.session_state[state_key], lo), hi)


def remember(state_key, value):
    _kept()[state_key] = value


def current_value(state_key):
    """Aktueller Wert, auch wenn der Regler in diesem Lauf nicht gezeichnet wird (dann der gemerkte)."""
    if state_key in KEPT_KEYS:
        return st.session_state.get(state_key, _kept().get(state_key, SETTING_SPECS[state_key].default))
    return st.session_state.get(state_key, SETTING_SPECS[state_key].default)


def _store(state_key, value):
    if state_key in KEPT_KEYS:
        _kept()[state_key] = value
        st.session_state.pop(state_key, None)    # beim nächsten Zeichnen aus KEPT einsetzen
    else:
        st.session_state[state_key] = value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        _store(state_key, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value)) if isinstance(value, bool) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            _store(state_key, C.PRESETS[name][key])
    st.session_state["order_seed"] = 0


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
    st.session_state["order_seed"] = 0
