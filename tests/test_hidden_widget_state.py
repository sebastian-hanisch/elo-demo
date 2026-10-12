"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück. Die Obergrenze der Neuling-Runde hängt von den Runden ab."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 600


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def widget_keys(at):
    return {s.key for s in at.slider}


def test_widgets_are_only_drawn_where_they_act():
    at = fresh()
    assert not {"k0_slider", "join_slider", "gap_slider"} & widget_keys(at)
    at.checkbox(key="provisional_toggle").check().run(timeout=TIMEOUT)
    assert "k0_slider" in widget_keys(at) and "join_slider" not in widget_keys(at)
    at.checkbox(key="newcomer_toggle").check().run(timeout=TIMEOUT)
    assert {"join_slider", "gap_slider"} <= widget_keys(at)
    assert not at.exception


def test_hidden_values_survive_and_are_not_written_into_hidden_widgets():
    at = fresh()
    at.checkbox(key="provisional_toggle").check().run(timeout=TIMEOUT)
    at.slider(key="k0_slider").set_value(200).run(timeout=TIMEOUT)
    at.checkbox(key="provisional_toggle").uncheck().run(timeout=TIMEOUT)
    assert "k0_slider" not in at.session_state
    at.checkbox(key="provisional_toggle").check().run(timeout=TIMEOUT)
    assert at.slider(key="k0_slider").value == 200
    assert not at.exception


def test_join_round_follows_a_shrinking_number_of_rounds():
    at = fresh()
    at.checkbox(key="newcomer_toggle").check().run(timeout=TIMEOUT)
    at.slider(key="join_slider").set_value(100).run(timeout=TIMEOUT)
    at.slider(key="rounds_slider").set_value(20).run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="join_slider").value <= 18


def test_preset_sets_values_and_hidden_widgets_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Neuling im Feld"]].click().run(timeout=TIMEOUT)
    assert at.checkbox(key="newcomer_toggle").value is True and at.checkbox(key="provisional_toggle").value is True
    assert at.slider(key="join_slider").value == 60 and at.slider(key="rounds_slider").value == 100 and at.slider(key="sigma_slider").value == 0.0
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["K zu groß"]].click().run(timeout=TIMEOUT)
    assert at.slider(key="k_slider").value == 160 and at.checkbox(key="newcomer_toggle").value is False
    assert "join_slider" not in at.session_state
    assert not at.exception


def test_reordering_the_games_shows_a_second_table():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["🔀 Partien neu ordnen"]].click().run(timeout=TIMEOUT)
    assert at.session_state["order_seed"] == 1 and not at.exception
