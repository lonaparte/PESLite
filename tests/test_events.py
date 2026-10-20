"""Unified simulation-file events: registration, scenarios and set validation."""

import math
from dataclasses import dataclass

import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly import Event, Scenario, register_event_type
from peslite.assembly.events import EVENT_TYPES, SourceScenario, connected_at, events_for, switching_schedule
from peslite.assembly.params import ConfigError, dumps, from_dict, read_tree
from peslite.control import Startup


def _gfl():
    return peslite.load(EXAMPLES / "gfl-example.pes")


def _event(name, **entry):
    return {f"events.{name}": entry}


def _set(name, t, **paths):
    return _event(name, type="set", t=t, set=paths)


@register_event_type
class _Mark(Event):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_mark"
        target: str
        t: float
        label: str = "test"
        level_pu: float = 0.0

        _quantities = {"level_pu": "current"}

    type = "test_mark"

    @staticmethod
    def variables(event, params):
        return {f"units.{event.target}.ctrl.references.iq_ref_pu": event.level_pu}


def test_builtin_and_custom_event_types_are_registered_and_typed():
    assert {"connect", "disconnect", "set", "test_mark"} <= set(EVENT_TYPES)
    assert peslite.register_event_type is register_event_type
    p = _gfl().replace(**_event("mark", type="test_mark", target="vsc", t=0.3, label="seen"))
    assert p.events["mark"].label == "seen"
    assert [event.t for event in p.unit("vsc").events] == [0.2, 0.3]
    assert p.changes[-1].values == {"units.vsc.ctrl.references.iq_ref_pu": 0.0}

    with pytest.raises(ConfigError, match="units.nope.*unknown writable variable"):
        _gfl().replace(**_event("mark", type="test_mark", target="nope", t=0.3))
    scaled = _gfl().replace(**_event("mark", type="test_mark", target="vsc", t=0.3,
                                     level=_gfl().unit("vsc").base.i_phase_peak))
    assert scaled.events["mark"].level_pu == pytest.approx(1.0)
    with pytest.raises(ValueError, match="already registered"):
        register_event_type(_Mark)

    class Silent(Event):
        @dataclass(frozen=True, kw_only=True)
        class Params:
            type: str = "test_silent"
            t: float

        type = "test_silent"

    with pytest.raises(TypeError, match="it needs variables"):
        register_event_type(Silent)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"type": "fault", "t": 0.3}, "unknown event type 'fault'"),
        ({"type": "connect", "t": 0.3}, "events.bad.target: required value missing"),
        ({"type": "connect", "target": "vsc", "t": 0.3, "ramp": -0.1},
         "ramp must be finite and >= 0"),
        ({"type": "disconnect", "target": "vsc", "t": math.inf},
         "events.bad.t must be finite"),
    ],
)
def test_event_entries_reject_unknown_missing_and_nonfinite_values(entry, message):
    with pytest.raises(ConfigError, match=message):
        _gfl().replace(**_event("bad", **entry))


def test_cross_section_names_are_scoped_and_ambiguous_event_targets_are_rejected():
    tree = read_tree(EXAMPLES / "gfl-example.pes")
    tree["sources"]["vsc"] = tree["sources"].pop("grid")
    tree["events"] = {}
    params = from_dict(tree)
    system = peslite.System(params)

    assert "vsc" in params.sources and "vsc" in params.units
    with pytest.raises(ValueError, match="ambiguous.*sources.*units"):
        system.switch("vsc", False, 0.0)

    tree = read_tree(EXAMPLES / "gfl-example.pes")
    tree["sources"]["vsc"] = tree["sources"].pop("grid")
    with pytest.raises(ConfigError, match="target.*ambiguous.*sources.*units"):
        from_dict(tree)


def test_switching_sequences_are_checked():

    with pytest.raises(ConfigError, match="already connected"):
        _gfl().replace(**_event("again", type="connect", target="vsc", t=0.3))
    with pytest.raises(ConfigError, match="also switched at t = 0.2"):
        _gfl().replace(**_event("off", type="disconnect", target="vsc", t=0.2))
    with pytest.raises(ConfigError, match="is a bus, which has no breaker"):
        _gfl().replace(**_event("off", type="disconnect", target="pcc", t=0.3))


def test_switching_schedule_and_scenario_connection_ramps():
    p = _gfl().replace(
        **_event("off", type="disconnect", target="vsc", t=2.0),
        **_event("on", type="connect", target="vsc", t=3.0, ramp=2.0),
    )
    schedule = switching_schedule(p.changes)["units.vsc"]
    assert schedule == [(0.2, True), (2.0, False), (3.0, True)]
    assert not connected_at(schedule, 0.1) and connected_at(schedule, 1.0)
    scenario = Scenario(events_for(p.changes, "units.vsc"))
    assert not scenario.connected(0.1) and scenario.connected(0.7)
    assert not scenario.connected(2.5) and scenario.connected(4.0)
    assert scenario.since(4.0) == (3.0, 2.0)

    startup = Startup(0.5, run=False)
    startup.command(True, 2.0)
    for _ in range(3):
        startup.advance()
    assert 0.0 < startup.value < 1.0 and not startup.complete
    for _ in range(3):
        startup.advance()
    assert startup.value == 1.0 and startup.complete

    always = Scenario()
    assert always.connected(0.0) and always.since(0.0) == (-math.inf, 0.0)


def test_set_events_are_applied_and_validated_in_order():
    p = _gfl().replace(
        **_set("enable", 1.0, **{"units.vsc.protection.frequency.enable": 1,
                                "units.vsc.protection.frequency.limit": 0.5}),
        **_set("retune", 2.0, **{"units.vsc.protection.frequency.limit": 0.8}),
    )
    changes = [change for change in p.changes if not change.connections]
    assert [change.event for change in changes] == ["enable", "retune"]
    assert [change.params.unit("vsc").protection.frequency.limit for change in changes] == [0.5, 0.8]
    assert changes[0].touches("units.vsc.") == [
        "protection.frequency.enable", "protection.frequency.limit"]

    with pytest.raises(ConfigError, match="required when enable is 1"):
        _gfl().replace(**_set("bad", 1.0, **{"units.vsc.protection.frequency.enable": 1}))
    with pytest.raises(ConfigError, match="simulation.t_end: cannot change during a run"):
        _gfl().replace(**_set("bad", 1.0, **{"simulation.t_end": 4.0}))
    with pytest.raises(ConfigError, match="nothing to change"):
        _gfl().replace(**_set("bad", 1.0))


def test_set_paths_use_si_pu_rules_and_detect_same_time_conflicts():
    p = _gfl().replace(**_set("power", 1.0, **{"units.vsc.ctrl.references.p_ref": 1e6}))
    assert p.events["power"].set == {"units.vsc.ctrl.references.p_ref": 1e6}
    assert p.changes[-1].params.unit("vsc").ctrl.references.p_ref_pu == pytest.approx(0.5)

    with pytest.raises(ConfigError, match="is also set at t = 1.0"):
        _gfl().replace(
            **_set("a", 1.0, **{"units.vsc.ctrl.references.p_ref_pu": 0.5}),
            **_set("b", 1.0, **{"units.vsc.ctrl.references.p_ref_pu": 0.8}),
        )
    with pytest.raises(ConfigError, match="pwm.f_sw cannot change during a run"):
        _gfl().replace(**_set("bad", 1.0, **{"units.vsc.pwm.f_sw": 10_000.0}))


def test_source_scenario_preserves_frequency_phase_and_voltage_changes():
    p = _gfl().replace(
        **_set("frequency", 1.0, **{"sources.grid.f": 51.0}),
        **_set("angle", 2.0, **{"sources.grid.angle": 0.1}),
        **_set("voltage", 3.0, **{"sources.grid.v_pu": 0.9}),
    )
    steps = [(change.t, change.params.sources["grid"]) for change in p.changes if not change.connections]
    scenario = SourceScenario(p.sources["grid"], p.base.f0, steps)
    assert scenario.angle(0.5) == 0.0
    assert scenario.angle(1.5) == pytest.approx(math.pi)
    assert scenario.angle(2.5) == pytest.approx(3 * math.pi + 0.1)
    assert scenario.magnitude(2.9) == pytest.approx(p.base.v_phase_peak)
    assert scenario.magnitude(3.1) == pytest.approx(0.9 * p.base.v_phase_peak)

    initial = _gfl().replace(**{"sources.grid.f": 50.5, "sources.grid.angle": 0.2})
    scenario = SourceScenario(initial.sources["grid"], initial.base.f0)
    assert scenario.angle(0.01) == pytest.approx(2 * math.pi * 0.5 * 0.01 + 0.2)
    assert yaml.safe_load(dumps(_gfl()))["sources"]["grid"]["f"] == 50.0


def test_events_round_trip_in_resolved_file_and_set_paths_are_replaceable():
    p = _gfl().replace(**_set("power", 1.0, **{"units.vsc.ctrl.references.p_ref_pu": 0.5}))
    q = p.replace(**{"events.power.set.units.vsc.ctrl.references.p_ref_pu": 0.7,
                     "events.power.t": 1.5})
    assert q.events["power"].set == {"units.vsc.ctrl.references.p_ref_pu": 0.7}
    assert q.events["power"].t == 1.5
    rebuilt = from_dict(yaml.safe_load(dumps(q)))
    assert rebuilt.events["power"].set == q.events["power"].set
    assert rebuilt.changes[0].paths == q.changes[0].paths


def test_connection_events_and_explicit_variable_writes_have_the_same_binding():
    original = _gfl()
    explicit = original.replace(**{"events": {
        "start": {"type": "set", "t": 0.2, "set": {"units.vsc.connected": True}},
    }})
    left, right = original.changes[0], explicit.changes[0]
    assert left.values == right.values == {"units.vsc.connected": True}
    assert left.connections == right.connections == ("units.vsc.connected",)
    assert left.params is None and right.params is None
    assert switching_schedule(original.changes) == switching_schedule(explicit.changes)
    scenario = Scenario(explicit.unit("vsc").events)
    assert not scenario.connected(0.1) and scenario.connected(0.3)


@pytest.mark.parametrize("value", [-1, 2, 0.5, "true", None])
def test_connection_variables_require_boolean_values(value):
    with pytest.raises(ConfigError, match="expected 1 .* or 0"):
        _gfl().replace(**_set("stop", 0.3, **{"units.vsc.connected": value}))


@pytest.mark.parametrize("value", [False, 0, 0.0])
def test_connection_variables_follow_the_existing_switch_syntax_and_round_trip(value):
    p = _gfl().replace(**_set("stop", 0.3, **{"units.vsc.connected": value}))
    assert p.changes[-1].values == {"units.vsc.connected": False}
    assert from_dict(yaml.safe_load(dumps(p))).changes[-1].values == p.changes[-1].values


def test_variable_binding_detects_mixed_syntax_connection_conflicts():
    with pytest.raises(ConfigError, match="also switched"):
        _gfl().replace(**_set("stop", 0.2, **{"units.vsc.connected": False}))
    with pytest.raises(ConfigError, match="already connected"):
        _gfl().replace(**_set("start", 0.3, **{"units.vsc.connected": True}))


def test_scoped_connection_variables_do_not_confuse_equal_names_in_different_sections():
    tree = read_tree(EXAMPLES / "gfl-example.pes")
    tree["sources"]["vsc"] = tree["sources"].pop("grid")
    tree["events"] = {"stop": {"type": "set", "t": 0.1, "set": {
        "sources.vsc.connected": False, "units.vsc.connected": False,
    }}}
    params = from_dict(tree)
    system = peslite.System(params)
    system.apply(params.changes[0])
    assert system.sources["vsc"].branch.breaker_open
    assert not system.units["vsc"].breaker


def test_custom_event_scope_is_determined_by_its_variable_paths():
    tree = read_tree(EXAMPLES / "gfl-example.pes")
    tree["sources"]["vsc"] = tree["sources"].pop("grid")
    tree["events"] = {"mark": {"type": "test_mark", "target": "vsc", "t": 0.1, "level_pu": 0.2}}
    p = from_dict(tree)
    assert p.changes[0].values == {"units.vsc.ctrl.references.iq_ref_pu": 0.2}


def test_custom_events_cannot_bypass_declared_variables_with_an_action_callback():
    class Callback(_Mark):
        type = "test_callback_event"

        @dataclass(frozen=True, kw_only=True)
        class Params:
            type: str = "test_callback_event"
            t: float

        @staticmethod
        def apply(event, system, t):
            pass

    with pytest.raises(TypeError, match="not an apply"):
        register_event_type(Callback)


@pytest.mark.parametrize("bindings", [{}, None, 1])
def test_custom_events_must_return_nonempty_variable_bindings(bindings, monkeypatch):
    monkeypatch.setattr(_Mark, "variables", staticmethod(lambda event, params: bindings))
    with pytest.raises(ConfigError, match="nonempty mapping"):
        _gfl().replace(**_event("mark", type="test_mark", target="vsc", t=0.3))


def test_variable_bindings_reject_duplicate_si_and_per_unit_spellings():
    with pytest.raises(ConfigError, match="same variable"):
        _gfl().replace(**_set("bad", 0.3, **{
            "sources.grid.v": 500.0, "sources.grid.v_pu": 0.9,
        }))


@pytest.mark.parametrize("path", ["units", "units.vsc.", "units.missing.connected"])
def test_invalid_variable_bindings_report_the_requested_path(path):
    with pytest.raises(ConfigError) as error:
        _gfl().replace(**_set("bad", 0.3, **{path: True}))
    assert path in str(error.value)
