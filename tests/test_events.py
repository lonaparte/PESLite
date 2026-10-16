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


RUNS = []


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
    def apply(event, system, t):
        RUNS.append((event.label, system, t))


def test_builtin_and_custom_event_types_are_registered_and_typed():
    assert {"connect", "disconnect", "set", "test_mark"} <= set(EVENT_TYPES)
    assert peslite.register_event_type is register_event_type
    p = _gfl().replace(**_event("mark", type="test_mark", target="vsc", t=0.3, label="seen"))
    assert p.events["mark"].label == "seen"
    assert [event.t for event in p.unit("vsc").events] == [0.2, 0.3]
    marker = object()
    RUNS.clear()
    EVENT_TYPES["test_mark"].apply(p.events["mark"], marker, 0.3)
    assert RUNS == [("seen", marker, 0.3)]

    with pytest.raises(ConfigError, match="events.mark.target: unknown name"):
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

    with pytest.raises(TypeError, match="it needs apply"):
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
    schedule = switching_schedule(p.events)["vsc"]
    assert schedule == [(0.2, True), (2.0, False), (3.0, True)]
    assert not connected_at(schedule, 0.1) and connected_at(schedule, 1.0)
    scenario = Scenario(events_for(p.events, "vsc"))
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
    assert [change.event for change in p.changes] == ["enable", "retune"]
    assert [change.params.unit("vsc").protection.frequency.limit for change in p.changes] == [0.5, 0.8]
    assert p.changes[0].touches("units.vsc.") == [
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
    assert p.changes[0].params.unit("vsc").ctrl.references.p_ref_pu == pytest.approx(0.5)

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
    steps = [(change.t, change.params.sources["grid"]) for change in p.changes]
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
