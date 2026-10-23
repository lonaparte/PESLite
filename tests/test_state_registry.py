"""State paths and state-table columns are fixed when a simulation is assembled."""

import pytest

from peslite.solver import ConfigError, StateRegistry


class _States:
    def __init__(self, **values):
        self.values = values
        self.loads = []

    def get_state(self):
        return dict(self.values)

    def set_state(self, values):
        self.loads.append(dict(values))
        self.values.update(values)


def test_registry_fixes_names_columns_and_owners_at_construction():
    plant = _States(i=1 + 2j)
    ctrl = _States(enabled=True, integral=3.0)
    states = StateRegistry({"vsc": plant, "vsc.ctrl": ctrl})

    assert states.names == ("vsc.i", "vsc.ctrl.enabled", "vsc.ctrl.integral")
    assert states.columns == (
        "vsc.i.re", "vsc.i.im", "vsc.ctrl.enabled", "vsc.ctrl.integral"
    )
    assert states.boolean_names == {"vsc.ctrl.enabled"}

    plant.values["i"] = 4 + 5j
    assert states.read_flat() == {
        "vsc.i.re": 4.0,
        "vsc.i.im": 5.0,
        "vsc.ctrl.enabled": 1.0,
        "vsc.ctrl.integral": 3.0,
    }
    states.load({"vsc.ctrl.integral": 7.0}, ["vsc.ctrl"])
    assert ctrl.loads == [{"integral": 7.0}]
    assert plant.loads == []


def test_registry_rejects_duplicate_paths_and_flat_columns_during_construction():
    with pytest.raises(ConfigError, match="state name 'vsc.x'.*both"):
        StateRegistry({"vsc": _States(x=1.0), "": _States(**{"vsc.x": 2.0})})

    with pytest.raises(ConfigError, match="state-table column 'x.re'.*both"):
        StateRegistry({"": _States(x=1 + 0j, **{"x.re": 2.0})})

    with pytest.raises(ConfigError, match="state prefix 'vsc'.*both"):
        StateRegistry([("vsc", _States(a=1.0)), ("vsc", _States(b=2.0))])


@pytest.mark.parametrize("prefix", ["", "nested.unit"])
def test_periodic_projection_uses_the_existing_snapshot_without_changing_restart_columns(prefix):
    class Logical(_States):
        def periodic_state(self, values):
            return {"i": values["i"], "history.x": values["history.x"]}

    states = StateRegistry({prefix: Logical(i=1 + 2j, **{"history.x": 3.0, "index": 1})})
    row = states.read_flat()
    index = f"{prefix}.index" if prefix else "index"
    expected = {key: value for key, value in row.items() if key != index}
    assert states.periodic_state(row) == expected
    assert index in row
    assert index in states.columns
    assert states.read_flat() == row
