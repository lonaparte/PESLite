"""Terminal progress and the quantities it watches."""

import io
import re
from contextlib import redirect_stdout

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError


def _progress(*args):
    output = io.StringIO()
    with redirect_stdout(output):
        assert peslite.main([
            "gfl-example", "--pwm-averaging",
            "--set", "simulation.t_end=0.01",
            "--set", "simulation.solver.linearisations=0",
            *args,
        ]) == 0
    return [line for line in output.getvalue().splitlines()
            if line.startswith("Simulating gfl-example ")]


def test_watch_prints_quantities_on_each_progress_update(tmp_path):
    lines = _progress(
        "--progress",
        "--watch", "vsc.vdc_pu",
        "--watch", "vsc.i_c,vsc.u_dc",
        "--out", str(tmp_path),
    )
    assert 2 <= len(lines) <= 101
    assert "  0.0%" in lines[0]
    assert "100.0%" in lines[-1]
    assert "vsc.vdc_pu" not in lines[0]
    for line in lines[1:]:
        values = dict(re.findall(r"   (\S+) = (\S+)", line))
        assert list(values) == ["vsc.vdc_pu", "vsc.i_c", "vsc.u_dc"]
        assert float(values["vsc.u_dc"]) == pytest.approx(
            1500 * float(values["vsc.vdc_pu"]), rel=0.02
        )
    progress = peslite.load(tmp_path / "simulation.pes").simulation.progress
    assert progress.enable is True
    assert not hasattr(progress, "period")
    assert progress.watch == ["vsc.vdc_pu", "vsc.i_c", "vsc.u_dc"]


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_interactive_cli_uses_one_progress_bar_unless_disabled(tmp_path):
    output = _Terminal()
    with redirect_stdout(output):
        assert peslite.main([
            "gfl-example", "--pwm-averaging",
            "--set", "simulation.t_end=0.002",
            "--set", "simulation.solver.linearisations=0",
            "--out", str(tmp_path / "shown"),
        ]) == 0
    assert "\rSimulating gfl-example [" in output.getvalue()
    assert "100.0%" in output.getvalue()

    output = _Terminal()
    with redirect_stdout(output):
        assert peslite.main([
            "gfl-example", "--pwm-averaging", "--no-progress",
            "--set", "simulation.t_end=0.002",
            "--set", "simulation.solver.linearisations=0",
            "--out", str(tmp_path / "hidden"),
        ]) == 0
    assert "Simulating gfl-example" not in output.getvalue()


def test_watch_values_match_result_columns_and_states(gfl, monkeypatch, tmp_path):
    params = gfl(**{
        "simulation.output.signals": 1,
        "simulation.output.period": 1e-3,
        "simulation.output.precision": 17,
        "simulation.t_end": 0.002,
        "simulation.progress.enable": 1,
        "simulation.progress.watch": ["pcc.u_a"],
    })
    seen = {}
    values_at = peslite.Simulation.watch_values
    monkeypatch.setattr(
        peslite.Simulation,
        "watch_values",
        lambda self, t, logs: seen.setdefault(t, values_at(self, t, logs)),
    )
    with redirect_stdout(io.StringIO()):
        result = peslite.Simulation(params).run(out_dir=tmp_path)
    t, values = min(
        (t, values) for t, values in seen.items()
        if np.any(np.isclose(result.t, t, rtol=0.0, atol=1e-12))
    )
    index = int(np.argmin(abs(result.t - t)))
    assert result.t[index] == pytest.approx(t)
    columns = result.columns()
    for name in ("pcc.u_a", "vsc.u_dc", "vsc.id_pu", "grid.angle"):
        assert values[name] == pytest.approx(columns[name][index]), name
    state = complex(result.states["pcc.u_C.re"][index],
                    result.states["pcc.u_C.im"][index])
    assert values["pcc.u_C"] == pytest.approx(abs(state))
    assert values["pcc.u_C.re"] == pytest.approx(state.real)
    assert values["vsc.u_dc"] == result.plant["vsc.u_dc"][index]


def test_watch_uses_entity_first_state_names(gfl, tmp_path):
    params = gfl(**{"simulation.progress.enable": 1})
    simulation = peslite.Simulation(params)
    with redirect_stdout(io.StringIO()):
        simulation.run(out_dir=tmp_path)
    values = simulation.watch_values(0.002, {})
    assert "vsc.i_c" in values
    assert "vsc.dclink.u_C" in values
    assert not any(name.startswith("plant.") for name in values)


def test_watching_does_not_change_the_run(gfl, tmp_path):
    base = gfl()
    with redirect_stdout(io.StringIO()):
        first = peslite.Simulation(base).run(out_dir=tmp_path / "plain")
        second = peslite.Simulation(base.replace(**{
            "simulation.progress.enable": 1,
            "simulation.progress.watch": ["vsc.vdc_pu", "pcc.u_C"]
        })).run(out_dir=tmp_path / "watched")
    assert first.n_rhs == second.n_rhs
    for name in first.states:
        assert np.array_equal(first.states[name], second.states[name], equal_nan=True), name


def test_progress_period_is_not_a_configuration_field(gfl):
    with pytest.raises(ConfigError, match="period"):
        gfl(**{"simulation.progress.period": 1e-3})


def test_watch_needs_progress_lines(gfl, capsys, tmp_path):
    with pytest.raises(SystemExit):
        peslite.main(["gfl-example", "--watch", "vsc.vdc_pu"])
    assert "add --progress" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # the same rule for the row selection of a restart
        peslite.main(["inspect", "gfl-example", "--initial-time", "0.5", "--resolved"])
    assert "add --initial FILE" in capsys.readouterr().err

    params = gfl(**{"simulation.progress.watch": ["vsc.vdc_pu"]})
    with redirect_stdout(io.StringIO()) as output:
        peslite.Simulation(params).run(out_dir=tmp_path)
    assert output.getvalue() == ""

    with pytest.raises(ConfigError, match="expected a list of quantity names"):
        gfl(**{"simulation.progress.enable": 1,
               "simulation.progress.watch": "vsc.vdc_pu"})


def test_unknown_watched_name_reports_known_names(gfl, tmp_path):
    params = gfl(**{
        "simulation.progress.enable": 1,
        "simulation.progress.watch": ["vsc.vdc"],
    })
    message = r"unknown name\(s\) \['vsc.vdc'\]; known: .*'vsc.vdc_pu'"
    with pytest.raises(ConfigError, match=message):
        with redirect_stdout(io.StringIO()):
            peslite.Simulation(params).run(out_dir=tmp_path)
