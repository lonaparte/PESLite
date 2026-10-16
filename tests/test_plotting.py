"""The optional plotting add-on reads completed CSV output and writes vector PDF figures."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.addons import plot_csv, plot_result
from peslite.addons.functions import plotting
from peslite.addons.functions.plotting import _time_scale
from peslite.solver.simulation import main


pytest.importorskip("matplotlib")


def _table(path: Path, header: str, rows: str) -> Path:
    path.write_text(header + "\n" + rows, encoding="utf-8")
    return path


def test_plot_csv_writes_an_ieee_width_vector_pdf(tmp_path):
    source = _table(
        tmp_path / "renamed-state-output.csv",
        "t,vsc.dclink.u_C,vsc.ctrl.SRFPLL.theta",
        "0,1500,0\n0.001,1501,0.01\n0.002,1499,0.02\n",
    )

    figure = plot_csv(
        source,
        ["vsc.dclink.u_C", "vsc.ctrl.SRFPLL.theta"],
        ylabel=r"$x$ (pu)",
        labels=[r"$u_{dc}$", r"$\theta$"],
        latex=False,
    )

    assert figure == tmp_path / "fig_renamed-state-output.pdf"
    assert figure.read_bytes().startswith(b"%PDF")
    assert figure.stat().st_size > 1000


def test_plot_result_finds_the_actual_csv_path_in_result_files(tmp_path):
    state_file = _table(tmp_path / "custom-name.csv", "t,state", "0,0\n1e-6,1\n")
    _table(tmp_path / "another.csv", "t,signal", "0,2\n1e-6,3\n")
    result = SimpleNamespace(files=[tmp_path / "summary.json", state_file])

    figure = plot_result(result, "state", latex=False)

    assert figure == tmp_path / "fig_custom-name.pdf"
    assert figure.is_file()


def test_plot_result_requires_waveforms_from_one_csv(tmp_path):
    first = _table(tmp_path / "states.csv", "t,state", "0,0\n1,1\n")
    second = _table(tmp_path / "plant.csv", "t,signal", "0,2\n1,3\n")

    with pytest.raises(ValueError, match="must come from one result CSV"):
        plot_result(SimpleNamespace(files=[first, second]), ["state", "signal"], latex=False)


@pytest.mark.parametrize(
    ("times", "expected_unit", "expected"),
    [
        ([0.0, 2.0], "s", [0.0, 2.0]),
        ([0.0, 2e-3], "ms", [0.0, 2.0]),
        ([0.0, 2e-6], "us", [0.0, 2.0]),
    ],
)
def test_time_axis_uses_engineering_units(times, expected_unit, expected):
    scaled, unit, _factor = _time_scale(np.asarray(times), "auto")

    assert unit == expected_unit
    assert scaled == pytest.approx(expected)


def test_cli_runs_then_plots_named_waveforms(examples, tmp_path, monkeypatch):
    monkeypatch.setattr(plotting.shutil, "which", lambda _program: None)
    out = tmp_path / "named-output"

    status = main([
        str(examples / "gfl-example.pes"),
        "--set", "simulation.t_end=0.0001",
        "--set", "simulation.solver.linearisations=0",
        "--out", str(out),
        "--plot", "vsc.dclink.u_C", "vsc.ctrl.SRFPLL.theta",
        "--plot-label", r"$u_{dc}$",
        "--plot-label", r"$\theta$",
    ])

    assert status == 0
    assert (out / "fig_states.pdf").read_bytes().startswith(b"%PDF")
