"""Import the package from ``src/`` and provide shared test cases."""

import copy
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT / "src"))

import peslite  # noqa: E402

QUIET = {"simulation.solver.linearisations": 0, "simulation.progress.enable": 0}
BOUNDED_TEST_PROCESS_TIMEOUT = 120
SUITES = ("fast", "long", "ui")


def pytest_addoption(parser):
    parser.addoption(
        "--run-long",
        action="store_true",
        help="run only long numerical, compiler and external-reference tests",
    )
    parser.addoption(
        "--run-ui",
        action="store_true",
        help="run only graphical-interface tests",
    )
    parser.addoption(
        "--run-full",
        action="store_true",
        help="run the fast, long and UI test suites",
    )


def _selected_suite(config):
    requested = [
        suite for suite in ("long", "ui", "full")
        if config.getoption(f"--run-{suite}")
    ]
    if len(requested) > 1:
        raise pytest.UsageError("choose only one of --run-long, --run-ui and --run-full")
    return None if requested == ["full"] else (requested[0] if requested else "fast")


def pytest_configure(config):
    """Reject conflicting suite selections before test collection starts."""
    _selected_suite(config)


def pytest_collection_modifyitems(config, items):
    """Assign one suite to every test, select it and leave long tests until last."""
    categories = {}
    for item in items:
        marked = [name for name in SUITES if item.get_closest_marker(name) is not None]
        if len(marked) > 1:
            raise pytest.UsageError(
                f"{item.nodeid} belongs to multiple test suites: {', '.join(marked)}"
            )
        category = marked[0] if marked else "fast"
        if not marked:
            item.add_marker("fast")
        categories[item] = category

    selected_suite = _selected_suite(config)
    if selected_suite is not None:
        selected = [item for item in items if categories[item] == selected_suite]
        deselected = [item for item in items if categories[item] != selected_suite]
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
    parallel_workers = getattr(config.option, "numprocesses", None)
    serial = [item for item in items if item.get_closest_marker("serial") is not None]
    if serial and parallel_workers not in (None, 0, "0", 1, "1"):
        raise pytest.UsageError("tests marked serial require a non-parallel pytest run")
    order = {"fast": 0, "ui": 1, "long": 2}
    items.sort(key=lambda item: (
        order[categories[item]],
        item.get_closest_marker("static") is None,
        item.get_closest_marker("serial") is not None,
    ))


@pytest.fixture
def exhaustive_loop():
    """The same map without optional optimization declarations; no alternative dynamics."""
    def reference(loop):
        return SimpleNamespace(period=loop.period, coordinates=loop.coordinates,
                               to_frame=loop.to_frame, from_frame=loop.from_frame,
                               advance=loop.advance)
    return reference

CASE = {
    "base": {"s_base": 2.0e6, "v_ll_rms": 690.0, "f0": 50.0},
    "buses": {"pcc": {"c_pu": 0.02, "r_d_pu": 0.5}},
    "sources": {"grid": {"bus": "pcc", "x_pu": 0.4, "r_pu": 0.04}},
    "units": {"vsc": {
        "type": "vsc",
        "bus": "pcc",
        "ac_filter": {"l_f_pu": 0.2, "r_f_pu": 0.01},
        "dclink": {"vdc_ref": 1500.0,
                   "capacitor": {"c_pu": 35.0, "r_esr_pu": 0.001},
                   "source": {"type": "current", "i_pu": 0.9}},
        "pwm": {"f_sw": 20000.0, "modulation_limit": 0.95},
        "bridge": {"model": "pwm_averaging"},
        "ctrl": {"type": "gfl", "loops": {
            "SRFPLL": {"period": 5e-5, "kp_pu": 20.0, "ki_pu": 1200.0},
            "CurrentLoop": {"period": 5e-5, "bandwidth": 200.0},
            "ThreePhaseModulation": {"period": 5e-5},
            "DCVoltageControl": {"period": 5e-5, "kp": 0.8, "ki": 90.0,
                                 "minimum": 0.0, "maximum": 1.15,
                                 "feedforward": 0.9, "output_polarity": -1.0},
        }},
    }},
    "simulation": {"t_end": 0.004, "solver": {"dt": 12.5e-6, "linearisations": 0}},
}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(autouse=True)
def _bound_heavy_test_processes(request):
    """Keep subprocesses launched by long and UI tests time-bounded."""
    if all(request.node.get_closest_marker(name) is None for name in ("long", "ui")):
        yield
        return

    monkeypatch = request.getfixturevalue("monkeypatch")
    run = subprocess.run

    def bounded_run(*args, **kwargs):
        kwargs.setdefault("timeout", BOUNDED_TEST_PROCESS_TIMEOUT)
        return run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", bounded_run)
    yield


@pytest.fixture
def examples():
    return EXAMPLES


@pytest.fixture
def case():
    return lambda: copy.deepcopy(CASE)


@pytest.fixture
def gfl():
    """A short averaged run based on the annotated bundled GFL example."""
    def make(**over):
        return peslite.load(
            EXAMPLES / "gfl-example.pes",
            **{**QUIET, "simulation.t_end": 0.004, "units.vsc.bridge.model": "pwm_averaging",
               "events.connect_vsc.t": 0.0, **over},
        )
    return make
