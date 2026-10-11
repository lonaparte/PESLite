"""Import the package from ``src/`` and provide shared issue #2 cases."""

import copy
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT / "src"))

import peslite  # noqa: E402

QUIET = {"simulation.solver.linearisations": 0, "simulation.progress.enable": 0}

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
            "pll": {"type": "srf_pll", "period": 5e-5, "kp_pu": 20.0, "ki_pu": 1200.0},
            "cc": {"type": "dq_current_pi", "period": 5e-5, "bandwidth": 200.0},
            "dvc": {"type": "dc_voltage_pi", "period": 5e-5, "kp_pu": 0.8,
                    "ki_pu": 90.0, "limit_pu": 1.15, "id0_export_pu": 0.9},
        }},
    }},
    "simulation": {"t_end": 0.004, "solver": {"dt": 12.5e-6, "linearisations": 0}},
}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
