"""Pinned OpenPELS native-ngspice cases through the shared reference-test contract."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

import peslite
from peslite.assembly.syntax import from_dict
from reference_metrics import check_comparison, normalized_errors


OPENPELS_COMMIT = "afc618f1d803c57baf1b9f75876a5267743692ab"
_OPENPELS_ROOT = os.environ.get("OPENPELS_ROOT")
OPENPELS_ROOT = Path(_OPENPELS_ROOT) if _OPENPELS_ROOT else None
NGSPICE = shutil.which("ngspice")
if OPENPELS_ROOT is None or not OPENPELS_ROOT.is_dir() or NGSPICE is None:
    pytest.skip(
        "set OPENPELS_ROOT to the pinned OpenPELS checkout and install ngspice",
        allow_module_level=True,
    )
assert OPENPELS_ROOT is not None and NGSPICE is not None

pytestmark = [pytest.mark.long, pytest.mark.openpels, pytest.mark.serial]


@dataclass(frozen=True)
class OpenPELSCase:
    """One equivalent PESLite/native-ngspice power-stage configuration."""

    name: str
    source: str
    topology: str
    input_voltage: float
    duty: float
    inductance: float
    capacitance: float
    load: float
    resistance: float
    initial_voltage: float
    duration: float = 550e-6
    step: float = 50e-9
    output_period: float = 500e-9
    switching_frequency: float = 100e3

    def params(self):
        bridge = {
            "type": "half_bridge",
            "model": "switching",
            "switch_resistance": self.resistance,
            "diode_resistance": self.resistance,
        }
        buses = {}
        elements = {}
        bus = "output"
        dclink = {"vdc_ref": self.input_voltage, "source": {"type": "voltage"}}
        initial_state = "output.u_C"
        if self.topology == "buck":
            bridge.update(upper_controlled=True, lower_controlled=False)
            buses[bus] = {"phases": 1, "c": self.capacitance}
            elements["load"] = {"type": "resistor", "bus": bus, "r": self.load}
        else:
            bus = "input"
            bridge.update(upper_controlled=False, lower_controlled=True, reverse_port=True)
            buses[bus] = {"phases": 1}
            elements["supply"] = {
                "type": "voltage_source", "bus": bus,
                "v": self.input_voltage, "r": 1e-3,
            }
            dclink = {
                "vdc_ref": self.initial_voltage,
                "capacitor": {"c": self.capacitance},
                "source": {
                    "type": "current",
                    "i": -self.initial_voltage / self.load,
                    "k": 1.0 / self.load,
                },
            }
            initial_state = "converter.dclink.u_C"
        return from_dict({
            "base": {"s_base": 500.0},
            "buses": buses,
            "elements": elements,
            "units": {
                "converter": {
                    "type": "dc_converter", "bus": bus,
                    "filter": {"l_f": self.inductance, "r_f": 0.0},
                    "dclink": dclink,
                    "bridge": bridge,
                    "pwm": {"f_sw": self.switching_frequency},
                    "ctrl": {
                        "type": "open_loop", "computation": 0.0,
                        "loops": {"DutyModulation": {}},
                        "references": {"duty": self.duty},
                    },
                },
            },
            "simulation": {
                "t_end": self.duration,
                "initial": {"states": {initial_state: self.initial_voltage}},
                "energy_check": "off", "progress": {"enable": False},
                "solver": {
                    "type": "fixed", "method": "rk4", "dt": self.step,
                    "linearisations": 0,
                },
                "output": {
                    "period": self.output_period, "states": True,
                    "signals": False, "energy": False, "precision": 17,
                },
            },
        })


CASES = (
    OpenPELSCase(
        "native_ngspice_buck",
        "catalog/entries/ngspice_openmagnetics_buck_converter/source/buck_converter.cir",
        "buck", 10.0, 0.626718, 36.45830e-6, 100e-6, 2.5, 1e-3, 5.0,
    ),
    OpenPELSCase(
        "native_ngspice_boost",
        "catalog/entries/ngspice_openmagnetics_boost_converter/source/boost_converter.cir",
        "boost", 12.0, 0.7988166, 312e-6, 100e-6, 50.0, 1e-2, 50.0,
    ),
)


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"pinned OpenPELS source no longer contains exactly one {old!r}")
    return text.replace(old, new, 1)


def _record_voltage(deck: str) -> str:
    control = """.control
set noaskquit
set filetype=ascii
set wr_singlescale
set wr_vecnames
run
linearize v(vout)
wrdata results/waveform.csv v(vout)
quit
.endc
"""
    return re.sub(r"(?im)^\s*\.end\s*$", control + ".end", deck, count=1)


def _matched_deck(case: OpenPELSCase) -> str:
    deck = (OPENPELS_ROOT / case.source).read_text(encoding="utf-8")
    if case.topology == "buck":
        replacements = (
            ("Vpwm pwm_ctrl 0 PULSE(0 5 0 1.000000e-08 1.000000e-08 6.267180e-06 1.000000e-05)",
             "Vpwm pwm_ctrl 0 PULSE(5 0 3.133590e-06 1e-09 1e-09 3.732820e-06 1.000000e-05)"),
            (".model SW1 SW VT=2.500000 VH=0.500000",
             ".model SW1 SW VT=2.500000 VH=0.500000 RON=0.001 ROFF=1e9"),
            ("Rsnub_s1 vin_dc sw 100.000000", "Rsnub_s1 vin_dc sw 1e12"),
            ("Csnub_s1 vin_dc sw 1.000000e-10", "Csnub_s1 vin_dc sw 1e-15"),
            (".model DIDEAL D(IS=1.000000e-14 RS=1.000000e-06)",
             ".model DIDEAL D(IS=100u RS=0.001 N=0.05 Tt=0 Cjo=0)"),
            (".tran 5.000000e-08 5.500000e-04 5.000000e-04",
             ".tran 5.000000e-08 5.500000e-04 5.000000e-04 UIC"),
        )
    else:
        replacements = (
            ("L1 l_in sw 3.120000e-04 IC=4.225000",
             "L1 l_in sw 3.120000e-04 IC=0"),
            ("Vpwm pwm_ctrl 0 PULSE(0 5.000000 0 1.000000e-08 1.000000e-08 7.988166e-06 1.000000e-05)",
             "Vpwm pwm_ctrl 0 PULSE(5 0 3.994083e-06 1e-09 1e-09 2.011834e-06 1.000000e-05)"),
            ("Rsnub_s1 sw 0 10000.000000", "Rsnub_s1 sw 0 1e12"),
            ("Csnub_s1 sw 0 1.000000e-10", "Csnub_s1 sw 0 1e-15"),
            (".model DIDEAL D(IS=1.000000e-14 RS=1.000000e-06)",
             ".model DIDEAL D(IS=100u RS=0.01 N=0.05 Tt=0 Cjo=0)"),
        )
    for old, new in replacements:
        deck = _replace_once(deck, old, new)
    return _record_voltage(deck)


def _ngspice(case: OpenPELSCase, directory: Path):
    directory.mkdir()
    (directory / "results").mkdir()
    (directory / "case.cir").write_text(_matched_deck(case), encoding="utf-8")
    started = perf_counter()
    completed = subprocess.run(
        [NGSPICE, "-b", "case.cir"], cwd=directory,
        capture_output=True, text=True, timeout=120,
    )
    elapsed = perf_counter() - started
    log = (completed.stdout + "\n" + completed.stderr).lower()
    assert completed.returncode == 0, log[-2000:]
    assert not any(message in log for message in (
        "timestep too small", "run simulation(s) aborted", "doanalyses: tran:"
    )), log[-2000:]
    values = np.loadtxt(directory / "results" / "waveform.csv", skiprows=1)
    return values[:, 0], values[:, 1], elapsed


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_native_ngspice_power_stage_matches_peslite(case, tmp_path, record_property):
    started = perf_counter()
    result = peslite.Simulation(case.params()).run(out_dir=tmp_path / "peslite")
    peslite_seconds = perf_counter() - started
    reference_t, reference_voltage, ngspice_seconds = _ngspice(case, tmp_path / "ngspice")
    state = "output.u_C" if case.topology == "buck" else "converter.dclink.u_C"
    actual_t = np.asarray(result.t)
    selected = (actual_t >= reference_t[0]) & (actual_t <= reference_t[-1])
    actual = {"output_voltage": np.asarray(result.states[state])[selected]}
    expected = {
        "output_voltage": np.interp(actual_t[selected], reference_t, reference_voltage)
    }
    errors = normalized_errors(actual, expected)
    difference = actual["output_voltage"] - expected["output_voltage"]
    root_mean_square = float(np.sqrt(np.mean(difference ** 2)))
    maximum_absolute = float(np.max(np.abs(difference)))
    record_property("openpels_commit", OPENPELS_COMMIT)
    record_property("root_mean_square_difference_V", root_mean_square)
    record_property("maximum_absolute_difference_V", maximum_absolute)
    assert root_mean_square <= 0.02
    assert maximum_absolute <= 0.1
    check_comparison(
        errors, 0.005,
        {"peslite": peslite_seconds, "ngspice": ngspice_seconds},
        "ngspice", record_property,
    )
