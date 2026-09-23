"""Recording of states, signals and energy during a run, and export of the result files.

Plant signals are in SI, controller logs in each controller's pu bases; complex states are
split into ``.re``/``.im`` columns.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..firmware.transforms import complex2abc
from ..params import Params, dump

__all__ = ["SimulationResult", "Recorder"]


@dataclass
class SimulationResult:
    params: Params
    t: np.ndarray
    plant: dict[str, np.ndarray]
    control: dict[str, np.ndarray]
    states: dict[str, np.ndarray] = field(default_factory=dict)
    energy: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    n_rhs: int = 0

    @property
    def tripped(self) -> bool:
        return bool(self.summary.get("tripped", 0.0))

    def columns(self) -> dict[str, np.ndarray]:
        """Return flat real columns: plant quantities in SI and controller logs in pu.

        Names: ``<bus>.v_a``, ``<unit>.i_conv_a``, ``<branch>.i_a``, ``<source>.i_a``, ``<source>.angle``,
        ``<unit>.u_dc``/``<unit>.i_dc`` (V/A) and the controllers' logged signals.
        """
        cols: dict[str, np.ndarray] = {"t": self.t}
        for key, vec in self.plant.items():
            head, _, what = key.rpartition(".")
            if what in ("u_g", "i_c", "i", "u") and len(vec) and np.iscomplexobj(vec):
                stem = {"u_g": "v", "i_c": "i_conv", "i": "i", "u": "v"}[what]
                abc = np.array([complex2abc(z) for z in vec])
                for k, ph in enumerate("abc"):
                    cols[f"{head}.{stem}_{ph}"] = abc[:, k]
        for name in self.params.units:
            if f"{name}.u_dc" in self.plant:
                cols[f"{name}.u_dc"] = self.plant[f"{name}.u_dc"]
            if f"{name}.i_dc" in self.plant:
                cols[f"{name}.i_dc"] = self.plant[f"{name}.i_dc"]
        for name in self.params.sources:
            if f"{name}.angle" in self.plant:
                cols[f"{name}.angle"] = self.plant[f"{name}.angle"]
        for key, arr in self.plant.items():
            if key.startswith("ctrl_"):
                cols[key[5:]] = arr
        return cols

    def to_csv(self, path: str | Path) -> None:
        _write(path, self.columns())

    def control_to_csv(self, path: str | Path) -> None:
        """Write one control-log CSV per unit, ``<path stem>.<unit>.csv``; return the paths."""
        path = Path(path)
        written = []
        for name in self.params.units:
            head = f"{name}."
            cols = {"t": self.control[f"{name}.t"]} if f"{name}.t" in self.control else {}
            cols.update({k[len(head):]: v for k, v in self.control.items()
                         if k.startswith(head) and k != f"{name}.t"})
            if len(cols) > 1:
                out = path.with_name(f"{path.stem}.{name}{path.suffix}")
                _write(out, cols)
                written.append(out)
        return written

    # ------------------------------------------------------------ states
    def final_states(self) -> dict[str, float]:
        """Return the last row of the state table (``t`` included)."""
        return {k: float(v[-1]) for k, v in self.states.items()}

    def states_to_csv(self, path: str | Path) -> None:
        """Write the state table at full precision, so a row read back reproduces the state exactly."""
        keys = list(self.states)
        with Path(path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            columns = [self.states[k].tolist() for k in keys]
            for row in zip(*columns):
                w.writerow([repr(v) for v in row])

    def energy_to_csv(self, path: str | Path) -> None:
        keys = list(self.energy)
        with Path(path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for row in zip(*(self.energy[k] for k in keys)):
                w.writerow([f"{v:.10g}" for v in row])

    def save(self, out_dir: str | Path, info: dict[str, Any] | None = None) -> list[Path]:
        """Write the configured output files into ``out_dir`` and return their paths.

        info: extra entries for ``summary.json``.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        if self.params.output.states:
            self.states_to_csv(out / "states.csv")
            written.append(out / "states.csv")
        if self.params.output.signals:
            self.to_csv(out / "plant.csv")
            written += [out / "plant.csv"] + self.control_to_csv(out / "control.csv")
        if self.params.output.energy and self.energy:
            self.energy_to_csv(out / "energy.csv")
            written.append(out / "energy.csv")
        summary = {**(info or {}), "wall_time_s": self.wall_time, "n_rhs": self.n_rhs, **self.summary}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        dump(self.params, out / "params.yaml")
        written += [out / "summary.json", out / "params.yaml"]
        return written

def _write(path: str | Path, cols: dict) -> None:
    keys = list(cols)
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for row in zip(*(cols[k] for k in keys)):
            w.writerow([f"{v:.10g}" for v in row])


class Recorder:
    """Collects snapshots during a run. ``keep_states=False`` keeps only the last state row."""

    def __init__(self, keep_states: bool = True) -> None:
        self.t: list[float] = []
        self.plant: dict[str, list] = {}
        self.ctrl_t: dict[str, list[float]] = {}   # one time grid per unit
        self.ctrl: dict[str, list] = {}
        self.last_ctrl_log: dict[str, dict[str, float]] = {}
        self.keep_states = keep_states
        self.state_keys: list[str] | None = None
        self.state_t: list[float] = []
        self.state_rows: list[list[float]] = []
        self.energy_t: list[float] = []
        self.energy_rows: dict[str, list] = {}

    def energy_row(self, t: float, columns: dict[str, float]) -> None:
        self.energy_t.append(t)
        for k, v in columns.items():
            self.energy_rows.setdefault(k, []).append(v)

    def state_row(self, t: float, row: dict[str, float]) -> None:
        if self.state_keys is None:
            self.state_keys = list(row)
        elif len(row) != len(self.state_keys):
            raise RuntimeError("the set of named states changed during the run")
        if not self.keep_states:
            self.state_t.clear()
            self.state_rows.clear()
        self.state_t.append(t)
        self.state_rows.append(list(row.values()))

    def plant_snapshot(self, t: float, signals: dict[str, Any]) -> None:
        n_before = len(self.t)
        self.t.append(t)
        for k, v in signals.items():
            self.plant.setdefault(k, []).append(v)
        for unit, log in self.last_ctrl_log.items():
            for k, v in log.items():
                key = f"ctrl_{unit}.{k}"
                if key not in self.plant:  # controller signal appearing after the first snapshots
                    self.plant[key] = [math.nan] * n_before
                self.plant[key].append(v)

    def control_sample(self, unit: str, t: float, log: dict[str, float]) -> None:
        """Record one controller's log at one of its sampling instants."""
        self.ctrl_t.setdefault(unit, []).append(t)
        for k, v in log.items():
            self.ctrl.setdefault(f"{unit}.{k}", []).append(v)

    def arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        plant = {k: np.asarray(v) for k, v in self.plant.items()}
        ctrl = {f"{unit}.t": np.asarray(times) for unit, times in self.ctrl_t.items()}
        ctrl.update({k: np.asarray(v) for k, v in self.ctrl.items()})
        states: dict[str, np.ndarray] = {"t": np.asarray(self.state_t, dtype=float)}
        if self.state_keys:
            table = np.asarray(self.state_rows, dtype=float).reshape(len(self.state_rows), len(self.state_keys))
            states.update({k: table[:, j] for j, k in enumerate(self.state_keys)})
        energy: dict[str, np.ndarray] = {}
        if self.energy_t:
            energy = {"t": np.asarray(self.energy_t, dtype=float)}
            energy.update({k: np.asarray(v, dtype=float) for k, v in self.energy_rows.items()})
        return np.asarray(self.t), plant, ctrl, states, energy
