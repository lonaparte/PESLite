"""Ahead-of-time export of a resolved :class:`~peslite.Simulation`.

An exporter receives the same assembled simulation used by the Python runner.  It may specialise
the implementation to that parameter tree, but it must preserve the public state, signal and file
names.  Exported programs are calculators, not recordings of a Python run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

from .params import Params

if TYPE_CHECKING:  # pragma: no cover - imports used only by type checkers
    from ..solver.simulation import Simulation

__all__ = ["ExportResult", "EXPORTERS", "export"]


def _validate_dynamic_capabilities(simulation: "Simulation", backend: str) -> None:
    """Reject runtime semantics that a generated backend cannot reproduce."""
    if simulation.p.simulation.solver.measurement_interval is not None:
        raise ValueError(
            f"{backend} export does not implement an additional ADC integration grid; "
            "omit simulation.solver.measurement_interval"
        )
    unsupported = [
        name for name, unit in simulation.system.units.items()
        if not getattr(getattr(unit.dclink, "source", None), "native_export", True)
    ]
    if unsupported:
        raise ValueError(
            f"{backend} export does not implement the time-varying input source of "
            f"unit(s) {unsupported}"
        )
    unsupported_loops = [
        f"{unit_name}.{loop_name}"
        for unit_name, unit in simulation.system.units.items()
        for loop_name, loop in getattr(getattr(unit.ctrl, "graph", None), "nodes", {}).items()
        if not getattr(loop, "native_export", True)
    ]
    if unsupported_loops:
        raise ValueError(
            f"{backend} export does not implement control loop(s) {unsupported_loops}"
        )


def _signal_names(path: str, phases: int = 1) -> tuple[str, ...]:
    """Keep the canonical signal path; three-phase columns only append phase suffixes."""
    if phases == 1:
        return (path,)
    if phases == 3:
        return path + "_a", path + "_b", path + "_c"
    raise ValueError(f"signal phases must be 1 or 3, got {phases}")


@dataclass(frozen=True)
class ExportResult:
    """Files and static diagnostics of one exported implementation."""

    format: str
    out_dir: Path
    files: tuple[Path, ...]
    diagnostics: tuple[dict[str, Any], ...] = ()


Exporter = Callable[..., ExportResult]
EXPORTERS: dict[str, Exporter] = {}


def _exporter(name: str):
    def register(function: Exporter) -> Exporter:
        EXPORTERS[name] = function
        return function
    return register


def export(what: Params | "Simulation", format: str, out_dir: str | Path | None = None,
           *, name: str = "run", variables: Iterable[str] = (), **options) -> ExportResult:
    """Export a configured simulation using the backend selected by ``format``.

    ``what`` may be a resolved :class:`Params` tree or an unrun :class:`Simulation`.  With no
    explicit directory the result is written to ``export/<name>``; the Python API consequently
    uses ``export/run`` by default, matching ``Simulation.run()``'s ``output/run`` convention.
    ``variables`` names parameters retained for run-time overrides by the exported program;
    ``("all",)`` selects every parameter supported by that backend and configuration.
    ``("common",)`` keeps that set except parameters flagged by structural sensitivity checks.
    Backend options (for example Imperix ``target_config``, ``sdk`` and ``build``) are forwarded
    to the selected exporter; ordinary simulation and other backends do not interpret them.
    """
    # The local imports avoid making the C++ backend part of ordinary simulation import time.
    from ..solver.simulation import Simulation

    key = str(format).lower()
    if key in ("cpp", "imperix") and key not in EXPORTERS:
        import_module(f".{key}", __package__)
    if key not in EXPORTERS:
        known = ", ".join(sorted(EXPORTERS)) or "none"
        raise ValueError(f"unknown export format {format!r}; known: {known}")
    caller_owned = isinstance(what, Simulation)
    simulation = what if caller_owned else Simulation(what)
    if simulation.result is not None:
        raise RuntimeError("an already-run Simulation cannot be exported; build one from its Params")
    destination = Path(out_dir) if out_dir is not None else Path("export") / name
    selected = tuple(str(path) for path in variables)
    # Backends may prepare an internally constructed Simulation in place.  A Simulation supplied
    # by the caller remains private to the caller and therefore must still be copied by a backend.
    return EXPORTERS[key](simulation, destination, name, selected, not caller_owned, **options)
