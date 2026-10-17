"""Ahead-of-time export of a resolved :class:`~peslite.Simulation`.

An exporter receives the same assembled simulation used by the Python runner.  It may specialise
the implementation to that parameter tree, but it must preserve the public state, signal and file
names.  Exported programs are calculators, not recordings of a Python run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable

from .params import Params

if TYPE_CHECKING:  # pragma: no cover - imports used only by type checkers
    from ..solver.simulation import Simulation

__all__ = ["ExportResult", "EXPORTERS", "export"]


@dataclass(frozen=True)
class ExportResult:
    """Files and static diagnostics of one generated standalone simulator."""

    format: str
    out_dir: Path
    files: tuple[Path, ...]
    diagnostics: tuple[dict[str, Any], ...] = ()


Exporter = Callable[["Simulation", Path, str, tuple[str, ...], bool], ExportResult]
EXPORTERS: dict[str, Exporter] = {}


def _exporter(name: str):
    def register(function: Exporter) -> Exporter:
        EXPORTERS[name] = function
        return function
    return register


def export(what: Params | "Simulation", format: str, out_dir: str | Path | None = None,
           *, name: str = "run", variables: Iterable[str] = ()) -> ExportResult:
    """Export a complete configured simulator in ``format``.

    ``what`` may be a resolved :class:`Params` tree or an unrun :class:`Simulation`.  With no
    explicit directory the result is written to ``export/<name>``; the Python API consequently
    uses ``export/run`` by default, matching ``Simulation.run()``'s ``output/run`` convention.
    ``variables`` names parameters retained for run-time overrides by the exported program;
    ``("all",)`` selects every parameter supported by that backend and configuration.
    ``("common",)`` keeps that set except parameters flagged by structural sensitivity checks.
    """
    # The local imports avoid making the C++ backend part of ordinary simulation import time.
    from ..solver.simulation import Simulation

    key = str(format).lower()
    if key == "cpp" and key not in EXPORTERS:
        from . import cpp as _cpp  # noqa: F401
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
    return EXPORTERS[key](simulation, destination, name, selected, not caller_owned)
