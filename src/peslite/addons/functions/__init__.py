"""Optional user-facing functions that stay outside the simulation hot path."""

from .energy_audit import energy_closure
from .plotting import plot_csv, plot_result

__all__ = ["energy_closure", "plot_csv", "plot_result"]
