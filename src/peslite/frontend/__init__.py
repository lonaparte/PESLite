"""Backend for the CodeSim VS Code extension (``codesim-frontend``).

The extension's topology view, parameter editor, waveform plots and small-signal panel drive a
resident process over line-delimited JSON on stdin/stdout. This package serves that protocol for
PESLite simulation files: each file is one selectable "system". Start it with::

    python -m peslite.frontend

The extension recognizes a PESLite checkout by ``src/peslite/frontend`` and runs this module.
"""

from .descriptor import SignalRecorder, Topology, small_signal_descriptor
from .service import FrontendService, discover_cases, main

__all__ = ["FrontendService", "SignalRecorder", "Topology", "discover_cases", "main",
           "small_signal_descriptor"]
