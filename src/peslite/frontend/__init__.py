"""PESLite frontend service.

The topology view, parameter editor, waveform plots and small-signal panel communicate with a
resident process over line-delimited JSON on stdin/stdout. Each PESLite simulation file is one
selectable system. Start the service with::

    python -m peslite.frontend

PESLite initially incorporated the [CodeSim](https://github.com/NoClassDefError/codesim) frontend implementation and has since developed it further.
"""

from .descriptor import SignalRecorder, Topology, small_signal_descriptor
from .service import FrontendService, discover_cases, main

__all__ = ["FrontendService", "SignalRecorder", "Topology", "discover_cases", "main",
           "small_signal_descriptor"]
