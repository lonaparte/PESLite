# PESLite

PESLite is an open-source, lightweight, multirate, and extensible power electronics simulator in
Python. It provides time-domain simulation for power-electronic converters and converter-based
systems—combining switching and averaged models, sampled control, event-driven operation,
and flexible multi-time-scale numerical integration via readable simulation files and a Python
API—designed as a deterministic simulation backend for LLM reasoning and verification.

> This documentation describes the current development version on the `main` branch. For released
> versions and changes, see [GitHub Releases](https://github.com/lonaparte/PESLite/releases) and
> the [changelog](https://github.com/lonaparte/PESLite/blob/main/CHANGELOG.md).

PESLite supports:

- AC and DC converter networks, including grid-connected and wireless-power-transfer systems.
- Configurable grid-following, grid-forming and DC converter control.
- Switching, PWM-period-averaged and continuously averaged models.
- Sampled control, sensing and converter protection.
- Fixed-step, adaptive and multirate integration with exact event boundaries.
- Simulation events and continuation from saved states.
- Streamed waveform output, optional plotting and energy accounting.
- Custom control loops, circuit elements, events and solvers through the Python extension API.
- Standalone C++17 export with no third-party C++ runtime dependencies.

The power circuit uses SI units. Controller quantities ending in `_pu` use each converter's own
per-unit base.

## Installation

```bash
pip install peslite
```

PESLite requires Python 3.10.1 or newer.

## Quick start

Run a bundled example:

```bash
peslite gfl-example
```

To run a file named `case.pes` in the current directory:

```bash
peslite case.pes
```

Results are written to `output/<case-name>/`; these commands use `output/gfl-example/` and
`output/case/`, respectively.

See [Getting Started](Getting-Started.md) to create a case and inspect its results.

## Documentation

- [Getting Started](Getting-Started.md)
- [Project Workspace](Project-Workspace.md)
- [Simulation Files](Simulation-Files.md)
- [Graphical Interface](Graphical-Interface.md)
- [Events Timeline](Events-Timeline.md)
- [Output and Results](Output-and-Results.md)
- [Solvers](Solvers.md)
- [Multirate Simulation](Multirate-Simulation.md)
- [Passive Components](Passive-Components.md)
- [Electrical Sources](Electrical-Sources.md)
- [Bridge and PWM](Bridge-and-PWM.md)
- [Voltage-Source Converters](Voltage-Source-Converters.md)
- [DC converters and WPT](DC-Converters-and-WPT.md)
- [Black-box Models](Black-Box-Models.md)
- [Architecture](Architecture.md)
- [Plotting](Plotting.md)
- [Online Stability Analysis](Online-Stability-Analysis.md)
- [Extending PESLite](Extending-PESLite.md)
- [C++ Export](Cpp-Export.md)
- [Imperix ELF Export (Experimental)](Imperix-Export.md)
- [CLI Reference](CLI-Reference.md)
- [Examples](Examples.md)
- [FAQ](FAQ.md)

## Project links

- [Examples](https://github.com/lonaparte/PESLite/tree/main/examples)
- [GitHub repository](https://github.com/lonaparte/PESLite)
- [PyPI](https://pypi.org/project/peslite/)
- [Issue tracker](https://github.com/lonaparte/PESLite/issues)
