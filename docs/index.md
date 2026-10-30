# PESLite

PESLite is an open-source, lightweight, multirate, and extensible power electronics simulator in
Python. It is designed for time-domain studies of power-electronic converters and converter-based
systems, combining switching and averaged models, sampled control, event-driven operation
and flexible numerical integration across multiple time scales.

> This documentation describes the current development version on the `main` branch. For released
> versions and changes, see [GitHub Releases](https://github.com/lonaparte/PESLite/releases) and
> the [changelog](https://github.com/lonaparte/PESLite/blob/main/CHANGELOG.md).

PESLite supports:

- converter networks described by readable YAML simulation files;
- switching, PWM-period-averaged and ideal averaged bridge models;
- grid-following and grid-forming control;
- fixed-step, adaptive and multirate integration;
- ADC and PWM periods, sampled control, protection and energy accounting;
- events and continuation from saved states;
- export as a standalone C++17 simulator.

The power circuit uses SI units. Controller quantities ending in `_pu` use each converter's own
per-unit base.

## Installation

```bash
pip install peslite
```

PESLite requires Python 3.10 or newer.

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
- [Events Timeline](Events-Timeline.md)
- [Output and Results](Output-and-Results.md)
- [Solvers](Solvers.md)
- [Multirate Simulation](Multirate-Simulation.md)
- [Passive Network Components](Passive-Network-Components.md)
- [Electrical Sources](Electrical-Sources.md)
- [Bridge and PWM](Bridge-and-PWM.md)
- [Voltage-Source Converters](Voltage-Source-Converters.md)
- [DC converters and WPT](DC-Converters-and-WPT.md)
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
