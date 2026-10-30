# PESLite

**[Documentation](https://github.com/lonaparte/PESLite/wiki)**
· **[PyPI](https://pypi.org/project/peslite/)**
· **[Releases](https://github.com/lonaparte/PESLite/releases)**

PESLite is an open-source, lightweight, multirate, and extensible power electronics simulator in
Python. It provides time-domain simulation for power-electronic converters and converter-based 
systems—combining switching and averaged models, digital control timing, event-driven operation, 
and flexible multi-time-scale numerical integration via readable YAML files and a Python 
API—designed as a deterministic simulation backend for LLM reasoning and verification.

## Features

- AC and DC converter networks, including grid-connected and wireless-power-transfer systems.
- Configurable grid-following, grid-forming and DC converter control.
- Switching, PWM-period-averaged and continuously averaged models.
- Digital control timing, sensing and converter protection.
- Fixed-step, adaptive and multirate integration with exact event boundaries.
- Simulation events and continuation from saved states.
- Streamed waveform output, optional plotting and energy accounting.
- Custom control loops, circuit elements, events and solvers through the Python extension API.
- Standalone C++17 export with no third-party C++ runtime dependencies.

## Install

```bash
pip install peslite
```

Use `pip install "peslite[all]"` to include every optional runtime add-on.

PESLite requires Python 3.10 or newer.

## Quick start

Run a bundled example:

```bash
peslite gfl-example
```

Run `case.pes` from the current directory:

```bash
peslite case.pes
```

Results are written to `output/<case-name>/` by default.

Override a parameter by dotted path when needed:

```bash
peslite case.pes --set simulation.t_end=1.0 --out output/test
```

Bridge and solver presets are available through three run modes:

```bash
peslite case.pes --switching
peslite case.pes --pwm-averaging
peslite case.pes --averaging
```

See [Converter and Bridge Models](https://github.com/lonaparte/PESLite/wiki/Converter-and-Bridge-Models)
for model fidelity, control timing and solver presets.

Use `peslite --help` for the full interface.

The same workflow is available from Python:

```python
import peslite

params = peslite.load("case.pes")
result = peslite.Simulation(params).run()

print(result.summary)
print(result.final_states())
```

## C++ export

Select the parameters that should remain variable in the compiled simulator, or use `common`:

```bash
peslite-convert case.pes common
c++ -O3 -DNDEBUG -std=c++17 export/case/peslite.cpp -o export/case/peslite
```

The generated executable uses the same parameter-override and output conventions as PESLite and
writes the same result structure. See the [C++ Export guide](https://github.com/lonaparte/PESLite/wiki/Cpp-Export)
for compilation, runtime configuration and current backend limits.

## Documentation

- [Getting Started](https://github.com/lonaparte/PESLite/wiki/Getting-Started)
- [Examples](https://github.com/lonaparte/PESLite/wiki/Examples)
- [Simulation Files](https://github.com/lonaparte/PESLite/wiki/Simulation-Files)
- [Converter and Bridge Models](https://github.com/lonaparte/PESLite/wiki/Converter-and-Bridge-Models)
- [Control and PWM Timing](https://github.com/lonaparte/PESLite/wiki/Control-and-PWM-Timing)
- [Solvers](https://github.com/lonaparte/PESLite/wiki/Solvers)
- [Multirate Simulation](https://github.com/lonaparte/PESLite/wiki/Multirate-Simulation)
- [Events and Restart](https://github.com/lonaparte/PESLite/wiki/Events-and-Restart)
- [Output and Results](https://github.com/lonaparte/PESLite/wiki/Output-and-Results)
- [Project Workspace](https://github.com/lonaparte/PESLite/wiki/Project-Workspace)
- [Extending PESLite](https://github.com/lonaparte/PESLite/wiki/Extending-PESLite)
- [Architecture](https://github.com/lonaparte/PESLite/wiki/Architecture)
- [CLI Reference](https://github.com/lonaparte/PESLite/wiki/CLI-Reference)

The Wiki documents the current `main` branch. Releases and the changelog record versioned changes.

## Development

```bash
python -m pytest
```

## License

GNU Affero General Public License v3.0 only (`AGPL-3.0-only`). See `LICENSE`.
