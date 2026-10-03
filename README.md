# PESLite

**[Documentation](https://github.com/lonaparte/PESLite/wiki)**
· **[PyPI](https://pypi.org/project/peslite/)**
· **[Releases](https://github.com/lonaparte/PESLite/releases)**

PESLite is an open-source, lightweight, multirate, and extensible power electronics simulator in
Python. It is designed for time-domain studies of power-electronic converters and converter-based
systems, combining switching and averaged models, digital control timing, event-driven operation,
and flexible numerical integration across multiple time scales in readable YAML simulation files
and a Python API.

## Features

- Converter networks with buses, branches, grid sources, R-L loads and multiple independently
  named converter units.
- Grid-following and grid-forming controllers assembled from typed control loops.
- Per-converter `switching`, PWM-period-averaged and ideal continuously averaged bridge models.
- ADC sampling windows, controller interrupts, computation time, PWM shadow/active registers,
  startup sequencing and unit-owned protection.
- Fixed-step, adaptive and multirate integration with exact event boundaries.
- Connect, disconnect and validated parameter-change events, plus continuation from saved states.
- Streamed CSV output, energy accounting, port-Hamiltonian structure reporting and a lazy Python
  result API.
- Optional IEEE/LaTeX-style vector PDF plots from completed CSV results.
- Optional project-local extensibility through a `PESaddons/` directory beside a simulation file, plus
  registration APIs for custom control loops, circuit elements, events and solvers.
- Export as a parameter-specialized, standalone C++17 simulator with no third-party C++ runtime
  dependencies.

Plant quantities use SI units. Controller quantities ending in `_pu` use each converter's own
per-unit base.

## Install

```bash
pip install peslite
```

Use `pip install "peslite[all]"` to include every optional runtime add-on.

PESLite requires Python 3.10 or newer.

## Quick start

Run a bundled case:

```bash
peslite gfl-example
```

Run a simulation file and override any parameter by dotted path:

```bash
peslite case.pes --set simulation.t_end=1.0 --out output/test
```

Results are written to `output/<name>/`. Bridge/solver presets are available through
three run modes:

```bash
peslite case.pes --switching
peslite case.pes --pwm-averaging
peslite case.pes --averaging
```

- `--switching` uses exact switching bridges with fixed-step RK4 integration.
- `--pwm-averaging` averages each PWM period while retaining sampled digital-control timing and
  uses fixed-step RK4 integration.
- `--averaging` uses ideal averaged controlled-voltage-source bridges, continuous control and
  adaptive DP45 integration.

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

Select the parameters that should remain variable in the compiled simulator, or use `all`:

```bash
peslite-convert case.pes all
c++ -O3 -DNDEBUG -std=c++17 export/case/peslite.cpp -o export/case/peslite
```

The generated executable uses the same parameter-override and output conventions as PESLite and
writes the same result structure. See the [C++ Export guide](https://github.com/lonaparte/PESLite/wiki/Cpp-Export)
for compilation, runtime configuration and current backend limits.

## Documentation

- [Getting Started](https://github.com/lonaparte/PESLite/wiki/Getting-Started)
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
