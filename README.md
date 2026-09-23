# peslite

Time-domain simulation of grid-connected voltage-source converters in Python.

- Networks of buses, lines, grid sources and any number of converters, defined in YAML.
- Grid-following (PLL, current loop, dc-voltage loop) and grid-forming control
  (PSC, droop, VSG, dVOC, matching), each loop on its own clock.
- Switching (ideal switches, exact switching instants), averaged and step-averaged bridges.
- Fixed-step, adaptive (SciPy or built-in DP45) and multirate integration.
- ADC sampling (instantaneous or window average), computation delay, PWM, protection.
- Energy accounting of the power circuit and restart from any saved state.

The power circuit works in SI units (V, A, H, F, ohm); controllers work in pu of each
converter's own base. Time is in s, angles in rad.

## Installation

```bash
python -m pip install -e .            # numpy, scipy, PyYAML
```

Python 3.10 or newer.

## Quick start

```bash
# run the first configuration in configs/ (sorted by name)
python peslite.py

# run a configuration by name or path; results go to output/<name>/
python peslite.py gfl
python peslite.py configs/gfm-psc.yaml --out output/psc

# override any parameter by its dotted path
python peslite.py gfl --set simulation.t_end=1 --set units.vsc.delay.steps=1
python peslite.py gfl --set simulation.solver.type=adaptive --set simulation.solver.method=DP45
python peslite.py gfm-droop --set simulation.bridge=switching

# continue a run from its last saved state (or from time T with --initial-time T)
python peslite.py gfl --out output/a
python peslite.py gfl --initial output/a/states.csv --out output/b

# inspect a configuration
python peslite.py gfl --list-states     # state names (the states.csv columns)
python peslite.py gfl --ph-report       # port-Hamiltonian structure of the circuit
python peslite.py --help
```

From Python (in this folder, or anywhere after `pip install -e .`):

```python
import peslite

p = peslite.load("configs/gfl.yaml", **{"simulation.t_end": 1.0})
r = peslite.Simulation(p).run()

r.states["plant.vsc.dclink.u_C"]   # a state over time
r.plant["vsc.i_c"]                 # plant signals (complex space vectors)
r.control["vsc.id_pu"]             # controller log of unit "vsc"
r.final_states()                   # last row, usable as an initial state
r.summary                          # trips, alarms, peaks
r.save("output/run")               # states.csv, summary.json, params.yaml
```

## Example configurations

| File | Content |
|---|---|
| `configs/gfl.yaml` | Grid-following converter; every key is annotated |
| `configs/gfm-psc.yaml` | Grid-forming, power-synchronization control |
| `configs/gfm-droop.yaml` | Grid-forming, droop with virtual admittance and current loop |
| `configs/gfm-vsg.yaml` | Grid-forming, virtual synchronous generator |
| `configs/gfm-dvoc.yaml` | Grid-forming, dispatchable virtual oscillator control |
| `configs/gfm-matching.yaml` | Grid-forming, matching control |
| `configs/two-converters.yaml` | A grid-forming and a grid-following unit on one grid |

## Frequently used settings

| Path | Values |
|---|---|
| `simulation.t_end` | end time, s |
| `simulation.bridge` | `switching` \| `averaged` \| `step_averaged` |
| `simulation.solver.type` / `.method` | `fixed`: `euler` \| `heun` \| `rk4`; `adaptive`: `RK45` \| `DOP853` \| `Radau` \| `BDF` \| `LSODA` \| `DP45` |
| `simulation.solver.dt` | maximum fixed step, s |
| `simulation.solver.subsystems` | own steps per subsystem, e.g. `{vsc.dclink: 10, pcc: 0.1}` |
| `simulation.log.plant_period` | snapshot interval, s |
| `simulation.energy_check` | `warn` \| `strict` \| `off` |
| `units.<u>.control.type` | `gfl` \| `gfm` \| `custom` |
| `units.<u>.control.loops.<loop>.period` | loop period, s |
| `units.<u>.measurement.average` | `instantaneous` \| `window` (with `window_s`) |
| `units.<u>.pwm.method` / `.sync` | `spwm` \| `svpwm`; `asynchronous` \| `synchronous` |
| `units.<u>.delay.steps` | computation delay in PWM updates |
| `output.states` / `.signals` / `.energy` | which files are written |

## Output

| File | Content |
|---|---|
| `states.csv` | every state at each snapshot; any row can start a new run |
| `plant.csv`, `control.<unit>.csv` | plant signals and controller logs (`output.signals: true`) |
| `energy.csv` | stored energy and power balance (`output.energy: true`) |
| `summary.json`, `params.yaml` | run summary and the full parameter set |

## Custom parts

```python
# replace one control loop's algorithm
ctrl = peslite.UniteType(p.unit("vsc"), loop_overrides={"pll": MyPLL()})
sim = peslite.Simulation(p, instances={"vsc.ctrl": ctrl})

# or pass a factory, so the part can be rebuilt
sim = peslite.Simulation(p, parts={"vsc.ctrl": lambda cfg: peslite.UniteType(cfg)})
```

Other replaceable parts: `pwm_method=` and `limiter=` of `UniteType`, `<unit>.modulator`,
`<unit>.delay`, `solver`, and extra circuit elements through `System(p, elements=[...])`.
See `examples/custom_plant.py` (custom network section, synchronization law and solver) and
`examples/compare_solvers.py`.

## Layout

The repository folder is the `peslite` package.

```
peslite.py      command line
simulation.py   simulation loop
__init__.py     package exports
phs/            circuit model, energy accounting, solvers
params/         parameter schema, validation, file I/O
power/          sources, lines, buses, bridge, dc link
control/        PLL, current and dc-voltage loops, grid-forming laws, control graph
firmware/       pu conversion, limiter, delay, transforms
modulation/     PWM methods and modulators
sensing/        ADC sampling
protection/     relay
results/        recording and result files
assembly/       converter unit, system assembly
configs/        example configurations
examples/       usage examples
```

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See `LICENSE`.
