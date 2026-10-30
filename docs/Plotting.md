# Plotting

Install the optional plotting dependency:

```bash
pip install "peslite[plot]"
```

## Command line

```bash
peslite case.pes --plot vsc.dclink.u_C vsc.ctrl.SRFPLL.theta \
  --plot-ylabel '$x$ (pu)'
```

PESLite runs the simulation first, reads the generated CSV and writes a vector PDF beside it.
Requested columns must belong to one CSV table. Plant and controller columns require
`simulation.output.signals: 1`.

| Option | Value |
|---|---|
| `--plot` | one or more column names |
| `--plot-x` | horizontal column; default `t` |
| `--plot-title` | figure title |
| `--plot-label` | waveform label; repeat in `--plot` order |
| `--plot-xlabel`, `--plot-ylabel` | axis labels |
| `--plot-color`, `--plot-style` | waveform appearance; repeat in `--plot` order |
| `--plot-xlim`, `--plot-ylim` | axis limits |
| `--plot-time-unit` | `s`, `ms` or `us` |
| `--plot-width` | IEEE column-width preset |
| `--plot-height` | figure height in inches |
| `--plot-linewidth` | line width in points |
| `--plot-output` | output PDF path |
| `--plot-no-grid`, `--plot-no-legend` | disable grid or legend |

```bash
peslite case.pes --plot vsc.dclink.u_C \
  --plot-title 'DC-link voltage' \
  --plot-label '$u_{dc}$' \
  --plot-ylabel '$u_{dc}$ (V)' \
  --plot-xlim 0.1 0.2 \
  --plot-width double \
  --plot-output dc-link.pdf
```

`--plot-xlim` uses the CSV value scale. `--plot-time-unit` changes only the displayed time scale.
Without an explicit time unit, the plot selects seconds, milliseconds or microseconds.

## Python API

Plot a completed Python result:

```python
pdf = result.plot(
    ["vsc.dclink.u_C", "vsc.ctrl.SRFPLL.theta"],
    labels=[r"$u_{dc}$", r"$\theta$"],
    ylabel=r"$x$ (pu)",
)
```

Plot Python or standalone C++ CSV output directly:

```python
from peslite.addons import plot_csv

pdf = plot_csv(
    "output/run/states.csv",
    "vsc.dclink.u_C",
)
```

Both functions return the written PDF path. Matplotlib is imported only when plotting is called;
plotting is not included in a generated C++ simulator.

## Output style

The default output uses IEEE figure dimensions, serif/Computer Modern text, inward ticks,
restrained colors and vector PDF. External LaTeX is used when available, with a bundled
Computer Modern-style fallback.
