"""Optional IEEE-style PDF plotting function for numeric CSV results.

Matplotlib is imported only when :func:`plot_csv` is called.  Simulation and C++ export therefore
do not depend on, import, or execute plotting code.
"""

from __future__ import annotations

import csv
import logging
import math
import shutil
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["plot_csv", "plot_result"]


PALETTE = (
    "#1B4F8A",  # deep navy
    "#8B2020",  # dark burgundy
    "#1A6B2A",  # forest green
    "#B05A00",  # burnt orange
    "#5B2D8E",  # deep purple
    "#444444",  # dark gray
    "#0B6B6B",  # deep teal
    "#6B3A1F",  # dark brown
    "#5A6B00",  # olive green
    "#8B1A5A",  # deep rose
    "#0A6080",  # steel cyan
    "#2E7D00",  # bright green
)

_WIDTHS = {"single": 3.54, "double": 7.16}
_TIME_FACTORS = {"s": 1.0, "ms": 1e3, "us": 1e6}
_TIME_LABELS = {"s": r"$t$ (s)", "ms": r"$t$ (ms)", "us": r"$t$ ($\mu$s)"}
_LATEX_ESCAPES = str.maketrans({
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
})


def _matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise RuntimeError(
            "plotting requires Matplotlib; install it with `pip install 'peslite[plot]'`"
        ) from exc
    return matplotlib


def _columns(values: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        result = (values,)
    else:
        result = tuple(values)
    if not result or any(not value for value in result):
        raise ValueError("at least one non-empty CSV column is required")
    if len(set(result)) != len(result):
        raise ValueError("CSV columns must not be repeated")
    return result


def _read_numeric_columns(path: Path, x: str,
                          columns: tuple[str, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    header = _header(path)
    if header is None:
        raise ValueError(f"CSV file is empty: {path}")
    if not header or any(not name for name in header):
        raise ValueError(f"CSV header contains an empty column name: {path}")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"CSV header contains duplicate columns: {', '.join(duplicates)}")
    requested = (x, *columns)
    missing = [name for name in requested if name not in header]
    if missing:
        available = ", ".join(header)
        raise ValueError(f"unknown CSV column(s) {', '.join(missing)}; available: {available}")
    try:
        data = np.loadtxt(
            path, delimiter=",", skiprows=1,
            usecols=tuple(header.index(name) for name in requested),
            dtype=float, ndmin=2,
        )
    except ValueError as exc:
        raise ValueError(f"cannot read numeric data from {path}: {exc}") from None
    if data.shape[0] == 0:
        raise ValueError(f"CSV file has a header but no data rows: {path}")
    x_values = data[:, 0]
    if not np.all(np.isfinite(x_values)):
        raise ValueError(f"x column {x!r} contains non-finite values")
    return x_values, [data[:, index] for index in range(1, data.shape[1])]


def _header(path: Path) -> tuple[str, ...] | None:
    """The first row of a CSV file, or ``None`` when the file is empty."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        row = next(csv.reader(stream), None)
    return None if row is None else tuple(row)


def _result_csv(files: Sequence[str | Path], columns: tuple[str, ...], x: str) -> Path:
    tables = [Path(path) for path in files if Path(path).suffix.lower() == ".csv"]
    headers = {path: _header(path) or () for path in tables if path.is_file()}
    requested = {x, *columns}
    matches = [path for path, header in headers.items() if requested <= set(header)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"plot columns occur together in more than one result CSV: {names}")
    locations = {
        column: [path.name for path, header in headers.items() if column in header]
        for column in columns
    }
    absent = [column for column, found in locations.items() if not found]
    if absent:
        raise ValueError(
            f"plot column(s) not recorded: {', '.join(absent)}; enable the corresponding "
            "simulation output if they are plant or controller signals"
        )
    detail = "; ".join(f"{column}: {', '.join(locations[column])}" for column in columns)
    raise ValueError(f"plot columns must come from one result CSV ({detail})")


def _series_values(values: Sequence[Any] | None, count: int, what: str,
                   default: Sequence[Any]) -> tuple[Any, ...]:
    if values is None:
        return tuple(default[index % len(default)] for index in range(count))
    result = (values,) if isinstance(values, str) else tuple(values)
    if len(result) != count:
        raise ValueError(f"{what} must contain exactly {count} value(s)")
    return result


def _labels(values: Mapping[str, str] | Sequence[str] | None,
            columns: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return columns
    if isinstance(values, Mapping):
        unknown = sorted(set(values) - set(columns))
        if unknown:
            raise ValueError(f"labels contain unknown columns: {', '.join(unknown)}")
        return tuple(values.get(column, column) for column in columns)
    result = tuple(values)
    if len(result) != len(columns):
        raise ValueError(f"labels must contain exactly {len(columns)} value(s)")
    return result


def _tex_escape(value: str) -> str:
    """Escape an automatically generated plain-text label for external LaTeX."""
    return value.translate(_LATEX_ESCAPES)


def _text(value: str, use_tex: bool) -> str:
    """Keep explicit math/LaTeX intact and escape ordinary text for external LaTeX."""
    return (_tex_escape(value) if use_tex and "$" not in value and "\\" not in value
            else value)


def _time_scale(values: np.ndarray, unit: str) -> tuple[np.ndarray, str, float]:
    if unit not in ("auto", *_TIME_FACTORS):
        raise ValueError("time_unit must be 'auto', 's', 'ms', or 'us'")
    if unit == "auto":
        extent = float(np.max(np.abs(values)))
        unit = "s" if extent >= 1.0 or extent == 0.0 else "ms" if extent >= 1e-3 else "us"
    factor = _TIME_FACTORS[unit]
    return values * factor, unit, factor


def _limits(values: Sequence[float] | None, what: str) -> tuple[float, float] | None:
    if values is None:
        return None
    result = tuple(float(value) for value in values)
    if len(result) != 2 or not all(map(math.isfinite, result)) or result[0] >= result[1]:
        raise ValueError(f"{what} must be two finite, increasing values")
    return result


def _render_pdf(
    output: Path,
    x_values: np.ndarray,
    y_values: list[np.ndarray],
    *,
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    x: str,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    width: float,
    height: float,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    colors: tuple[str, ...],
    linestyles: tuple[str, ...],
    linewidth: float,
    grid: bool,
    legend: bool,
    time_unit: str,
    tex_mode: str,
) -> None:
    matplotlib = _matplotlib()
    from matplotlib.figure import Figure
    from matplotlib.ticker import ScalarFormatter

    if tex_mode == "pgf":
        from matplotlib.backends.backend_pgf import FigureCanvasPgf as Canvas
    else:
        from matplotlib.backends.backend_pdf import FigureCanvasPdf as Canvas

    use_tex = tex_mode != "mathtext"
    time_factor = 1.0
    shown_x = x_values
    shown_xlim = xlim
    shown_time_unit = "s"
    if x == "t":
        shown_x, shown_time_unit, time_factor = _time_scale(x_values, time_unit)
        if xlim is not None:
            shown_xlim = (xlim[0] * time_factor, xlim[1] * time_factor)
    style: dict[str, Any] = {
        "text.usetex": use_tex,
        "font.family": "serif" if use_tex else "cmr10",
        "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "grid.color": "#BBBBBB",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.borderaxespad": 0.4,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
    }
    if tex_mode == "latex":
        style["text.latex.preamble"] = r"\usepackage{amsmath,amssymb,amsfonts}"
    elif tex_mode == "pgf":
        style.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{amsmath,amssymb,amsfonts}",
        })

    with matplotlib.rc_context(style):
        figure = Figure(figsize=(width, height), layout="constrained")
        canvas = Canvas(figure)
        axis = figure.subplots()
        for index, (name, values) in enumerate(zip(columns, y_values)):
            axis.plot(
                shown_x, values,
                color=colors[index], linestyle=linestyles[index],
                linewidth=linewidth, label=_text(labels[index], use_tex),
            )
        if title:
            axis.set_title(_text(title, use_tex))
        if xlabel is not None:
            axis.set_xlabel(_text(xlabel, use_tex))
        else:
            axis.set_xlabel(_TIME_LABELS[shown_time_unit] if x == "t"
                            else (_tex_escape(x) if use_tex else x))
        if ylabel:
            axis.set_ylabel(_text(ylabel, use_tex))
        if shown_xlim is not None:
            axis.set_xlim(*shown_xlim)
        if ylim is not None:
            axis.set_ylim(*ylim)
        if grid:
            axis.grid(True)
        x_formatter = ScalarFormatter(useMathText=x != "t")
        y_formatter = ScalarFormatter(useMathText=True)
        if x == "t":
            x_formatter.set_scientific(False)
            x_formatter.set_useOffset(False)
        else:
            x_formatter.set_powerlimits((0, 0))
        y_formatter.set_powerlimits((0, 0))
        axis.xaxis.set_major_formatter(x_formatter)
        axis.yaxis.set_major_formatter(y_formatter)
        axis.tick_params(which="both", top=True, right=True)
        if legend:
            plot_legend = axis.legend()
            frame = plot_legend.get_frame()
            frame.set_edgecolor("black")
            frame.set_alpha(1.0)
            frame.set_linewidth(style["axes.linewidth"] * 0.7)
        font_logger = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")
        previous_level = font_logger.level
        font_logger.setLevel(logging.ERROR)
        try:
            canvas.print_pdf(str(output))
        finally:
            font_logger.setLevel(previous_level)


def plot_csv(
    csv_file: str | Path,
    columns: str | Sequence[str],
    *,
    output: str | Path | None = None,
    x: str = "t",
    labels: Mapping[str, str] | Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: str | float = "single",
    height: float | None = None,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    linewidth: float = 1.5,
    grid: bool = True,
    legend: bool = True,
    latex: bool | None = None,
    time_unit: str = "auto",
) -> Path:
    """Plot numeric CSV ``columns`` to an IEEE-style vector PDF.

    ``latex=None`` selects external LaTeX when the required executable is available and otherwise
    uses Matplotlib's Computer Modern math fonts.  Pass ``latex=False`` for a deterministic
    no-executable fallback, or ``latex=True`` to require an external LaTeX installation.
    """
    path = Path(csv_file)
    if x != "t" and time_unit != "auto":
        raise ValueError("time_unit applies only when x='t'")
    selected = _columns(columns)
    if x in selected:
        raise ValueError(f"horizontal-axis column {x!r} cannot also be a plotted waveform")
    label_values = _labels(labels, selected)
    x_values, y_values = _read_numeric_columns(path, x, selected)
    color_values = _series_values(colors, len(selected), "colors", PALETTE)
    style_values = _series_values(linestyles, len(selected), "linestyles", ("-", "--", "-.", ":"))
    x_limits = _limits(xlim, "xlim")
    y_limits = _limits(ylim, "ylim")

    if isinstance(width, str):
        try:
            width_value = _WIDTHS[width]
        except KeyError:
            raise ValueError("width must be 'single', 'double', or a positive number") from None
    else:
        width_value = float(width)
    height_value = width_value * (math.sqrt(5.0) - 1.0) / 2.0 if height is None else float(height)
    if not math.isfinite(width_value) or width_value <= 0:
        raise ValueError("width must be positive and finite")
    if not math.isfinite(height_value) or height_value <= 0:
        raise ValueError("height must be positive and finite")
    if not math.isfinite(linewidth) or linewidth <= 0:
        raise ValueError("linewidth must be positive and finite")

    destination = (Path(output) if output is not None
                   else path.with_name(f"fig_{path.stem}.pdf"))
    if destination.suffix.lower() != ".pdf":
        raise ValueError("plot output must have a .pdf suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)

    has_latex = shutil.which("latex") is not None
    has_dvipng = shutil.which("dvipng") is not None
    has_pdflatex = shutil.which("pdflatex") is not None
    if latex is True and not (has_latex and has_dvipng or has_pdflatex):
        raise RuntimeError("latex=True requires latex+dvipng or pdflatex")
    if latex is False:
        modes = ("mathtext",)
    elif has_latex and has_dvipng:
        modes = ("latex",) if latex is True else ("latex", "mathtext")
    elif has_pdflatex:
        modes = ("pgf",) if latex is True else ("pgf", "mathtext")
    else:
        modes = ("mathtext",)

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".pdf", dir=destination.parent, delete=False
    )
    temporary_file.close()
    temporary = Path(temporary_file.name)
    try:
        for index, mode in enumerate(modes):
            try:
                _render_pdf(
                    temporary, x_values, y_values,
                    columns=selected, labels=label_values, x=x,
                    title=title, xlabel=xlabel, ylabel=ylabel,
                    width=width_value, height=height_value,
                    xlim=x_limits, ylim=y_limits,
                    colors=color_values, linestyles=style_values,
                    linewidth=linewidth, grid=grid, legend=legend,
                    time_unit=time_unit,
                    tex_mode=mode,
                )
                temporary.replace(destination)
                return destination
            except (OSError, RuntimeError, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                if index + 1 == len(modes):
                    raise RuntimeError(f"could not render {destination}: {exc}") from exc
                warnings.warn(
                    f"external LaTeX rendering failed ({exc}); using Computer Modern fallback",
                    RuntimeWarning,
                )
    finally:
        temporary.unlink(missing_ok=True)
    raise RuntimeError(f"could not render {destination}")  # pragma: no cover


def plot_result(result: Any, columns: str | Sequence[str], **options: Any) -> Path:
    """Plot columns from the one CSV generated by ``result`` that contains all of them."""
    selected = _columns(columns)
    x = str(options.get("x", "t"))
    try:
        files = result.files
    except AttributeError:
        raise TypeError("result must provide the files written by a simulation") from None
    source = _result_csv(files, selected, x)
    return plot_csv(source, selected, **options)
