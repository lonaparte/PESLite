"""PESLite command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import keyword
import math
import subprocess
import sys
import sysconfig
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Callable, Mapping

from .assembly.params import Params, _load_case, _unit_items, dumps, to_dict
from .solver.model import ConfigError
from .solver.simulation import Simulation, SimulationResult, json_value, validate_model

# --------------------------------------------------------- command-line entry

def _example_configs_dir() -> Path:
    """Return the root examples directory, from a checkout or an installed wheel."""
    source = Path(__file__).resolve().parents[2] / "examples"
    if source.is_dir():
        return source
    # data-files live below the installation prefix, outside the import package. A ``--target``
    # installation puts them below the metadata root; a normal environment uses sysconfig's data root.
    for distribution_name in ("peslite", "peslite-core"):
        try:
            dist = distribution(distribution_name)
        except PackageNotFoundError:
            continue
        roots = (Path(dist.locate_file("")).resolve(), Path(sysconfig.get_path("data")).resolve())
        for root in roots:
            candidate = root / "share" / "peslite" / "examples"
            if candidate.is_dir():
                return candidate
    return source


_EXAMPLE_CONFIGS = _example_configs_dir()
_RESULTS = Path.cwd() / "output"
_CONFIG_SUFFIXES = (".pes", ".cir")


def _config_path(value: str | None) -> Path:
    if value is None:
        default = _EXAMPLE_CONFIGS / "gfl-example.pes"
        if default.is_file():
            return default.resolve()
        configs = sorted(
            p for p in _EXAMPLE_CONFIGS.iterdir()
            if p.is_file() and p.suffix.lower() in _CONFIG_SUFFIXES
        )
        if not configs:
            raise FileNotFoundError("no bundled example configurations available")
        return configs[0]
    path = Path(value).expanduser()
    candidates = [path]
    if not path.suffix:
        candidates += [path.with_suffix(s) for s in _CONFIG_SUFFIXES]
    example = _EXAMPLE_CONFIGS / path
    candidates.append(example)
    if not path.suffix:
        candidates += [example.with_suffix(s) for s in _CONFIG_SUFFIXES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"configuration not found: {value}")

def parse_override(text: str):
    key, _, value = text.partition("=")
    if not key or not _:
        raise argparse.ArgumentTypeError("expected key.path=value")
    for cast in (int, float):
        try:
            v = cast(value)
            if cast is int and str(v) != value:
                continue
            return key, v
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    if value.lower() in ("null", "none"):
        return key, None
    return key, value


def _solver_choice_is_set(overrides: Mapping[str, Any]) -> bool:
    """Whether ``--set`` selected a solver family or method for this run."""
    return any(
        path in ("simulation.solver.type", "simulation.solver.method")
        for path in overrides
    )


@dataclass
class _CommandOutcome:
    code: int
    payload: dict[str, Any]


class _CommandFailure(Exception):
    def __init__(self, code: int, error_code: str, message: str, path: str | None = None,
                 config: str | Path | None = None) -> None:
        super().__init__(message)
        self.code, self.error_code, self.path, self.config = code, error_code, path, config


def _absolute(path: str | Path) -> str:
    return str(Path(path).resolve())


def _payload(command: str, *, config: str | Path | None = None, status: str = "ok",
             code: int = 0, output=None, simulation=None, diagnostics=(), error=None,
             **extra) -> dict[str, Any]:
    return {
        "schema": "peslite.cli/v1",
        "command": command,
        "status": status,
        "exit_code": code,
        "config": _absolute(config) if config is not None else None,
        "output": output,
        "simulation": simulation,
        "diagnostics": list(diagnostics),
        "physical_validation": "not_performed",
        "error": error,
        **extra,
    }


def _summary_diagnostics(p: Params, summary: Mapping[str, Any]) -> list[dict[str, str]]:
    diagnostics = []
    diagnostics.extend({
        "severity": "warning",
        "code": "initial_state_adjustment",
        "message": (
            f"initial state {item['name']!r} requested {item['requested']!r} "
            f"but started at {item['actual']!r}"
        ),
    } for item in summary.get("initial_state_adjustments", ()))
    for unit_name, _cfg in _unit_items(p.units):
        trip_cause = summary.get(f"{unit_name}.trip_cause")
        alarms = summary.get(f"{unit_name}.alarms") or []
        messages = ([f"tripped: {trip_cause}"] if trip_cause else []) + list(alarms)
        diagnostics.extend({"severity": "warning", "code": "unit_warning",
                            "message": f"{unit_name}: {message}"}
                           for message in messages)
    return diagnostics


def _run_exit(summary: Mapping[str, Any]) -> tuple[int, str | None]:
    reason = summary.get("stop_reason")
    if summary.get("status") == "timed_out":
        return 5, "timeout"
    if reason and str(reason).startswith("RHS evaluation limit"):
        return 5, "resource_limit"
    if summary.get("status") == "diverged":
        return 4, "simulation_diverged"
    if reason:
        return 4, "simulation_stopped"
    return 0, None


def _run_outcome(config: Path, out: Path, result: SimulationResult) -> _CommandOutcome:
    summary = result.summary
    simulation_status = summary.get("status", "completed")
    reason = summary.get("stop_reason")
    code, error_code = _run_exit(summary)
    diagnostics = _summary_diagnostics(result.params, summary)
    output = {
        "directory": _absolute(out),
        "summary": _absolute(out / "summary.json"),
        "files": [_absolute(path) for path in result.files],
    }
    simulation = {
        "status": simulation_status,
        "t_start": summary.get("t_start"),
        "t_end": result.params.simulation.t_end,
        "t_stop": summary.get("t_stop"),
        "wall_time": result.wall_time,
        "rhs_evaluations": result.n_rhs,
        "tripped": bool(summary.get("tripped")),
    }
    error = None if code == 0 else {"code": error_code, "message": reason, "path": None}
    return _CommandOutcome(code, _payload(
        "run", config=config, status="ok" if code == 0 else "error", code=code,
        output=output, simulation=simulation, diagnostics=diagnostics, error=error,
    ))


_BRIDGE_PRESETS = {"switching": ("fixed", "RK4"), "pwm_averaging": ("fixed", "RK4"),
                   "averaging": ("adaptive", "DP45")}


def _command_parser(*args, **kwargs) -> argparse.ArgumentParser:
    """Build a command parser with exact, stable option names."""
    return argparse.ArgumentParser(*args, allow_abbrev=False, **kwargs)


def _add_configuration_arguments(ap: argparse.ArgumentParser, *, optional: bool = False) -> None:
    ap.add_argument("config", nargs="?" if optional else None,
                    help="simulation file path or bundled example name"
                         + ("; default: gfl-example" if optional else ""))
    ap.add_argument("--set", action="append", default=[], type=parse_override, metavar="PATH=VALUE",
                    help="override a dotted parameter path (repeatable)")
    ap.add_argument("--initial", default=None, metavar="FILE",
                    help="initial values: a states.csv row or a simulation file's simulation.initial block")
    ap.add_argument("--initial-time", type=float, default=None, metavar="T",
                    help="with a states.csv: start from the row at time T instead of the last row")
    bridge = ap.add_mutually_exclusive_group()
    bridge.add_argument("--switching", action="store_true",
                        help="exact-switching bridge with fixed-step RK4; --set has priority")
    bridge.add_argument("--pwm-averaging", action="store_true",
                        help="PWM-period-averaged bridge with fixed-step RK4; --set has priority")
    bridge.add_argument("--averaging", action="store_true",
                        help="ideal averaged bridge with adaptive DP45; --set has priority")


def _parse_command_args(ap: argparse.ArgumentParser, argv=None):
    args = ap.parse_args(argv)
    if args.initial_time is not None and args.initial is None:
        ap.error("--initial-time selects a row of a states.csv: add --initial FILE")
    return args


def _load_command_params(args, extra_overrides: Mapping[str, Any] | None = None
                         ) -> tuple[Path, Params, str | None, str]:
    try:
        config = _config_path(args.config)
    except FileNotFoundError as exc:
        if args.json:
            raise _CommandFailure(2, "input_error", str(exc), config=args.config) from None
        raise
    set_overrides = dict(args.set)
    overrides = {**set_overrides, **(extra_overrides or {})}
    try:
        p, interface_text = _load_case(
            config, initial=args.initial, initial_time=args.initial_time, **overrides,
        )
    except ConfigError as exc:
        if args.json:
            raise _CommandFailure(2, "configuration_error", str(exc), config=config) from None
        raise
    model = next((flag for flag in _BRIDGE_PRESETS if getattr(args, flag)), None)
    if model is None:
        return config, p, None, interface_text
    effective_models = []
    for name, cfg in _unit_items(p.units):
        path = f"units.{name}.bridge.model"
        selected = cfg.bridge.model
        if path not in set_overrides:
            try:
                p = p.replace(**{path: model})
            except ConfigError as exc:
                warnings.warn(
                    f"--{model.replace('_', '-')} does not apply to units.{name}: "
                    f"{exc}; keeping bridge.model = {selected!r}",
                    stacklevel=3,
                )
            else:
                selected = model
        effective_models.append(selected)
    solver_type, method = (_BRIDGE_PRESETS[model] if all(
        selected == model for selected in effective_models
    ) else _BRIDGE_PRESETS["switching"])
    changes = {}
    if not _solver_choice_is_set(set_overrides):
        changes.update({
            "simulation.solver.type": solver_type,
            "simulation.solver.method": method,
        })
        if solver_type == "adaptive" and not any(
                path == "simulation.solver.subsystems"
                or path.startswith("simulation.solver.subsystems.")
                for path in set_overrides):
            changes["simulation.solver.subsystems"] = {}
    return config, p.replace(**changes) if changes else p, model, interface_text


def _run_main(argv=None, *, prog: str = "peslite run") -> int | _CommandOutcome:
    """Run a simulation file and save its states, summary and configured signals."""
    ap = _command_parser(
        prog=prog, description=_run_main.__doc__,
        epilog="commands: run, create, validate, inspect, export, debug; a simulation file without a command implies run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_configuration_arguments(ap, optional=True)
    ap.add_argument("--out", default=None,
                    help="output directory (default: output/<file name>, or "
                         "a bridge-mode suffix with --switching/--pwm-averaging/--averaging)")
    ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
    ap.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                    help="stop after this much wall time")
    ap.add_argument("--max-rhs", type=int, default=None, metavar="COUNT",
                    help="stop after this many solver RHS evaluations")
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None,
                    help="show or hide the terminal progress bar")
    ap.add_argument("--watch", action="append", default=None, metavar="NAME",
                    help="append a state or plant/control value to the progress bar; repeatable, "
                         "or use NAME,NAME")
    ap.add_argument("--report", action="store_true",
                    help="generate <simulation name>_report.json using the optional analysis addon")
    plots = ap.add_argument_group("IEEE-style PDF plot (after simulation)")
    plots.add_argument("--plot", nargs="+", default=None, metavar="NAME",
                       help="plot these recorded columns against time; all must belong to one result CSV")
    presentation = [  # their destinations are the keywords of SimulationResult.plot
        plots.add_argument("--plot-output", dest="output", default=None, metavar="PDF",
                           help="PDF path relative to the run directory (default: fig_<table>.pdf)"),
        plots.add_argument("--plot-x", dest="x", default="t", metavar="NAME",
                           help="horizontal-axis column (default: t)"),
        plots.add_argument("--plot-time-unit", dest="time_unit", choices=("auto", "s", "ms", "us"), default="auto",
                           help="time-axis unit (default: choose s, ms, or us from the range)"),
        plots.add_argument("--plot-title", dest="title", default=None, metavar="TEXT"),
        plots.add_argument("--plot-xlabel", dest="xlabel", default=None, metavar="TEXT"),
        plots.add_argument("--plot-ylabel", dest="ylabel", default=None, metavar="TEXT"),
        plots.add_argument("--plot-label", dest="labels", action="append", default=None, metavar="TEXT",
                           help="legend label for each waveform, in --plot order; repeatable"),
        plots.add_argument("--plot-color", dest="colors", action="append", default=None, metavar="COLOR",
                           help="Matplotlib color for each waveform, in --plot order; repeatable"),
        plots.add_argument("--plot-style", dest="linestyles", action="append", default=None, metavar="STYLE",
                           help="line style for each waveform, in --plot order; repeatable"),
        plots.add_argument("--plot-width", dest="width", choices=("single", "double"), default="single",
                           help="IEEE single- or double-column width (default: single)"),
        plots.add_argument("--plot-height", dest="height", type=float, default=None, metavar="INCHES"),
        plots.add_argument("--plot-linewidth", dest="linewidth", type=float, default=1.5, metavar="POINTS"),
        plots.add_argument("--plot-xlim", dest="xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX")),
        plots.add_argument("--plot-ylim", dest="ylim", type=float, nargs=2, default=None, metavar=("MIN", "MAX")),
        plots.add_argument("--plot-no-grid", dest="grid", action="store_false", help="disable the background grid"),
        plots.add_argument("--plot-no-legend", dest="legend", action="store_false", help="disable the legend"),
    ]
    args = _parse_command_args(ap, argv)

    plot_options = {action.dest: getattr(args, action.dest) for action in presentation}
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0.0):
        ap.error("--timeout must be finite and positive")
    if args.max_rhs is not None and args.max_rhs <= 0:
        ap.error("--max-rhs must be positive")
    if args.json and (args.progress is not None or args.watch):
        ap.error("--json cannot be combined with terminal progress options")
    if args.plot is None and any(plot_options[action.dest] != action.default for action in presentation):
        ap.error("plot presentation options require --plot NAME [NAME ...]")
    overrides = {}
    if args.json:
        overrides["simulation.progress.enable"] = False
    if args.progress is not None:
        overrides["simulation.progress.enable"] = args.progress
    if args.watch is not None:
        overrides["simulation.progress.watch"] = [name for value in args.watch
                                                   for name in value.split(",") if name]
    try:
        config, p, model, interface_text = _load_command_params(args, overrides)
    except FileNotFoundError as exc:
        ap.error(str(exc))
    if (args.progress is None and not p.simulation.progress.enable
            and bool(getattr(sys.stdout, "isatty", lambda: False)())):
        p = p.replace(**{"simulation.progress.enable": True})
    if args.watch and not p.simulation.progress.enable:
        ap.error("--watch prints on the progress bar: add --progress")
    if args.plot:
        try:
            from .assembly.addons import import_addon
            _plot_result = import_addon("plotting", "function").plot_result
        except (ImportError, RuntimeError) as exc:
            ap.error(f"--plot: {exc}")
    sim = _assemble_command(p, config, args.json, interface_text)
    suffix = "-" + model.replace("_", "-") if model is not None else ""
    out = Path(args.out) if args.out else _RESULTS / f"{config.stem}{suffix}"
    reporter = None
    if args.report:
        try:
            from .assembly.addons import AddonUnavailableError, import_addon
            reporter = import_addon("online_analysis", "function").small_signal_report
        except AddonUnavailableError:
            pass
        if not callable(reporter):
            warnings.warn("--report: the online stability analysis addon is unavailable; "
                          "continuing the simulation without a report", RuntimeWarning, stacklevel=2)
            reporter = None
    limits = {name: value for name, value in (("timeout", args.timeout), ("max_rhs", args.max_rhs))
              if value is not None}
    if reporter is not None:
        reporter(sim, out_dir=out, name=config.stem, info={"config": str(config)}, **limits)
        r = sim.result
    else:
        r = sim.run(out_dir=out, info={"config": str(config)}, name=config.stem, **limits)
    s = r.summary
    diagnostics = _summary_diagnostics(p, s)
    if not args.json:
        for diagnostic in diagnostics:
            print(f"warning: {diagnostic['message']}", file=sys.stderr)
    if args.plot:
        output, plot_options["output"] = plot_options["output"], None
        if output:
            output = Path(output) if Path(output).is_absolute() else out / output
            plot_options["output"] = output if output.suffix else output.with_suffix(".pdf")
        try:
            r.plot(args.plot, **plot_options)
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            ap.error(str(exc))
    if args.json:
        return _run_outcome(config, out, r)
    t_stop = s.get("t_stop", 0)
    status = s.get("status", "completed").replace("_", " ").title()
    trip = ", tripped" if s.get("tripped") else ""
    print(
        f"{status}: t = {t_stop:.6g} s, wall = {r.wall_time:.1f} s, "
        f"RHS = {r.n_rhs}{trip}, output = {out}"
    )
    return _run_exit(s)[0]


def _assemble_command(p: Params, config: Path, json_output: bool,
                      interface_text: str = "") -> Simulation:
    try:
        return validate_model(p, interface_text=interface_text)
    except ConfigError as exc:
        if json_output:
            raise _CommandFailure(3, "assembly_error", str(exc), config=config) from None
        raise


def _open_file(path: Path) -> None:
    """Open *path* with the operating system's associated application."""
    import os
    import shlex
    import subprocess

    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    editor = (os.environ.get("VISUAL", "").strip()
              or os.environ.get("EDITOR", "").strip())
    if editor:
        try:
            command = shlex.split(editor)
        except ValueError as exc:
            raise OSError(f"invalid editor command: {exc}") from exc
        options = {}
    else:
        command = ["open" if sys.platform == "darwin" else "xdg-open"]
        options = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
            "text": True,
            "timeout": 10,
        }
    try:
        result = subprocess.run([*command, str(path)], **options)
    except subprocess.SubprocessError as exc:
        raise OSError(str(exc)) from exc
    if result.returncode:
        detail = (result.stderr.strip() if result.stderr else
                  f"exit code {result.returncode}")
        raise OSError(f"{command[0]}: {detail}")


def _create_main(argv=None) -> int | _CommandOutcome:
    """Create a simulation file or project add-on from a bundled template."""
    ap = _command_parser(prog="peslite create", description=_create_main.__doc__)
    ap.add_argument("name", help="file name to create (.pes, .cir or .py)")
    ap.add_argument("--open", action="store_true",
                    help="open the created file with the configured editor or default application")
    ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
    args = ap.parse_args(argv)
    name = Path(args.name)
    if name.name != args.name or name.suffix not in {".pes", ".cir", ".py"}:
        ap.error("provide a file name ending in .pes, .cir or .py in the current directory")
    target = Path.cwd() / name
    templates = _EXAMPLE_CONFIGS / "templates"
    template = templates / "template.pes"
    replacements = ()
    if name.suffix == ".py":
        class_name = "".join(part[:1].upper() + part[1:] for part in name.stem.split("_"))
        if (not name.stem.isascii() or not name.stem.isidentifier()
                or keyword.iskeyword(name.stem) or not class_name.isidentifier()):
            ap.error("Python file names must be ASCII Python identifiers")
        kind = "component" if name.stem == "component" else "controller"
        template = templates / f"{kind}.py"
        target = Path.cwd() / "PESaddons" / ("components" if kind == "component" else "control") / name
        replacements = ((f"Template{kind.title()}", class_name),
                        (f"template_{kind}", name.stem.lower()))
    try:
        text = template.read_text(encoding="utf-8")
        for original, replacement in replacements:
            text = text.replace(original, replacement)
        if name.suffix == ".cir":
            from .assembly.cir import convert_text
            text = convert_text(text, name.with_suffix(".pes"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except OSError as exc:
        ap.error(str(exc))
    if args.open:
        try:
            _open_file(target)
        except OSError as exc:
            warnings.warn(f"could not open {target}: {exc}", stacklevel=2)
    if args.json:
        return _CommandOutcome(0, _payload("create", output={
            "directory": _absolute(target.parent), "summary": None, "files": [_absolute(target)],
        }))
    print("created", target)
    return 0


def _validate_main(argv=None) -> int | _CommandOutcome:
    """Validate and assemble one configuration without running it."""
    ap = _command_parser(prog="peslite validate", description=_validate_main.__doc__)
    _add_configuration_arguments(ap)
    ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
    args = _parse_command_args(ap, argv)
    try:
        config, p, _model, interface_text = _load_command_params(args)
    except FileNotFoundError as exc:
        ap.error(str(exc))
    simulation = _assemble_command(p, config, args.json, interface_text)
    validation = {
        "status": "valid",
        "units": list(simulation.units),
        "state_count": len(simulation.state_names()),
        "solver": {"type": p.simulation.solver.type, "method": p.simulation.solver.method},
    }
    if args.json:
        return _CommandOutcome(0, _payload("validate", config=config, validation=validation))
    print(
        f"Valid: {config}  units = {len(validation['units'])}, "
        f"states = {validation['state_count']}, "
        f"solver = {validation['solver']['type']}/{validation['solver']['method']}"
    )
    return 0


def _inspect_main(argv=None) -> int | _CommandOutcome:
    """Inspect parameters, states, electrical groups and the port-Hamiltonian structure without running."""
    ap = _command_parser(prog="peslite inspect", description=_inspect_main.__doc__)
    _add_configuration_arguments(ap)
    ap.add_argument("--resolved", action="store_true", help="show the complete resolved configuration")
    ap.add_argument("--states", action="store_true", help="show the generated state names")
    ap.add_argument("--ph-report", action="store_true", help="show the port-Hamiltonian structure report")
    ap.add_argument("--groups", action="store_true", help="show the automatic electrical partition")
    ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
    args = _parse_command_args(ap, argv)
    try:
        config, p, _model, interface_text = _load_command_params(args)
    except FileNotFoundError as exc:
        ap.error(str(exc))
    selected = {"resolved": args.resolved, "states": args.states,
                "ph_report": args.ph_report, "groups": args.groups}
    if not any(selected.values()):
        selected = dict.fromkeys(selected, True)
    simulation = None
    if selected["states"] or selected["ph_report"] or selected["groups"]:
        assembly_p = p
        if selected["ph_report"] and p.simulation.energy_check == "off":
            assembly_p = p.replace(**{"simulation.energy_check": "warn"})
        simulation = _assemble_command(assembly_p, config, args.json, interface_text)
    values = {}
    if selected["resolved"]:
        values["resolved"] = to_dict(p)
    if selected["states"]:
        values["states"] = simulation.state_names()
        values["state_count"] = len(values["states"])
    if selected["ph_report"]:
        values["ph_report"] = simulation.ph_report.to_dict()
    if selected["groups"]:
        values["groups"] = [circuit.grouping_report() for circuit in simulation.system.circuits]
    if args.json:
        return _CommandOutcome(0, _payload("inspect", config=config, **values))
    views = {
        "resolved": ("Resolved configuration", lambda: dumps(p, interface_text).rstrip()),
        "states": ("States", lambda: "\n".join(values["states"])),
        "ph_report": ("Port-Hamiltonian report", lambda: str(simulation.ph_report)),
        "groups": ("Electrical groups", lambda: json.dumps(values["groups"], indent=2)),
    }
    sections = [(title, render()) for name, (title, render) in views.items() if selected[name]]
    print(sections[0][1] if len(sections) == 1 else
          "\n\n".join(title + ":\n" + body for title, body in sections))
    return 0


def _export_main(argv=None) -> int | _CommandOutcome:
    """Export a standalone C++ simulator or an Imperix ELF executable."""
    ap = _command_parser(prog="peslite export", description=_export_main.__doc__)
    _add_configuration_arguments(ap)
    ap.add_argument("parameters", nargs="*", metavar="PARAM",
                    help="variable parameter path, common or all")
    ap.add_argument("--out", default=None, help="export directory")
    ap.add_argument("--run", action="store_true",
                    help="compile and run the exported C++ simulator")
    ap.add_argument("--imperix", action="store_true", help="export an Imperix ELF")
    ap.add_argument("--imperix-config", metavar="FILE", help="ADC/PWM bindings; default: PESFILE.imperix.json")
    ap.add_argument("--imperix-sdk", metavar="DIRECTORY", help="installed official Imperix SDK")
    ap.add_argument("--imperix-source-only", action="store_true", help="generate sources without building an ELF")
    ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
    args = _parse_command_args(ap, argv)
    if args.imperix and args.run:
        ap.error("--run is available only for C++ export")
    if not args.imperix and (args.imperix_config or args.imperix_sdk or args.imperix_source_only):
        ap.error("Imperix options require --imperix")
    try:
        config, p, model, interface_text = _load_command_params(args)
    except FileNotFoundError as exc:
        ap.error(str(exc))
    simulation = _assemble_command(p, config, args.json, interface_text)
    suffix = "-" + model.replace("_", "-") if model is not None else ""
    out = Path(args.out) if args.out else Path("export") / f"{config.stem}{suffix}"
    export_format = "imperix" if args.imperix else "cpp"
    options = {}
    if args.imperix:
        options = dict(target_config=args.imperix_config or config.with_suffix(".imperix.json"),
                       sdk=args.imperix_sdk, build=not args.imperix_source_only)
    try:
        result = simulation.export(export_format, out, name=f"{config.stem}{suffix}",
                                   variables=args.parameters, **options)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if args.json:
            raise _CommandFailure(4, "export_error", str(exc), config=config) from None
        ap.error(str(exc))
    diagnostics = [{"severity": item.get("severity", "warning"),
                    "code": item.get("code", "export_warning"), "message": item["message"]}
                   for item in result.diagnostics]
    run = None
    run_code, run_error = 0, None
    if args.run:
        from .assembly.cpp import compile_source

        run_out = _RESULTS / f"{config.stem}{suffix}"
        try:
            build = compile_source(result.files[0])
            completed = subprocess.run(
                [str(build.executable), str(run_out)], check=False
            )
            if completed.returncode:
                raise RuntimeError(
                    f"exported simulator exited with code {completed.returncode}"
                )
            summary_path = run_out / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            if args.json:
                raise _CommandFailure(4, "export_run_error", str(exc), config=config) from None
            ap.error(str(exc))
        run_code, error_code = _run_exit(summary)
        diagnostics.extend(_summary_diagnostics(p, summary))
        run_error = (None if run_code == 0 else
                     {"code": error_code, "message": summary.get("stop_reason"), "path": None})
        run = {
            "directory": _absolute(run_out),
            "summary": _absolute(summary_path),
            "files": [_absolute(path) for path in sorted(run_out.iterdir()) if path.is_file()],
            "build_cache_hit": build.cache_hit,
        }
    if args.json:
        output = {"directory": _absolute(out), "summary": None,
                  "files": [_absolute(path) for path in result.files]}
        simulation = (None if run is None else {
            "status": summary.get("status", "completed"),
            "t_start": summary.get("t_start"),
            "t_end": p.simulation.t_end,
            "t_stop": summary.get("t_stop"),
            "wall_time": summary.get("wall_time"),
            "rhs_evaluations": summary.get("n_rhs"),
            "tripped": bool(summary.get("tripped")),
        })
        return _CommandOutcome(run_code, _payload(
            "export", config=config, status="ok" if run_code == 0 else "error",
            code=run_code, output=output, simulation=simulation,
            diagnostics=diagnostics, error=run_error,
            export={"format": export_format, "run": run},
        ))
    for diagnostic in diagnostics:
        print("warning:", diagnostic["message"], file=sys.stderr)
    print("wrote", ", ".join(str(path) for path in result.files))
    if run is not None:
        source = "cached" if build.cache_hit else "compiled"
        print(f"ran {source} simulator; output: {run_out}")
    return run_code


def _debug_main(argv=None) -> int | _CommandOutcome:
    """Run a file-backed control-graph debug overlay and write structured evidence."""
    try:
        from .addons.functions.debug import (
            DebugPlan, debug_history, debug_root, execute_debug, parse_operation, read_debug_plan,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "peslite.addons.functions.debug":
            raise ConfigError("the debug command requires the standard peslite installation") from None
        raise

    values = list(argv or ())
    listing = bool(values and values[0] == "ls")
    if listing:
        values.pop(0)
        ap = _command_parser(prog="peslite debug ls", description="List debug runs for a case.")
        ap.add_argument("config", help="simulation file path or bundled example name")
        ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
        args = ap.parse_args(values)
        try:
            config = _config_path(args.config)
        except FileNotFoundError as exc:
            ap.error(str(exc))
        runs = debug_history(config, _RESULTS)
        records = [{"id": path.name, "directory": _absolute(path),
                    "plan": _absolute(path / "debug.pesdbg"),
                    "report": _absolute(path / "report.json")}
                   for path in runs]
        if args.json:
            return _CommandOutcome(0, _payload("debug", config=config, runs=records))
        if records:
            print("\n".join(f"{item['id']}  {item['directory']}" for item in records))
        else:
            print(f"No debug runs for {config.stem}.")
        return 0

    ap = _command_parser(prog="peslite debug", description=_debug_main.__doc__)
    _add_configuration_arguments(ap)
    ap.add_argument("operations", nargs="*", metavar="OPERATION",
                    help="quoted +/− connection or ~ parameter operation")
    origin = ap.add_mutually_exclusive_group()
    origin.add_argument("--plan", metavar="FILE", help="read this .pesdbg plan")
    origin.add_argument("--from", dest="from_run", metavar="RUN",
                        help="inherit one run ID or .pesdbg path")
    origin.add_argument("--fresh", action="store_true", help="do not inherit the latest run")
    ap.add_argument("--state-policy", choices=("continue", "freeze", "tracking"),
                    help="state handling for bypassed loop producers")
    ap.add_argument("--probe", action="append", default=[], metavar="NAME=PATH",
                    help="observe a typed graph source or input; repeatable")
    ap.add_argument("--analysis", action="append", default=[], metavar="NAME",
                    help="add small_signal, or retain a named analysis configured in .pesdbg")
    ap.add_argument("--timeout", type=float, default=None, metavar="SECONDS")
    ap.add_argument("--max-rhs", type=int, default=None, metavar="COUNT")
    ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
    args = _parse_command_args(ap, values)
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0.0):
        ap.error("--timeout must be finite and positive")
    if args.max_rhs is not None and args.max_rhs <= 0:
        ap.error("--max-rhs must be positive")
    try:
        config, p, _model, interface_text = _load_command_params(args, {
            "simulation.progress.enable": False,
        })
    except FileNotFoundError as exc:
        ap.error(str(exc))
    history = debug_history(config, _RESULTS)
    plan_path = None
    if args.plan:
        plan_path = Path(args.plan)
    elif args.from_run:
        candidate = Path(args.from_run)
        plan_path = (candidate if candidate.suffix == ".pesdbg"
                     else debug_root(config, _RESULTS) / candidate / "debug.pesdbg")
    elif not args.fresh and history:
        plan_path = history[0] / "debug.pesdbg"
    plan = read_debug_plan(plan_path) if plan_path is not None else DebugPlan()
    if plan.model:
        planned = Path(plan.model).expanduser()
        if not planned.is_absolute() and plan_path is not None:
            planned = plan_path.resolve().parent / planned
        if planned.resolve() != config.resolve():
            ap.error(f"debug plan model {plan.model!r} does not match {config}")
    missing_analysis = [name for name in args.analysis
                        if name != "small_signal" and name not in plan.analyses]
    if missing_analysis:
        ap.error(
            "--analysis needs a configured .pesdbg entry except for small_signal: "
            + ", ".join(missing_analysis)
        )
    operations = tuple(parse_operation(value, where="command line") for value in args.operations)
    probes = {}
    for written in args.probe:
        alias, separator, path = written.partition("=")
        if not separator or not alias or not path:
            ap.error("--probe requires NAME=PATH")
        probes[alias] = path
    plan = plan.with_updates(model=str(config.resolve()), operations=operations,
                             probes=probes, analyses=tuple(args.analysis))
    if args.state_policy is not None:
        plan = dataclasses.replace(plan, state_policy=args.state_policy)
    directory = plan_path.resolve().parent if plan_path is not None else config.resolve().parent
    try:
        run_dir, report = execute_debug(
            config, p, interface_text, plan, plan_directory=directory, output=_RESULTS,
            timeout=args.timeout, max_rhs=args.max_rhs,
        )
    except ConfigError as exc:
        if args.json:
            raise _CommandFailure(2, "configuration_error", str(exc), config=config) from None
        raise
    except OSError as exc:
        if args.json:
            raise _CommandFailure(2, "input_error", str(exc), config=config) from None
        raise
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        if args.json:
            raise _CommandFailure(4, "debug_error", str(exc), config=config) from None
        raise
    status = report["status"]
    code = 0 if status == "passed" else 6 if status == "failed" else 4
    if args.json:
        return _CommandOutcome(code, _payload(
            "debug", config=config, status="ok" if code == 0 else "error", code=code,
            output={"directory": _absolute(run_dir),
                    "summary": _absolute(run_dir / "report.json"),
                    "files": [_absolute(run_dir / "debug.pesdbg"),
                              _absolute(run_dir / "report.json")]},
            debug={"status": status, "checks": report["checks"]},
        ))
    print(f"Debug {status}: {run_dir}")
    return code


def _warning_diagnostics(caught) -> list[dict[str, str]]:
    return [{
        "severity": "warning",
        "code": "warning",
        "message": str(item.message),
    } for item in caught]


def _error_outcome(command: str, code: int, error_code: str, message: str, *,
                   config: str | Path | None = None, path: str | None = None) -> _CommandOutcome:
    return _CommandOutcome(code, _payload(
        command, config=config, status="error", code=code,
        error={"code": error_code, "message": message, "path": path},
    ))


def _json_command(command: str, invoke: Callable[[], int | _CommandOutcome]) -> int:
    stdout, stderr = io.StringIO(), io.StringIO()
    with warnings.catch_warnings(record=True) as caught, redirect_stdout(stdout), redirect_stderr(stderr):
        warnings.simplefilter("always")
        try:
            outcome = invoke()
        except _CommandFailure as exc:
            outcome = _error_outcome(command, exc.code, exc.error_code, str(exc),
                                     config=exc.config, path=exc.path)
        except SystemExit as exc:
            code = int(exc.code or 0)
            if code == 0:
                outcome = _CommandOutcome(0, _payload(command))
            else:
                lines = [line.strip() for line in stderr.getvalue().splitlines() if line.strip()]
                message = lines[-1].partition("error: ")[2] if lines else "invalid command"
                outcome = _error_outcome(command, code, "command_error", message)
        except (FileNotFoundError, OSError) as exc:
            outcome = _error_outcome(command, 2, "input_error", str(exc))
        except ConfigError as exc:
            outcome = _error_outcome(command, 2, "configuration_error", str(exc))
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            outcome = _error_outcome(command, 4, "simulation_error", str(exc))
        except Exception as exc:
            outcome = _error_outcome(command, 70, "internal_error", str(exc))
    if not isinstance(outcome, _CommandOutcome):
        outcome = _CommandOutcome(int(outcome), _payload(
            command, status="ok" if outcome == 0 else "error", code=int(outcome)
        ))
    outcome.payload["diagnostics"].extend(_warning_diagnostics(caught))
    print(json.dumps(json_value(outcome.payload), indent=2, allow_nan=False))
    return outcome.code


def _convert_main(argv=None) -> int:
    """Convert between .pes files and .cir netlists."""
    ap = _command_parser(prog="peslite convert", description=_convert_main.__doc__)
    ap.add_argument("source", help=".pes or .cir file to read")
    ap.add_argument("output", nargs="?",
                    help="file to write (default: the other format next to the source)")
    ap.add_argument("--json", action="store_true", help="write one machine-readable result object")
    args = ap.parse_args(argv)
    from .assembly.cir import convert_text, pes_text, read_cir_text
    from .assembly.syntax import SimulationFileError
    source = Path(args.source).expanduser()
    if source.suffix.lower() not in _CONFIG_SUFFIXES:
        ap.error("the source must be a .pes or .cir file")
    output = Path(args.output).expanduser() if args.output else source.with_suffix(
        ".pes" if source.suffix.lower() == ".cir" else ".cir")
    if output.suffix.lower() not in _CONFIG_SUFFIXES:
        ap.error("the output must be a .pes or .cir file")
    text = source.read_text(encoding="utf-8")
    try:
        if output.suffix.lower() == ".cir":
            result = convert_text(text, source)
        elif source.suffix.lower() == ".pes":
            result = text
        else:
            result = pes_text(*read_cir_text(text, source))
    except SimulationFileError as exc:
        raise ConfigError(str(exc)) from None
    output.write_text(result, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


def main(argv=None) -> int:
    """Create, run, validate, inspect, export, convert or debug a PESLite simulation."""
    values = list(sys.argv[1:] if argv is None else argv)
    commands = {"run", "create", "validate", "inspect", "export", "debug", "convert"}
    explicit_command = bool(values and values[0] in commands)
    command = values.pop(0) if explicit_command else "run"
    json_output = "--json" in values
    if command == "run":
        invoke = lambda: _run_main(values, prog="peslite run" if explicit_command else "peslite")
    elif command == "validate":
        invoke = lambda: _validate_main(values)
    elif command == "inspect":
        invoke = lambda: _inspect_main(values)
    elif command == "create":
        invoke = lambda: _create_main(values)
    elif command == "debug":
        invoke = lambda: _debug_main(values)
    elif command == "convert":
        invoke = lambda: _convert_main(values)
    else:
        invoke = lambda: _export_main(values)
    if json_output:
        return _json_command(command, invoke)
    try:
        return int(invoke())
    except (ConfigError, OSError) as exc:
        print(f"peslite {command}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

