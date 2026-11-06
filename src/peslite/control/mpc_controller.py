"""Sampled predictive current control through the ordinary converter command boundary.

Prediction uses SI filter currents and dimensionless leg duties. The physical plant,
ADC, protection, computation delay and PWM registers remain owned by the converter.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np

from ..solver.model import ConfigError
from ..solver.timing import TIME_EPS, PeriodicClock, control_clock_offset
from .blocks import abc2complex
from .controller import BridgeControlOutput, ControlInterface, ControlOutput, Startup
from .mpc import CCSMPC, FCSMPC, LinearModel, MPCResult
from .profiles import ReferenceParams, register_control_type


@dataclass(frozen=True)
class MPCCurrentReferences:
    """Scalar converter current reference in amperes."""

    current: float = 0.0


@dataclass(frozen=True)
class MPCOptions:
    """Predictive current-controller configuration; generic models use the Python API."""

    algorithm: str = field(default="ccs", metadata={"runtime": False})
    prediction_horizon: int = field(default=3, metadata={"runtime": False})
    control_horizon: int | None = field(default=None, metadata={"runtime": False})
    tracking_weight: Any = 1.0
    input_weight: Any = 0.0
    rate_weight: Any = 0.01
    input_min: float = 0.0
    input_max: float = 1.0
    current_min: float | None = None
    current_max: float | None = None
    max_iterations: int = 100
    max_evaluations: int = 10000
    tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-7
    fallback: str = "hold"
    fallback_input: list[float] | None = None


def _validate_options(ctrl, where):
    if ctrl.loops or ctrl.connections or ctrl.outputs:
        raise ConfigError(f"{where}: MPC uses standalone commands, without a control graph")
    options = ctrl.options
    if options.algorithm not in ("ccs", "fcs"):
        raise ConfigError(f"{where}.options.algorithm must be 'ccs' or 'fcs'")
    if options.fallback not in ("hold", "fixed", "block"):
        raise ConfigError(f"{where}.options.fallback must be 'hold', 'fixed' or 'block'")
    if not (math.isfinite(options.input_min) and math.isfinite(options.input_max)
            and 0.0 <= options.input_min < options.input_max <= 1.0):
        raise ConfigError(f"{where}.options: input bounds must satisfy 0 <= min < max <= 1")
    for name in ("current_min", "current_max"):
        value = getattr(options, name)
        if value is not None and not math.isfinite(value):
            raise ConfigError(f"{where}.options.{name} must be finite")
    if (options.current_min is not None and options.current_max is not None
            and options.current_min > options.current_max):
        raise ConfigError(f"{where}.options: current_min must not exceed current_max")


class MPCCurrentController:
    """CCS/FCS current control with a fixed PWM-load delay and shared lifecycle.

    Override :meth:`solve` to supply a custom algorithm without changing the runner.
    A custom CCS backend can instead be passed as ``backend``. State affecting future
    decisions is exposed through the existing named-state interface, not a second
    simulation or a private copy of the electrical plant.
    """

    continuous = False

    def __init__(self, cfg, topology=None, *, backend=None):
        self.is_ac = cfg.type == "vsc"
        self.n_current, self.n_input = (2, 3) if self.is_ac else (1, 1)
        self.topology = topology
        self.backend = backend
        self.startup = Startup(cfg.ctrl.period, run=False)
        self.periods = {"mpc": cfg.ctrl.period}
        self.logging = True
        self.last_log = {}
        self.solve_count = self.fallback_count = 0
        self.previous_enabled = False
        self._configure(cfg)
        self.previous_input = self.fixed_input.copy()
        self.last_result = None

    def _configure(self, cfg):
        _validate_options(cfg.ctrl, "ctrl")
        if cfg.bridge.model not in ("switching", "pwm_averaging"):
            raise ConfigError("MPC requires switching or pwm_averaging")
        if self.is_ac:
            if cfg.bridge.type != "three_phase_bridge":
                raise ConfigError("MPC current control requires a two-level three-phase bridge")
            filt = cfg.ac_filter
            vectors = [abc2complex(row) for row in np.eye(3)]
            self.voltage_map = np.array([[z.real for z in vectors], [z.imag for z in vectors]])
            if cfg.protection.frequency.enable or cfg.protection.rocof.enable:
                raise ConfigError("MPC current control has no frequency estimator; frequency and ROCOF protection require one")
        else:
            if (cfg.bridge.type != "half_bridge" or not cfg.bridge.upper_controlled
                    or not cfg.bridge.lower_controlled or cfg.bridge.reverse_port
                    or cfg.bridge.external_port != "midpoint"):
                raise ConfigError("MPC current control requires a synchronous Buck half bridge")
            filt = cfg.filter
            if filt is None:
                raise ConfigError("MPC current control requires an inductive filter")
            self.voltage_map = np.ones((1, 1))
        if cfg.pwm.sync != "asynchronous" or cfg.pwm.pattern_sync != "interval":
            raise ConfigError("MPC fixed-delay prediction requires asynchronous interval PWM")
        if cfg.pwm.dead_time_compensation:
            raise ConfigError("MPC commands must not be modified by duty-ratio dead-time compensation")
        self.T = cfg.ctrl.period
        ratio = self.T / cfg.pwm.load_period
        if not math.isclose(ratio, round(ratio), rel_tol=1e-9) or ratio < 1:
            raise ConfigError("MPC requires ctrl.period to be a whole multiple of the PWM load period")
        clock = PeriodicClock(cfg.pwm.load_period, cfg.pwm.grid_offset)
        t0 = control_clock_offset(cfg.ctrl, cfg.pwm)
        index = clock.index_at_or_after(t0 + cfg.ctrl.computation)
        if cfg.ctrl.computation > 0:
            index = max(index, clock.index_after(t0))
        delay = clock.at(index) - t0
        if delay < -TIME_EPS or delay > self.T + TIME_EPS:
            raise ConfigError("MPC fixed execution delay must be between zero and one control period")
        self.delay = min(self.T, max(0.0, delay))  # Canonical clock endpoint roundoff only.
        self.L = filt.l_f
        self.R = filt.r_f + cfg.bridge.on_resistance
        self.p = cfg
        self.options = cfg.ctrl.options
        self.omega = cfg.ctrl.references.omega if self.is_ac else None
        self.theta = 0.0
        opt = self.options
        self.actions = np.array(list(itertools.product((0.0, 1.0), repeat=self.n_input)))
        self.actions = self.actions[np.all((self.actions >= opt.input_min) & (self.actions <= opt.input_max), axis=1)]
        if opt.algorithm == "fcs" and len(self.actions) == 0:
            raise ConfigError("MPC input bounds exclude every finite control action")
        default_input = self.actions[0] if opt.algorithm == "fcs" else [opt.input_min] * self.n_input
        self.fixed_input = np.asarray(
            default_input if opt.fallback_input is None else opt.fallback_input, dtype=float,
        ).copy()
        if not self._admissible(self.fixed_input):
            raise ConfigError("MPC fallback_input must be a finite admissible command with one value per leg")
        # A template fixes dimensions and validates weights/constraints once per configuration.
        template = self._model(cfg.dclink.vdc_ref, np.zeros(self.n_current), enabled=True)
        self.ccs = CCSMPC(
            template, opt.prediction_horizon, opt.control_horizon,
            tracking_weight=opt.tracking_weight, input_weight=opt.input_weight,
            rate_weight=opt.rate_weight, input_bounds=(opt.input_min, opt.input_max),
            output_bounds=(
                -np.inf if opt.current_min is None else opt.current_min,
                np.inf if opt.current_max is None else opt.current_max,
            ), max_iterations=opt.max_iterations, tolerance=opt.tolerance,
            feasibility_tolerance=opt.feasibility_tolerance, backend=self.backend,
        )
        if opt.algorithm == "fcs":
            # The callbacks consume only prediction arrays prepared from the current ADC sample.
            self.fcs = FCSMPC(
                self.actions, opt.prediction_horizon, self._predict, self._stage_cost,
                control_horizon=opt.control_horizon, constraint=self._constraint,
                max_evaluations=opt.max_evaluations,
                feasibility_tolerance=opt.feasibility_tolerance,
            )

    def _integral(self, duration):
        """Exact RL input integral, including the physical R = 0 limit."""
        return duration / self.L if self.R == 0 else -math.expm1(-self.R * duration / self.L) / self.R

    def _model(self, vdc, voltage_integral, *, enabled):
        n, m = self.n_current, self.n_input
        A = np.zeros((n + m, n + m))
        B = np.zeros((n + m, m))
        # A blocked filtered branch opens and resets its current, including a
        # coincident block-register load that occurs after the ADC conversion.
        if enabled or self.delay == 0:
            A[:n, :n] = np.eye(n) * math.exp(-self.R * self.T / self.L)
        if enabled:
            A[:n, n:] = (math.exp(-self.R * (self.T - self.delay) / self.L)
                          * self._integral(self.delay) * vdc * self.voltage_map)
        B[:n] = self._integral(self.T - self.delay) * vdc * self.voltage_map
        B[n:] = np.eye(m)
        C = np.zeros((n, n + m))
        C[:, :n] = np.eye(n)
        offset = np.r_[-voltage_integral, np.zeros(m)]
        return LinearModel(A, B, C, offset)

    def prediction(self, t, measurement):
        """Build pure affine prediction data from ADC measurements and existing filter parameters."""
        count = self.options.prediction_horizon
        if self.is_ac:
            current = np.array([measurement.i_c.real, measurement.i_c.imag])
            vdc, voltage = measurement.u_dc, measurement.u_g
            self.theta = cmath.phase(voltage)
            target = complex(self.p.ctrl.references.id_ref_pu, self.p.ctrl.references.iq_ref_pu)
            target *= self.p.base.i_phase_peak * self.startup.value
            references = np.array([
                [z.real, z.imag] for z in (
                    target * cmath.exp(1j * (self.theta + self.omega * (k + 1) * self.T))
                    for k in range(count)
                )
            ])
        else:
            current = np.array([measurement.current])
            vdc, voltage = measurement.vin, measurement.voltage
            references = np.full((count, 1), self.p.ctrl.references.current * self.startup.value)
        models = []
        for k in range(count):
            enabled = k > 0 or self.previous_enabled
            duration = self.T if enabled else self.T - self.delay
            if self.is_ac:
                # Exact RL response to a sinusoidal terminal-voltage extrapolation.
                w = self.omega
                start = k * self.T + (0.0 if enabled else self.delay)
                denominator = self.R + 1j * w * self.L
                factor = (duration / self.L if denominator == 0 else
                          (cmath.exp(1j * w * duration) - math.exp(-self.R * duration / self.L)) / denominator)
                integral = voltage * cmath.exp(1j * w * start) * factor
                forcing = np.array([integral.real, integral.imag])
            else:
                forcing = np.array([voltage * self._integral(duration)])
            models.append(self._model(vdc, forcing, enabled=enabled))
        return np.r_[current, self.previous_input], references, models

    def _predict(self, state, action, k):
        model = self._models[k]
        return model.A @ state + model.B @ action + model.offset

    def _stage_cost(self, state, action, reference, previous, k):
        error = state[:self.n_current] - reference
        delta = action - previous
        return float(error @ self.ccs.Q @ error + action @ self.ccs.R @ action + delta @ self.ccs.S @ delta)

    def _constraint(self, state, action, k):
        current = state[:self.n_current]
        residuals = []
        if self.options.current_min is not None:
            residuals.extend(current - self.options.current_min)
        if self.options.current_max is not None:
            residuals.extend(self.options.current_max - current)
        return np.asarray(residuals)

    def solve(self, t, measurement) -> MPCResult:
        """Optimize once; subclasses may replace this method with a custom MPC algorithm."""
        state, reference, self._models = self.prediction(t, measurement)
        if self.options.algorithm == "ccs":
            return self.ccs.solve(state, reference, self.previous_input, models=self._models)
        return self.fcs.solve(state, reference, self.previous_input)

    def _admissible(self, action):
        action = np.asarray(action)
        if (action.shape != (self.n_input,) or action.dtype.kind not in "fiu"
                or not np.all(np.isfinite(action))
                or np.any(action < self.options.input_min) or np.any(action > self.options.input_max)):
            return False
        return self.options.algorithm != "fcs" or bool(np.any(np.all(self.actions == action, axis=1)))

    def __call__(self, t, measurement):
        self.startup.advance()
        elapsed = 0.0
        result = None
        gates = self.startup.active
        if gates:
            start = perf_counter()
            result = self.solve(t, measurement)
            elapsed = perf_counter() - start
            self.solve_count += 1
            valid = result.success and result.action is not None and self._admissible(result.action)
            if result.success and not valid:
                result = replace(result, success=False, action=None, status="invalid_solution",
                                 message="Optimizer returned an inadmissible actuator command")
            if valid:
                action = np.asarray(result.action, dtype=float).copy()
            else:
                self.fallback_count += 1
                action = (self.previous_input.copy() if self.options.fallback == "hold"
                          and self._admissible(self.previous_input) else self.fixed_input.copy())
                gates = self.options.fallback != "block"
        else:
            action = self.fixed_input.copy()
            valid = False
        self.previous_input = action
        self.previous_enabled = gates
        self.last_result = result
        if self.logging:
            self.last_log = {
                "mpc.success": float(valid),
                "mpc.objective": float(result.objective) if result is not None and result.objective is not None else math.nan,
                "mpc.iterations": float(result.iterations) if result is not None else 0.0,
                "mpc.evaluations": float(result.evaluations) if result is not None else 0.0,
                "mpc.seconds": elapsed,
                "mpc.solves": float(self.solve_count),
                "mpc.fallbacks": float(self.fallback_count),
                "mpc.delay": self.delay,
            }
        output = dict(log=self.last_log if self.logging else None, gates=gates,
                      startup_complete=self.startup.complete)
        if self.is_ac:
            return ControlOutput(action, theta=self.theta, omega=self.omega, **output)
        return BridgeControlOutput(duty=float(action[0]), **output)

    def command(self, run, ramp=0.0):
        self.startup.command(run, ramp)

    def initialize(self):
        """Construction and parameter events already prepare the optimizer."""

    def retune(self, cfg, paths, t):
        self._configure(cfg)

    def control_inputs(self, current, voltage, vin, power=None):
        return {"meas.current": current, "meas.voltage": voltage, "meas.vin": vin}

    def get_state(self):
        return {
            **{f"startup.{key}": value for key, value in self.startup.get_state().items()},
            **{f"previous_input.{i}": float(value) for i, value in enumerate(self.previous_input)},
            "previous_enabled": self.previous_enabled,
            "solve_count": self.solve_count,
            "fallback_count": self.fallback_count,
        }

    @property
    def discrete_state_names(self):
        return tuple(name for name in self.get_state()
                     if self.options.algorithm == "fcs" or not name.startswith("previous_input."))

    @property
    def linearization_state_names(self):
        return tuple(self.periodic_state(self.get_state()))

    def periodic_state(self, values):
        """Diagnostic counters do not affect the electrical recurrence or prediction."""
        return {name: value for name, value in values.items()
                if name not in ("solve_count", "fallback_count")}

    def set_state(self, values):
        unknown = set(values) - self.get_state().keys()
        if unknown:
            raise KeyError(f"MPC has no states {sorted(unknown)}")
        self.startup.set_state({key[8:]: value for key, value in values.items() if key.startswith("startup.")})
        for key, value in values.items():
            if key.startswith("previous_input."):
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("MPC previous input must be a finite duty ratio")
                self.previous_input[int(key.split(".")[1])] = value
            elif key == "previous_enabled":
                if value not in (0, 1):
                    raise ValueError("MPC previous_enabled must be 0 or 1")
                self.previous_enabled = bool(value)
            elif key in ("solve_count", "fallback_count"):
                if not math.isfinite(value) or value < 0 or value != int(value):
                    raise ValueError(f"MPC {key} must be a nonnegative integer")
                setattr(self, key, int(value))

    def set_logging(self, enabled):
        self.logging = bool(enabled)

    def log_names(self):
        return ("mpc.success", "mpc.objective", "mpc.iterations", "mpc.evaluations",
                "mpc.seconds", "mpc.solves", "mpc.fallbacks", "mpc.delay")

    def summary(self):
        return {"mpc": {"algorithm": self.options.algorithm, "solves": self.solve_count,
                        "fallbacks": self.fallback_count, "delay": self.delay,
                        "status": None if self.last_result is None else self.last_result.status}}


for _name, _unit, _references, _bridge in (
    ("mpc_dc", "dc_converter", MPCCurrentReferences, "half_bridge"),
    ("mpc_vsc", "vsc", ReferenceParams, "three_phase_bridge"),
):
    register_control_type(
        _name, references=_references, unit_types=frozenset((_unit,)),
        options=MPCOptions, requires_loops=False,
        supported_models=frozenset(("switching", "pwm_averaging")),
        validator=_validate_options, default_bridge=_bridge,
        interface=ControlInterface(inputs={}, outputs={}),
        controller_factory=MPCCurrentController,
    )
