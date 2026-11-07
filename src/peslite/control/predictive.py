"""Model-predictive current loops for the ordinary converter control graph.

The numerical optimizers live in :mod:`.mpc`.  This module only adapts converter
measurements, references and duty commands to those pure algorithms; ADC, PWM,
protection, start-up and saved-state behavior remain owned by the normal graph host.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from ..solver.model import ConfigError
from ..solver.timing import TIME_EPS, PeriodicClock, control_clock_offset
from .blocks import abc2complex
from .loops import (ANGLE, DUTY_RATIO, ENABLE, FREQUENCY, I, I_AB, SI_CURRENT,
                    SI_VOLTAGE, V, V_AB, Loop, register_loop_type)
from .mpc import CCSMPC, FCSMPC, LinearModel, MPCResult


@dataclass(frozen=True, kw_only=True)
class _PredictiveCurrentParams:
    """Parameters shared by scalar and three-phase predictive current loops."""

    period: Optional[float] = None
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
    fallback_input: list[float] | None = field(default=None, metadata={"runtime": False})

    def __post_init__(self) -> None:
        if self.algorithm not in ("ccs", "fcs"):
            raise ConfigError("algorithm must be 'ccs' or 'fcs'")
        if self.fallback not in ("hold", "fixed", "block"):
            raise ConfigError("fallback must be 'hold', 'fixed' or 'block'")
        if not (math.isfinite(self.input_min) and math.isfinite(self.input_max)
                and 0.0 <= self.input_min < self.input_max <= 1.0):
            raise ConfigError("input bounds must satisfy 0 <= input_min < input_max <= 1")
        if (self.current_min is not None and not math.isfinite(self.current_min)):
            raise ConfigError("current_min must be finite")
        if (self.current_max is not None and not math.isfinite(self.current_max)):
            raise ConfigError("current_max must be finite")
        if (self.current_min is not None and self.current_max is not None
                and self.current_min > self.current_max):
            raise ConfigError("current_min must not exceed current_max")


class _PredictiveCurrentControl(Loop):
    """Shared prediction, optimization, fallback and named-state implementation."""

    n_current: ClassVar[int]
    n_input: ClassVar[int]
    outputs_from_state = True
    cpp_export = False
    carried = ("last_result",)
    observations = ("success", "objective", "iterations", "evaluations", "seconds",
                    "solves", "fallbacks", "delay")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        if unit.bridge.model not in ("switching", "pwm_averaging"):
            raise ConfigError(f"{type(self).__name__} requires switching or pwm_averaging")
        if unit.pwm.sync != "asynchronous" or unit.pwm.pattern_sync != "interval":
            raise ConfigError(
                f"{type(self).__name__} requires asynchronous interval PWM"
            )
        if unit.pwm.dead_time_compensation:
            raise ConfigError(
                f"{type(self).__name__} commands cannot use duty-ratio dead-time compensation"
            )
        self.T = cfg.period
        ratio = self.T / unit.pwm.load_period
        if not math.isclose(ratio, round(ratio), rel_tol=1e-9) or ratio < 1:
            raise ConfigError(
                f"{type(self).__name__}.period must be a whole multiple of the PWM load period"
            )
        clock = PeriodicClock(unit.pwm.load_period, unit.pwm.grid_offset)
        t0 = control_clock_offset(unit.ctrl, unit.pwm)
        index = clock.index_at_or_after(t0 + unit.ctrl.computation)
        if unit.ctrl.computation > 0:
            index = max(index, clock.index_after(t0))
        delay = clock.at(index) - t0
        if delay < -TIME_EPS or delay > self.T + TIME_EPS:
            raise ConfigError("predictive-control execution delay must be within one loop period")
        self.delay = min(self.T, max(0.0, delay))
        self._configure_plant(unit)
        self.actions = np.array(list(itertools.product((0.0, 1.0), repeat=self.n_input)))
        self.actions = self.actions[np.all(
            (self.actions >= cfg.input_min) & (self.actions <= cfg.input_max), axis=1,
        )]
        if cfg.algorithm == "fcs" and len(self.actions) == 0:
            raise ConfigError("input bounds exclude every finite control action")
        default = self.actions[0] if cfg.algorithm == "fcs" else [cfg.input_min] * self.n_input
        self.fixed_input = np.asarray(
            default if cfg.fallback_input is None else cfg.fallback_input, dtype=float,
        ).copy()
        if not self._admissible(self.fixed_input):
            raise ConfigError(
                "fallback_input must contain one finite admissible command per controlled leg"
            )
        self.previous_input = self.fixed_input.copy()
        self.previous_enabled = False
        self.solve_count = 0
        self.fallback_count = 0
        self.last_result: MPCResult | None = None
        self._last_diagnostics = self._diagnostics(None, False, 0.0)
        template = self._model(unit.dclink.vdc_ref, np.zeros(self.n_current), enabled=True)
        self.ccs = CCSMPC(
            template, cfg.prediction_horizon, cfg.control_horizon,
            tracking_weight=cfg.tracking_weight, input_weight=cfg.input_weight,
            rate_weight=cfg.rate_weight, input_bounds=(cfg.input_min, cfg.input_max),
            output_bounds=(
                -np.inf if cfg.current_min is None else cfg.current_min,
                np.inf if cfg.current_max is None else cfg.current_max,
            ), max_iterations=cfg.max_iterations, tolerance=cfg.tolerance,
            feasibility_tolerance=cfg.feasibility_tolerance,
        )
        if cfg.algorithm == "fcs":
            self.fcs = FCSMPC(
                self.actions, cfg.prediction_horizon, self._predict, self._stage_cost,
                control_horizon=cfg.control_horizon, constraint=self._constraint,
                max_evaluations=cfg.max_evaluations,
                feasibility_tolerance=cfg.feasibility_tolerance,
            )

    def _configure_plant(self, unit) -> None:
        raise NotImplementedError

    def _forecast(self, inputs: Mapping[str, Any]):
        raise NotImplementedError

    def _disturbance(self, inputs: Mapping[str, Any], k: int, enabled: bool) -> np.ndarray:
        raise NotImplementedError

    def _write_action(self, outputs: dict[str, Any], action: np.ndarray) -> None:
        raise NotImplementedError

    def _integral(self, duration):
        """Exact RL input integral, including the physical R = 0 limit."""
        return duration / self.L if self.R == 0 else -math.expm1(-self.R * duration / self.L) / self.R

    def _model(self, vdc, voltage_integral, *, enabled):
        n, m = self.n_current, self.n_input
        A = np.zeros((n + m, n + m))
        B = np.zeros((n + m, m))
        if enabled or self.delay == 0:
            A[:n, :n] = np.eye(n) * math.exp(-self.R * self.T / self.L)
        if enabled:
            A[:n, n:] = (math.exp(-self.R * (self.T - self.delay) / self.L)
                          * self._integral(self.delay) * vdc * self.voltage_map)
        B[:n] = self._integral(self.T - self.delay) * vdc * self.voltage_map
        B[n:] = np.eye(m)
        C = np.zeros((n, n + m))
        C[:, :n] = np.eye(n)
        return LinearModel(A, B, C, np.r_[-voltage_integral, np.zeros(m)])

    def prediction(self, t: float, inputs: Mapping[str, Any]):
        """Build affine prediction data without reading or mutating the running plant."""
        current, vdc, references = self._forecast(inputs)
        models = [
            self._model(
                vdc, self._disturbance(inputs, k, k > 0 or self.previous_enabled),
                enabled=k > 0 or self.previous_enabled,
            )
            for k in range(self.cfg.prediction_horizon)
        ]
        return np.r_[current, self.previous_input], references, models

    def _predict(self, state, action, k):
        model = self._models[k]
        return model.A @ state + model.B @ action + model.offset

    def _stage_cost(self, state, action, reference, previous, k):
        error = state[:self.n_current] - reference
        delta = action - previous
        return float(error @ self.ccs.Q @ error + action @ self.ccs.R @ action
                     + delta @ self.ccs.S @ delta)

    def _constraint(self, state, action, k):
        current = state[:self.n_current]
        residuals = []
        if self.cfg.current_min is not None:
            residuals.extend(current - self.cfg.current_min)
        if self.cfg.current_max is not None:
            residuals.extend(self.cfg.current_max - current)
        return np.asarray(residuals)

    def solve(self, t: float, inputs: Mapping[str, Any]) -> MPCResult:
        """Optimize once; a custom registered loop may override this method."""
        state, reference, self._models = self.prediction(t, inputs)
        if self.cfg.algorithm == "ccs":
            return self.ccs.solve(state, reference, self.previous_input, models=self._models)
        return self.fcs.solve(state, reference, self.previous_input)

    def _admissible(self, action) -> bool:
        action = np.asarray(action)
        if (action.shape != (self.n_input,) or action.dtype.kind not in "fiu"
                or not np.all(np.isfinite(action))
                or np.any(action < self.cfg.input_min) or np.any(action > self.cfg.input_max)):
            return False
        return (self.cfg.algorithm != "fcs"
                or bool(np.any(np.all(self.actions == action, axis=1))))

    def _diagnostics(self, result, valid, elapsed):
        return {
            "success": float(valid),
            "objective": (float(result.objective)
                          if result is not None and result.objective is not None else 0.0),
            "iterations": float(result.iterations) if result is not None else 0.0,
            "evaluations": float(result.evaluations) if result is not None else 0.0,
            "seconds": elapsed,
            "solves": float(self.solve_count),
            "fallbacks": float(self.fallback_count),
            "delay": self.delay,
        }

    def initial_outputs(self):
        outputs = {"enable": float(self.previous_enabled)}
        self._write_action(outputs, self.previous_input)
        return outputs

    def sample_into(self, t, inputs, outputs):
        elapsed = 0.0
        result = None
        enable = self.startup.active
        if enable:
            start = perf_counter()
            result = self.solve(t, inputs)
            elapsed = perf_counter() - start
            self.solve_count += 1
            valid = result.success and result.action is not None and self._admissible(result.action)
            if result.success and not valid:
                result = replace(
                    result, success=False, action=None, status="invalid_solution",
                    message="Optimizer returned an inadmissible actuator command",
                )
            if valid:
                action = np.asarray(result.action, dtype=float).copy()
            else:
                self.fallback_count += 1
                action = (self.previous_input.copy()
                          if self.cfg.fallback == "hold" and self._admissible(self.previous_input)
                          else self.fixed_input.copy())
                enable = self.cfg.fallback != "block"
        else:
            action, valid = self.fixed_input.copy(), False
        self.previous_input = action
        self.previous_enabled = enable
        self.last_result = result
        self._last_diagnostics = self._diagnostics(result, valid, elapsed)
        self._write_action(outputs, action)
        outputs["enable"] = float(enable)

    def sample(self, t, inputs):
        outputs = self.initial_outputs()
        self.sample_into(t, inputs, outputs)
        return outputs

    def get_state(self):
        return {
            **{f"previous_input.{i}": float(value)
               for i, value in enumerate(self.previous_input)},
            "previous_enabled": self.previous_enabled,
            "solve_count": self.solve_count,
            "fallback_count": self.fallback_count,
        }

    @property
    def discrete_state_names(self):
        names = ("previous_enabled", "solve_count", "fallback_count")
        if self.cfg.algorithm == "fcs":
            return tuple(f"previous_input.{i}" for i in range(self.n_input)) + names
        return names

    @property
    def linearization_state_names(self):
        return tuple(f"previous_input.{i}" for i in range(self.n_input)) + ("previous_enabled",)

    def periodic_state(self, values):
        return {name: value for name, value in values.items()
                if name not in ("solve_count", "fallback_count")}

    def set_state(self, values):
        unknown = set(values) - set(self.get_state())
        if unknown:
            raise KeyError(f"{type(self).__name__} has no states {sorted(unknown)}")
        for key, value in values.items():
            if key.startswith("previous_input."):
                if (not math.isfinite(value) or value < self.cfg.input_min
                        or value > self.cfg.input_max):
                    raise ValueError("previous input must be a finite admissible duty ratio")
                self.previous_input[int(key.rsplit(".", 1)[1])] = value
            elif key == "previous_enabled":
                if value not in (0, 1):
                    raise ValueError("previous_enabled must be 0 or 1")
                self.previous_enabled = bool(value)
            else:
                if not math.isfinite(value) or value < 0 or value != int(value):
                    raise ValueError(f"{key} must be a nonnegative integer")
                setattr(self, key, int(value))

    def summary(self):
        return {
            "algorithm": self.cfg.algorithm,
            "solves": self.solve_count,
            "fallbacks": self.fallback_count,
            "delay": self.delay,
            "status": None if self.last_result is None else self.last_result.status,
        }

    def observe(self):
        return dict(self._last_diagnostics)


@register_loop_type
class PredictiveCurrentControl(_PredictiveCurrentControl):
    """Scalar inductor-current MPC which directly supplies a converter duty ratio."""

    @dataclass(frozen=True, kw_only=True)
    class Params(_PredictiveCurrentParams):
        type: str = "predictive_current_control"

    type = "predictive_current_control"
    role = "current"
    n_current, n_input = 1, 1
    inputs = {
        "reference": SI_CURRENT,
        "current": SI_CURRENT,
        "voltage": SI_VOLTAGE,
        "vin": SI_VOLTAGE,
    }
    outputs = {
        "duty": DUTY_RATIO,
        "enable": ENABLE,
    }

    def _configure_plant(self, unit):
        bridge = unit.bridge
        if (bridge.type != "half_bridge" or not bridge.upper_controlled
                or not bridge.lower_controlled or bridge.reverse_port
                or bridge.external_port != "midpoint"):
            raise ConfigError(
                "PredictiveCurrentControl requires a synchronous Buck half bridge"
            )
        if unit.filter is None:
            raise ConfigError("PredictiveCurrentControl requires an inductive filter")
        self.L = unit.filter.l_f
        self.R = unit.filter.r_f + bridge.on_resistance
        self.voltage_map = np.ones((1, 1))

    def _forecast(self, inputs):
        references = np.full(
            (self.cfg.prediction_horizon, 1),
            inputs["reference"] * self.startup.value,
        )
        return np.array([inputs["current"]]), inputs["vin"], references

    def _disturbance(self, inputs, k, enabled):
        duration = self.T if enabled else self.T - self.delay
        return np.array([inputs["voltage"] * self._integral(duration)])

    def _write_action(self, outputs, action):
        outputs["duty"] = float(action[0])


@register_loop_type
class ThreePhasePredictiveCurrentControl(_PredictiveCurrentControl):
    """Three-phase current MPC which directly supplies all bridge duty ratios."""

    @dataclass(frozen=True, kw_only=True)
    class Params(_PredictiveCurrentParams):
        type: str = "three_phase_predictive_current_control"

    type = "three_phase_predictive_current_control"
    role = "modulation"
    n_current, n_input = 2, 3
    inputs = {
        "v": V_AB,
        "i": I_AB,
        "frame": ANGLE,
        "omega": FREQUENCY,
        "id_ref": I,
        "iq_ref": I,
        "u_dc": V,
    }
    outputs = {
        "d_a": DUTY_RATIO,
        "d_b": DUTY_RATIO,
        "d_c": DUTY_RATIO,
        "enable": ENABLE,
    }

    def _configure_plant(self, unit):
        if unit.bridge.type != "three_phase_bridge":
            raise ConfigError(
                "ThreePhasePredictiveCurrentControl requires a two-level three-phase bridge"
            )
        self.L = unit.ac_filter.l_f
        self.R = unit.ac_filter.r_f + unit.bridge.on_resistance
        vectors = [abc2complex(row) for row in np.eye(3)]
        self.voltage_map = np.array([
            [value.real for value in vectors],
            [value.imag for value in vectors],
        ])
        self.i_base = unit.base.i_phase_peak
        self.v_base = unit.base.v_phase_peak
        self.vdc_base = unit.dc_base.v

    def _forecast(self, inputs):
        current = inputs["i"] * self.i_base
        target = complex(inputs["id_ref"], inputs["iq_ref"]) * self.i_base * self.startup.value
        frame, omega = inputs["frame"], inputs["omega"]
        references = np.array([
            [value.real, value.imag]
            for value in (
                target * cmath.exp(1j * (frame + omega * (k + 1) * self.T))
                for k in range(self.cfg.prediction_horizon)
            )
        ])
        return np.array([current.real, current.imag]), inputs["u_dc"] * self.vdc_base, references

    def _disturbance(self, inputs, k, enabled):
        duration = self.T if enabled else self.T - self.delay
        start = k * self.T + (0.0 if enabled else self.delay)
        omega = inputs["omega"]
        denominator = self.R + 1j * omega * self.L
        factor = (duration / self.L if denominator == 0 else
                  (cmath.exp(1j * omega * duration)
                   - math.exp(-self.R * duration / self.L)) / denominator)
        integral = inputs["v"] * self.v_base * cmath.exp(1j * omega * start) * factor
        return np.array([integral.real, integral.imag])

    def _write_action(self, outputs, action):
        outputs["d_a"], outputs["d_b"], outputs["d_c"] = map(float, action)
