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
from ..solver.timing import TIME_EPS, control_actuation_clock, control_clock_offset
from .blocks import abc2complex
from .loops import (ANGLE, DUTY_RATIO, ENABLE, FREQUENCY, I, I_AB, SI_CURRENT,
                    SI_VOLTAGE, V, V_AB, Loop, register_loop_type)
from .mpc import CCSMPC, FCSMPC, LinearModel, MPCResult, _QuadraticStageCost


class _RLCurrentPrediction:
    """Exact discrete affine prediction for one RL current path per coordinate."""

    def __init__(self, inductance, resistance, period, delay, voltage_map, *, blocked_min=None):
        self.L, self.R = inductance, resistance
        self.period, self.delay = period, delay
        self.voltage_map = np.asarray(voltage_map, dtype=float)
        self.voltage_map.setflags(write=False)
        self.n_current, self.n_input = self.voltage_map.shape
        self.decay = math.exp(-self.R * self.period / self.L)
        self.blocked_min = (None if blocked_min is None else
                            np.broadcast_to(blocked_min, (self.n_current,)).copy())
        if self.blocked_min is not None:
            self.blocked_min.setflags(write=False)

    def integral(self, duration):
        """Exact RL input integral, including the physical R = 0 limit."""
        return (duration / self.L if self.R == 0
                else -math.expm1(-self.R * duration / self.L) / self.R)

    def model(self, vdc, current, disturbance_before, disturbance_after, previous_applied):
        n, m = self.n_current, self.n_input
        A = np.zeros((n + m, n + m))
        B = np.zeros((n + m, m))
        before_decay = math.exp(-self.R * self.delay / self.L)
        after = self.period - self.delay
        after_decay = math.exp(-self.R * after / self.L)
        prior_state = np.eye(n) * before_decay
        prior_offset = -np.asarray(disturbance_before)
        if (not previous_applied and self.delay > 0.0
                and self.blocked_min is not None):
            blocked = prior_state @ current + prior_offset <= self.blocked_min
            prior_state[blocked] = 0.0
            prior_offset[blocked] = self.blocked_min[blocked]
        A[:n, :n] = after_decay * prior_state
        if previous_applied:
            A[:n, n:] = after_decay * self.integral(self.delay) * vdc * self.voltage_map
        B[:n] = self.integral(after) * vdc * self.voltage_map
        B[n:] = np.eye(m)
        C = np.zeros((n, n + m))
        C[:, :n] = np.eye(n)
        offset = after_decay * prior_offset - np.asarray(disturbance_after)
        return LinearModel(A, B, C, np.r_[offset, np.zeros(m)])


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
        pwm_timed = unit.bridge.pwm_timed
        if pwm_timed and (
                unit.pwm.sync != "asynchronous" or unit.pwm.pattern_sync != "interval"):
            raise ConfigError(
                f"{type(self).__name__} requires asynchronous interval PWM"
            )
        if pwm_timed and unit.pwm.dead_time_compensation:
            raise ConfigError(
                f"{type(self).__name__} commands cannot use duty-ratio dead-time compensation"
            )
        self.T = cfg.period
        actuation = control_actuation_clock(unit.ctrl, unit.pwm, pwm_timed=pwm_timed)
        ratio = self.T / actuation.period
        if not math.isclose(ratio, round(ratio), rel_tol=1e-9) or ratio < 1:
            raise ConfigError(
                f"{type(self).__name__}.period must be a whole multiple of the actuation period"
            )
        t0 = control_clock_offset(unit.ctrl, unit.pwm, pwm_timed=pwm_timed)
        index = actuation.index_at_or_after(t0 + unit.ctrl.computation)
        if unit.ctrl.computation > 0:
            index = max(index, actuation.index_after(t0))
        delay = actuation.at(index) - t0
        if delay < -TIME_EPS or delay > self.T + TIME_EPS:
            raise ConfigError("predictive-control execution delay must be within one loop period")
        self.delay = min(self.T, max(0.0, delay))
        self.plant = self._configure_plant(unit)
        if cfg.algorithm == "fcs":
            actions = np.array(list(itertools.product((0.0, 1.0), repeat=self.n_input)))
            self.actions = actions[np.all(
                (actions >= cfg.input_min) & (actions <= cfg.input_max), axis=1,
            )]
            if len(self.actions) == 0:
                raise ConfigError("input bounds exclude every finite control action")
            default = self.actions[0]
        else:
            default = [cfg.input_min] * self.n_input
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
        if cfg.algorithm == "ccs":
            template = self.plant.model(
                unit.dclink.vdc_ref, np.zeros(self.n_current),
                np.zeros(self.n_current), np.zeros(self.n_current),
                previous_applied=True,
            )
            self.optimizer = CCSMPC(
                template, cfg.prediction_horizon, cfg.control_horizon,
                tracking_weight=cfg.tracking_weight, input_weight=cfg.input_weight,
                rate_weight=cfg.rate_weight, input_bounds=(cfg.input_min, cfg.input_max),
                output_bounds=(
                    -np.inf if cfg.current_min is None else cfg.current_min,
                    np.inf if cfg.current_max is None else cfg.current_max,
                ), max_iterations=cfg.max_iterations, tolerance=cfg.tolerance,
                feasibility_tolerance=cfg.feasibility_tolerance,
            )
        else:
            self._cost = _QuadraticStageCost(
                self.n_current, self.n_input,
                cfg.tracking_weight, cfg.input_weight, cfg.rate_weight,
            )
            self.optimizer = FCSMPC(
                self.actions, cfg.prediction_horizon, self._predict, self._stage_cost,
                control_horizon=cfg.control_horizon, constraint=self._constraint,
                max_evaluations=cfg.max_evaluations,
                feasibility_tolerance=cfg.feasibility_tolerance,
            )

    def _configure_plant(self, unit) -> _RLCurrentPrediction:
        raise NotImplementedError

    def _rl_plant(self, unit, inductance, resistance, voltage_map, *, blocked_min=None):
        """Build the shared RL predictor after checking its physical model boundary."""
        if unit.bridge.parallel_capacitance > 0.0:
            raise ConfigError(
                f"{type(self).__name__} requires a bridge without internal energy storage"
            )
        return _RLCurrentPrediction(
            inductance, resistance, self.T, self.delay, voltage_map,
            blocked_min=blocked_min,
        )

    def _forecast(self, inputs: Mapping[str, Any]):
        raise NotImplementedError

    def _disturbance(self, inputs: Mapping[str, Any], start: float,
                     duration: float) -> np.ndarray:
        raise NotImplementedError

    def _write_action(self, outputs: dict[str, Any], action: np.ndarray) -> None:
        raise NotImplementedError

    def prediction(self, t: float, inputs: Mapping[str, Any]):
        """Build affine prediction data without reading or mutating the running plant."""
        current, vdc, references = self._forecast(inputs)
        models = []
        for k in range(self.cfg.prediction_horizon):
            start = k * self.T
            models.append(self.plant.model(
                vdc,
                current,
                self._disturbance(inputs, start, self.delay),
                self._disturbance(inputs, start + self.delay, self.T - self.delay),
                previous_applied=k > 0 or self.previous_enabled,
            ))
        return np.r_[current, self.previous_input], references, models

    def _predict(self, state, action, k):
        model = self._models[k]
        return model.A @ state + model.B @ action + model.offset

    def _stage_cost(self, state, action, reference, previous, k):
        return self._cost.evaluate(state[:self.n_current], action, reference, previous)

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
            return self.optimizer.solve(
                state, reference, self.previous_input, models=self._models,
            )
        return self.optimizer.solve(state, reference, self.previous_input)

    def _admissible(self, action) -> bool:
        if not self._valid_history(action):
            return False
        action = np.asarray(action)
        if np.any(action < self.cfg.input_min) or np.any(action > self.cfg.input_max):
            return False
        return (self.cfg.algorithm != "fcs"
                or bool(np.any(np.all(self.actions == action, axis=1))))

    def _valid_history(self, action) -> bool:
        """Whether an action can represent a previously applied bridge command."""
        action = np.asarray(action)
        if (action.shape != (self.n_input,) or action.dtype.kind not in "fiu"
                or not np.all(np.isfinite(action))
                or np.any(action < 0.0) or np.any(action > 1.0)):
            return False
        return self.cfg.algorithm != "fcs" or bool(np.all((action == 0.0) | (action == 1.0)))

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
        action = self.previous_input.tolist()
        enabled = self.previous_enabled
        counters = {
            "solve_count": self.solve_count,
            "fallback_count": self.fallback_count,
        }
        for key, value in values.items():
            if key.startswith("previous_input."):
                action[int(key.rsplit(".", 1)[1])] = value
            elif key == "previous_enabled":
                if value not in (0, 1):
                    raise ValueError("previous_enabled must be 0 or 1")
                enabled = bool(value)
            else:
                if not math.isfinite(value) or value < 0 or value != int(value):
                    raise ValueError(f"{key} must be a nonnegative integer")
                counters[key] = int(value)
        action = np.asarray(action)
        if not self._valid_history(action):
            raise ValueError("previous input must be a valid finite bridge command")
        self.previous_input = np.asarray(action, dtype=float)
        self.previous_enabled = enabled
        self.solve_count = counters["solve_count"]
        self.fallback_count = counters["fallback_count"]

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
        return self._rl_plant(
            unit,
            unit.filter.l_f,
            unit.filter.r_f + bridge.on_resistance,
            np.ones((1, 1)),
            blocked_min=0.0,
        )

    def _forecast(self, inputs):
        references = np.full(
            (self.cfg.prediction_horizon, 1),
            inputs["reference"] * self.startup.value,
        )
        return np.array([inputs["current"]]), inputs["vin"], references

    def _disturbance(self, inputs, start, duration):
        return np.array([inputs["voltage"] * self.plant.integral(duration)])

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
        vectors = [abc2complex(row) for row in np.eye(3)]
        plant = self._rl_plant(
            unit,
            unit.ac_filter.l_f,
            unit.ac_filter.r_f + unit.bridge.on_resistance,
            np.array([
                [value.real for value in vectors],
                [value.imag for value in vectors],
            ]),
        )
        self.i_base = unit.base.i_phase_peak
        self.v_base = unit.base.v_phase_peak
        self.vdc_base = unit.dc_base.v
        return plant

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

    def _disturbance(self, inputs, start, duration):
        omega = inputs["omega"]
        denominator = self.plant.R + 1j * omega * self.plant.L
        factor = (duration / self.plant.L if denominator == 0 else
                  (cmath.exp(1j * omega * duration)
                   - math.exp(-self.plant.R * duration / self.plant.L)) / denominator)
        integral = inputs["v"] * self.v_base * cmath.exp(1j * omega * start) * factor
        return np.array([integral.real, integral.imag])

    def _write_action(self, outputs, action):
        outputs["d_a"], outputs["d_b"], outputs["d_c"] = map(float, action)
