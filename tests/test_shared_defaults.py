"""Shared configurable defaults remain the single source for every consumer."""

import inspect
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from peslite.addons.functions.stability import small_signal, small_signal_report
from peslite.assembly.conduction import DiodeConduction
from peslite.assembly.cpp import _CppGenerator
from peslite.assembly.params import OutputParams, PWMParams, SolverParams
from peslite.components.pwm import (CarrierComparison, PhaseShiftModulator,
                                    SynchronousCarrier, carrier)
from peslite.control.controller import ControlGraph
from peslite.defaults import (
    DEFAULT_ALGEBRAIC_ATOL,
    DEFAULT_ALGEBRAIC_MAX_ITERATIONS,
    DEFAULT_ALGEBRAIC_RTOL,
    DEFAULT_COMMUTATION_LOCATOR_ITERATIONS,
    DEFAULT_COMMUTATION_MAX_TRANSITIONS,
    DEFAULT_LINEARIZATION_RELATIVE_STEP,
    DEFAULT_MULTIRATE_SWEEPS,
    DEFAULT_OUTPUT_PRECISION,
    DEFAULT_OUTPUT_WRITE_LENGTH,
    DEFAULT_PH_REPORT_RTOL,
    DEFAULT_PWM_ALIGNMENT,
    DEFAULT_PWM_CARRIER_PHASE,
    DEFAULT_PWM_MIN_INTERVAL,
    DEFAULT_SOLVER_ATOL,
    DEFAULT_SOLVER_MAX_STEP,
    DEFAULT_SOLVER_RTOL,
    DEFAULT_SOLVER_WARM_START,
    DEFAULT_SPLIT_LINEARIZATION_ROWS,
    DEFAULT_STABILITY_MULTIPLIER_TOLERANCE,
    DEFAULT_STABILITY_PERIODIC_TOLERANCE,
)
from peslite.solver.energy import ph_report
from peslite.solver.implicit import ImplicitSolver
from peslite.solver.integrators import (AdaptiveSolver, AutoSolver, DormandPrince45,
                                        FixedStepSolver, integrator)
from peslite.solver.linearization import linearize_loop
from peslite.solver.model import Model, _AlgebraicLoop
from peslite.solver.multirate import MultirateSolver
from peslite.solver.simulation import Recorder, _CSVTable
from peslite.solver.splitbound import split_error_bound


def _defaults(callable_):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(callable_).parameters.items()
    }


def test_shared_solver_defaults_are_selected_by_each_parameter_interface():
    configured = {item.name: item.default for item in fields(SolverParams)}

    assert configured["rtol"] == DEFAULT_SOLVER_RTOL
    assert configured["atol"] == DEFAULT_SOLVER_ATOL
    assert configured["max_step"] == DEFAULT_SOLVER_MAX_STEP
    assert configured["warm_start"] == DEFAULT_SOLVER_WARM_START
    assert configured["sweeps"] == DEFAULT_MULTIRATE_SWEEPS
    assert configured["linearisations"] == DEFAULT_SPLIT_LINEARIZATION_ROWS
    assert configured["write_length"] == DEFAULT_OUTPUT_WRITE_LENGTH
    for interface in (integrator, FixedStepSolver, AdaptiveSolver, DormandPrince45,
                      AutoSolver, ImplicitSolver, MultirateSolver):
        defaults = _defaults(interface)
        assert defaults["rtol"] == DEFAULT_SOLVER_RTOL
        assert defaults["atol"] == DEFAULT_SOLVER_ATOL
    for interface in (integrator, AdaptiveSolver, DormandPrince45, AutoSolver):
        assert _defaults(interface)["max_step"] == DEFAULT_SOLVER_MAX_STEP
    for interface in (integrator, AdaptiveSolver, AutoSolver):
        assert _defaults(interface)["warm_start"] == DEFAULT_SOLVER_WARM_START
    assert _defaults(MultirateSolver)["sweeps"] == DEFAULT_MULTIRATE_SWEEPS

    adaptive = AdaptiveSolver()
    dormand_prince = DormandPrince45()
    runtime = (
        integrator("RK4", 1.0),
        FixedStepSolver(1.0),
        adaptive,
        dormand_prince,
        ImplicitSolver(1.0),
    )
    for solver in runtime:
        assert solver.rtol == DEFAULT_SOLVER_RTOL
        assert solver.atol == DEFAULT_SOLVER_ATOL
    assert adaptive.max_step == DEFAULT_SOLVER_MAX_STEP
    assert adaptive.warm_start == DEFAULT_SOLVER_WARM_START
    assert dormand_prince.max_step == DEFAULT_SOLVER_MAX_STEP
    automatic = AutoSolver()
    assert automatic._dp.rtol == DEFAULT_SOLVER_RTOL
    assert automatic._dp.atol == DEFAULT_SOLVER_ATOL
    assert automatic._dp.max_step == DEFAULT_SOLVER_MAX_STEP
    assert automatic._warm_start == DEFAULT_SOLVER_WARM_START


def test_shared_execution_defaults_are_selected_by_their_owners():
    control = _defaults(ControlGraph)
    algebraic = _defaults(_AlgebraicLoop)
    conduction = _defaults(DiodeConduction)
    cpp = _defaults(_CppGenerator.source)

    assert control["cycle_iterations"] == DEFAULT_ALGEBRAIC_MAX_ITERATIONS
    assert control["cycle_atol"] == DEFAULT_ALGEBRAIC_ATOL
    assert control["cycle_rtol"] == DEFAULT_ALGEBRAIC_RTOL
    assert algebraic["max_sweeps"] == DEFAULT_ALGEBRAIC_MAX_ITERATIONS
    assert algebraic["atol"] == DEFAULT_ALGEBRAIC_ATOL
    assert algebraic["rtol"] == DEFAULT_ALGEBRAIC_RTOL
    assert conduction["max_transitions"] == DEFAULT_COMMUTATION_MAX_TRANSITIONS
    assert conduction["locator_iterations"] == DEFAULT_COMMUTATION_LOCATOR_ITERATIONS
    assert cpp["max_commutation_transitions"] == DEFAULT_COMMUTATION_MAX_TRANSITIONS
    assert cpp["commutation_locator_iterations"] == DEFAULT_COMMUTATION_LOCATOR_ITERATIONS

    runtime_loop = _AlgebraicLoop((), (), (), ())
    assert runtime_loop.max_sweeps == DEFAULT_ALGEBRAIC_MAX_ITERATIONS
    assert runtime_loop.atol == DEFAULT_ALGEBRAIC_ATOL
    assert runtime_loop.rtol == DEFAULT_ALGEBRAIC_RTOL
    runtime_conduction = DiodeConduction(SimpleNamespace(subsystems=()))
    assert runtime_conduction.max_transitions == DEFAULT_COMMUTATION_MAX_TRANSITIONS
    assert runtime_conduction.locator_iterations == DEFAULT_COMMUTATION_LOCATOR_ITERATIONS


def test_shared_pwm_and_linearization_defaults_are_explicit_parameters():
    pwm = {item.name: item.default for item in fields(PWMParams)}

    assert pwm["carrier_phase"] == DEFAULT_PWM_CARRIER_PHASE
    assert pwm["alignment"] == DEFAULT_PWM_ALIGNMENT
    assert _defaults(carrier)["phase"] == DEFAULT_PWM_CARRIER_PHASE
    assert _defaults(CarrierComparison)["phase"] == DEFAULT_PWM_CARRIER_PHASE
    assert _defaults(SynchronousCarrier)["phase"] == DEFAULT_PWM_CARRIER_PHASE
    assert _defaults(CarrierComparison)["alignment"] == DEFAULT_PWM_ALIGNMENT
    assert _defaults(SynchronousCarrier)["alignment"] == DEFAULT_PWM_ALIGNMENT
    assert _defaults(CarrierComparison)["min_interval"] == DEFAULT_PWM_MIN_INTERVAL
    assert _defaults(SynchronousCarrier)["min_interval"] == DEFAULT_PWM_MIN_INTERVAL
    assert _defaults(PhaseShiftModulator)["min_interval"] == DEFAULT_PWM_MIN_INTERVAL
    assert _defaults(linearize_loop)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
    assert _defaults(split_error_bound)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
    assert _defaults(split_error_bound)["rows"] == DEFAULT_SPLIT_LINEARIZATION_ROWS
    assert _defaults(small_signal)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
    assert _defaults(small_signal_report)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
    for interface in (small_signal, small_signal_report):
        assert (_defaults(interface)["periodic_tolerance"]
                == DEFAULT_STABILITY_PERIODIC_TOLERANCE)
        assert (_defaults(interface)["multiplier_tolerance"]
                == DEFAULT_STABILITY_MULTIPLIER_TOLERANCE)

    carrier_runtime = CarrierComparison(1.0)
    synchronous_runtime = SynchronousCarrier(1)
    assert carrier_runtime.phase == DEFAULT_PWM_CARRIER_PHASE
    assert carrier_runtime.alignment == DEFAULT_PWM_ALIGNMENT
    assert carrier_runtime.min_interval == DEFAULT_PWM_MIN_INTERVAL
    assert synchronous_runtime.phase == DEFAULT_PWM_CARRIER_PHASE
    assert synchronous_runtime.alignment == DEFAULT_PWM_ALIGNMENT
    assert synchronous_runtime.min_interval == DEFAULT_PWM_MIN_INTERVAL
    assert PhaseShiftModulator(1.0).min_interval == DEFAULT_PWM_MIN_INTERVAL

    class ProbeLoop:
        period = 1.0

        def __init__(self):
            self.points = []

        @staticmethod
        def to_frame(row, _t, _cols):
            return np.asarray([row["x"]], dtype=float)

        def from_frame(self, point, _t, _cols, _row):
            self.points.append(float(point[0]))
            return {"x": float(point[0])}

        @staticmethod
        def advance(row, _t, _view=None):
            return dict(row)

    loop = ProbeLoop()
    linearize_loop(loop, {"x": 2.0}, 0.0, [("scalar", ("x",), 1.0)])
    step = 2.0 * DEFAULT_LINEARIZATION_RELATIVE_STEP
    np.testing.assert_allclose(loop.points, [2.0 + step, 2.0 - step])


def test_shared_report_and_output_defaults_reach_each_interface():
    output = {item.name: item.default for item in fields(OutputParams)}

    assert output["precision"] == DEFAULT_OUTPUT_PRECISION
    assert _defaults(_CSVTable)["precision"] == DEFAULT_OUTPUT_PRECISION
    assert _defaults(_CSVTable)["batch_rows"] == DEFAULT_OUTPUT_WRITE_LENGTH
    assert _defaults(Recorder)["batch_rows"] == DEFAULT_OUTPUT_WRITE_LENGTH
    assert _defaults(ph_report)["rtol"] == DEFAULT_PH_REPORT_RTOL
    assert _defaults(Model.ph_report)["rtol"] == DEFAULT_PH_REPORT_RTOL
