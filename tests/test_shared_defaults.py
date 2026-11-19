"""Shared defaults are inert values selected explicitly by parameter declarations."""

import inspect
from dataclasses import fields

from peslite.addons.functions.stability import small_signal, small_signal_report
from peslite.assembly.conduction import DiodeConduction
from peslite.assembly.params import PWMParams, SolverParams
from peslite.components.pwm import CarrierComparison, SynchronousCarrier
from peslite.control.controller import ControlGraph
from peslite.defaults import (
    DEFAULT_COMMUTATION_LOCATOR_ITERATIONS,
    DEFAULT_COMMUTATION_MAX_TRANSITIONS,
    DEFAULT_CONTROL_ALGEBRAIC_ATOL,
    DEFAULT_CONTROL_ALGEBRAIC_MAX_ITERATIONS,
    DEFAULT_CONTROL_ALGEBRAIC_RTOL,
    DEFAULT_LINEARIZATION_RELATIVE_STEP,
    DEFAULT_PWM_ALIGNMENT,
    DEFAULT_SOLVER_ATOL,
    DEFAULT_SOLVER_RTOL,
)
from peslite.solver.integrators import FixedStepSolver, integrator
from peslite.solver.linearization import linearize_loop
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
    assert _defaults(integrator)["rtol"] == DEFAULT_SOLVER_RTOL
    assert _defaults(integrator)["atol"] == DEFAULT_SOLVER_ATOL
    assert _defaults(FixedStepSolver)["rtol"] == DEFAULT_SOLVER_RTOL
    assert _defaults(FixedStepSolver)["atol"] == DEFAULT_SOLVER_ATOL


def test_shared_execution_defaults_are_selected_by_their_owners():
    control = _defaults(ControlGraph)
    conduction = _defaults(DiodeConduction)

    assert control["cycle_iterations"] == DEFAULT_CONTROL_ALGEBRAIC_MAX_ITERATIONS
    assert control["cycle_atol"] == DEFAULT_CONTROL_ALGEBRAIC_ATOL
    assert control["cycle_rtol"] == DEFAULT_CONTROL_ALGEBRAIC_RTOL
    assert conduction["max_transitions"] == DEFAULT_COMMUTATION_MAX_TRANSITIONS
    assert conduction["locator_iterations"] == DEFAULT_COMMUTATION_LOCATOR_ITERATIONS


def test_shared_pwm_and_linearization_defaults_are_explicit_parameters():
    pwm = {item.name: item.default for item in fields(PWMParams)}

    assert pwm["alignment"] == DEFAULT_PWM_ALIGNMENT
    assert _defaults(CarrierComparison)["alignment"] == DEFAULT_PWM_ALIGNMENT
    assert _defaults(SynchronousCarrier)["alignment"] == DEFAULT_PWM_ALIGNMENT
    assert _defaults(linearize_loop)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
    assert _defaults(split_error_bound)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
    assert _defaults(small_signal)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
    assert _defaults(small_signal_report)["relative_step"] == DEFAULT_LINEARIZATION_RELATIVE_STEP
