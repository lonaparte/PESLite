"""Shared defaults selected explicitly by public and internal parameter interfaces.

Importing this module does not register or apply a default. A consumer opts into one by using the
named value in a parameter declaration; execution then reads the received parameter.
"""

DEFAULT_SOLVER_RTOL = 1.0e-6
DEFAULT_SOLVER_ATOL = 1.0e-9

DEFAULT_CONTROL_ALGEBRAIC_MAX_ITERATIONS = 16
DEFAULT_CONTROL_ALGEBRAIC_ATOL = 1.0e-10
DEFAULT_CONTROL_ALGEBRAIC_RTOL = 1.0e-8

DEFAULT_PWM_ALIGNMENT = "center"

DEFAULT_COMMUTATION_MAX_TRANSITIONS = 64
DEFAULT_COMMUTATION_LOCATOR_ITERATIONS = 80

DEFAULT_LINEARIZATION_RELATIVE_STEP = 1.0e-4
