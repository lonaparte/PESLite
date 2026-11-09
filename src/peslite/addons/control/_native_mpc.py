"""Shared fixed-size C++ MPC kernel used by native simulation and hardware export."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["MPC_KERNEL", "PredictiveLoopGenerator"]


MPC_KERNEL = r'''
template <std::size_t NX, std::size_t NU, std::size_t NY> struct PredictiveModel {
    std::array<double, NX * NX> a{};
    std::array<double, NX * NU> b{};
    std::array<double, NY * NX> c{};
    std::array<double, NX> offset{};
};

template <std::size_t NU> struct PredictiveResult {
    std::array<double, NU> action{};
    double objective = 0.0;
    std::size_t iterations = 0;
    std::size_t evaluations = 0;
    bool success = false;
};

inline double predictive_integral(double inductance, double resistance,
                                  double duration) noexcept {
    return resistance == 0.0 ? duration / inductance
                             : -std::expm1(-resistance * duration / inductance) / resistance;
}

template <std::size_t NC, std::size_t NU>
PredictiveModel<NC + NU, NU, NC> make_rl_prediction(
        double inductance, double resistance, double period, double delay, double dc_voltage,
        const std::array<double, NC>& current,
        const std::array<double, NC>& disturbance_before,
        const std::array<double, NC>& disturbance_after,
        const std::array<double, NC * NU>& voltage_map,
        bool previous_applied, bool has_blocked_min,
        const std::array<double, NC>& blocked_min) noexcept {
    constexpr std::size_t NX = NC + NU;
    PredictiveModel<NX, NU, NC> model;
    const double before_decay = std::exp(-resistance * delay / inductance);
    const double after = period - delay;
    const double after_decay = std::exp(-resistance * after / inductance);
    const double previous_gain = after_decay * predictive_integral(
        inductance, resistance, delay) * dc_voltage;
    const double new_gain = predictive_integral(inductance, resistance, after) * dc_voltage;
    for (std::size_t row = 0; row < NC; ++row) {
        double prior_state = before_decay;
        double prior_offset = -disturbance_before[row];
        if (!previous_applied && delay > 0.0 && has_blocked_min
                && prior_state * current[row] + prior_offset <= blocked_min[row]) {
            prior_state = 0.0;
            prior_offset = blocked_min[row];
        }
        model.a[row * NX + row] = after_decay * prior_state;
        model.offset[row] = after_decay * prior_offset - disturbance_after[row];
        model.c[row * NX + row] = 1.0;
        for (std::size_t column = 0; column < NU; ++column) {
            const double gain = voltage_map[row * NU + column];
            if (previous_applied)
                model.a[row * NX + NC + column] = previous_gain * gain;
            model.b[row * NU + column] = new_gain * gain;
        }
    }
    for (std::size_t column = 0; column < NU; ++column)
        model.b[(NC + column) * NU + column] = 1.0;
    return model;
}

template <std::size_t MAXIMUM>
bool predictive_solve(std::array<double, MAXIMUM * MAXIMUM> matrix,
                      std::array<double, MAXIMUM>& value,
                      std::size_t size) noexcept {
    double scale = 0.0;
    for (std::size_t row = 0; row < size; ++row)
        for (std::size_t column = 0; column < size; ++column)
            scale = std::max(scale, std::abs(matrix[row * MAXIMUM + column]));
    const double threshold = std::numeric_limits<double>::epsilon()
        * std::max(1.0, scale) * size;
    for (std::size_t column = 0; column < size; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < size; ++row)
            if (std::abs(matrix[row * MAXIMUM + column])
                    > std::abs(matrix[pivot * MAXIMUM + column])) pivot = row;
        if (!(std::abs(matrix[pivot * MAXIMUM + column]) > threshold)) return false;
        if (pivot != column) {
            for (std::size_t entry = 0; entry < size; ++entry)
                std::swap(matrix[column * MAXIMUM + entry],
                          matrix[pivot * MAXIMUM + entry]);
            std::swap(value[column], value[pivot]);
        }
        const double diagonal = matrix[column * MAXIMUM + column];
        for (std::size_t row = column + 1; row < size; ++row) {
            const double multiplier = matrix[row * MAXIMUM + column] / diagonal;
            matrix[row * MAXIMUM + column] = 0.0;
            for (std::size_t entry = column + 1; entry < size; ++entry)
                matrix[row * MAXIMUM + entry]
                    -= multiplier * matrix[column * MAXIMUM + entry];
            value[row] -= multiplier * value[column];
        }
    }
    for (std::size_t row = size; row-- > 0;) {
        for (std::size_t column = row + 1; column < size; ++column)
            value[row] -= matrix[row * MAXIMUM + column] * value[column];
        value[row] /= matrix[row * MAXIMUM + row];
    }
    for (std::size_t index = 0; index < size; ++index)
        if (!std::isfinite(value[index])) return false;
    return true;
}

template <std::size_t N, std::size_t MAX_LINEAR>
bool predictive_qp(
        const std::array<double, N * N>& hessian,
        const std::array<double, N>& linear,
        const std::array<double, N>& lower,
        const std::array<double, N>& upper,
        const std::array<double, MAX_LINEAR * N>& linear_matrix,
        const std::array<double, MAX_LINEAR>& linear_lower,
        const std::array<double, MAX_LINEAR>& linear_upper,
        std::size_t linear_rows, std::array<double, N>& solution,
        std::size_t max_iterations, double tolerance, double feasibility_tolerance,
        std::size_t& iterations) noexcept {
    constexpr std::size_t MAX_INEQUALITIES = 2 * (N + MAX_LINEAR);
    std::array<double, MAX_INEQUALITIES * N> inequalities{};
    std::array<double, MAX_INEQUALITIES> limits{}, allowed_error{};
    std::size_t count = 0;
    auto add_interval = [&](const double* row, double low, double high,
                            double error) {
        if (std::isfinite(high)) {
            for (std::size_t column = 0; column < N; ++column)
                inequalities[count * N + column] = row[column];
            limits[count] = high;
            allowed_error[count++] = error;
        }
        if (std::isfinite(low)) {
            for (std::size_t column = 0; column < N; ++column)
                inequalities[count * N + column] = -row[column];
            limits[count] = -low;
            allowed_error[count++] = error;
        }
    };
    std::array<double, N> identity_row{};
    for (std::size_t row = 0; row < N; ++row) {
        identity_row.fill(0.0);
        identity_row[row] = 1.0;
        add_interval(identity_row.data(), lower[row], upper[row], feasibility_tolerance);
    }
    for (std::size_t row = 0; row < linear_rows; ++row) {
        double scale = 0.0;
        for (std::size_t column = 0; column < N; ++column)
            scale = std::max(scale, std::abs(linear_matrix[row * N + column]));
        if (!(scale > 0.0)) {
            if (0.0 < linear_lower[row] - feasibility_tolerance
                    || 0.0 > linear_upper[row] + feasibility_tolerance) return false;
            continue;
        }
        std::array<double, N> normalized{};
        for (std::size_t column = 0; column < N; ++column)
            normalized[column] = linear_matrix[row * N + column] / scale;
        add_interval(normalized.data(), linear_lower[row] / scale,
                     linear_upper[row] / scale, feasibility_tolerance / scale);
    }
    double coefficient_scale = 1.0;
    for (double value : hessian)
        coefficient_scale = std::max(coefficient_scale, std::abs(value));
    for (double value : linear)
        coefficient_scale = std::max(coefficient_scale, std::abs(value));
    for (std::size_t column = 0; column < N; ++column)
        solution[column] = std::clamp(solution[column], lower[column], upper[column]);
    const std::size_t projection_limit = std::max<std::size_t>(32, 8 * count);
    for (std::size_t sweep = 0; sweep < projection_limit; ++sweep) {
        double violation_max = 0.0;
        for (std::size_t constraint = 0; constraint < count; ++constraint) {
            double value = -limits[constraint], norm = 0.0;
            for (std::size_t column = 0; column < N; ++column) {
                const double entry = inequalities[constraint * N + column];
                value += entry * solution[column];
                norm += entry * entry;
            }
            violation_max = std::max(
                violation_max, value - allowed_error[constraint]);
            if (value > allowed_error[constraint] && norm > 0.0)
                for (std::size_t column = 0; column < N; ++column)
                    solution[column] -= value / norm
                        * inequalities[constraint * N + column];
        }
        if (violation_max <= 0.0) break;
    }
    auto feasible = [&]() {
        for (std::size_t constraint = 0; constraint < count; ++constraint) {
            double value = -limits[constraint];
            for (std::size_t column = 0; column < N; ++column)
                value += inequalities[constraint * N + column] * solution[column];
            if (value > allowed_error[constraint]) return false;
        }
        return true;
    };
    if (!feasible()) return false;
    constexpr std::size_t KKT = 2 * N;
    std::array<bool, MAX_INEQUALITIES> active{};
    std::array<std::size_t, N> working{};
    std::size_t working_size = 0;
    const double stopping = std::max(tolerance,
        128.0 * std::numeric_limits<double>::epsilon());
    for (iterations = 1; iterations <= max_iterations; ++iterations) {
        std::array<double, N> gradient{};
        for (std::size_t row = 0; row < N; ++row) {
            gradient[row] = linear[row] / coefficient_scale;
            for (std::size_t column = 0; column < N; ++column)
                gradient[row] += hessian[row * N + column]
                    / coefficient_scale * solution[column];
        }
        std::array<double, KKT * KKT> system{};
        std::array<double, KKT> right{};
        const std::size_t size = N + working_size;
        for (std::size_t row = 0; row < N; ++row) {
            right[row] = -gradient[row];
            for (std::size_t column = 0; column < N; ++column)
                system[row * KKT + column]
                    = hessian[row * N + column] / coefficient_scale;
            system[row * KKT + row] += 1.0e-12;
        }
        for (std::size_t index = 0; index < working_size; ++index) {
            const std::size_t constraint = working[index];
            for (std::size_t column = 0; column < N; ++column) {
                const double value = inequalities[constraint * N + column];
                system[column * KKT + N + index] = value;
                system[(N + index) * KKT + column] = value;
            }
        }
        if (!predictive_solve<KKT>(system, right, size)) return false;
        double step_norm = 0.0;
        for (std::size_t column = 0; column < N; ++column)
            step_norm = std::max(step_norm, std::abs(right[column]));
        if (step_norm <= stopping) {
            std::size_t remove = working_size;
            double multiplier = -stopping;
            for (std::size_t index = 0; index < working_size; ++index)
                if (right[N + index] < multiplier) {
                    multiplier = right[N + index];
                    remove = index;
                }
            if (remove == working_size) return feasible();
            active[working[remove]] = false;
            for (std::size_t index = remove + 1; index < working_size; ++index)
                working[index - 1] = working[index];
            --working_size;
            continue;
        }
        double alpha = 1.0;
        std::size_t blocking = count;
        for (std::size_t constraint = 0; constraint < count; ++constraint) {
            if (active[constraint]) continue;
            double direction = 0.0, gap = limits[constraint];
            for (std::size_t column = 0; column < N; ++column) {
                const double entry = inequalities[constraint * N + column];
                direction += entry * right[column];
                gap -= entry * solution[column];
            }
            if (direction > stopping && gap / direction < alpha) {
                alpha = std::max(0.0, gap / direction);
                blocking = constraint;
            }
        }
        for (std::size_t column = 0; column < N; ++column)
            solution[column] += alpha * right[column];
        if (blocking != count) {
            if (working_size == N) return false;
            active[blocking] = true;
            working[working_size++] = blocking;
        }
    }
    iterations = max_iterations;
    return false;
}

template <std::size_t NX, std::size_t NU, std::size_t NY,
          std::size_t HORIZON, std::size_t MOVES>
PredictiveResult<NU> solve_ccs_mpc(
        const std::array<PredictiveModel<NX, NU, NY>, HORIZON>& models,
        const std::array<double, NX>& initial_state,
        const std::array<std::array<double, NY>, HORIZON>& references,
        const std::array<double, NU>& previous,
        const std::array<double, NY * NY>& tracking_weight,
        const std::array<double, NU * NU>& input_weight,
        const std::array<double, NU * NU>& rate_weight,
        double input_min, double input_max,
        double output_min, double output_max,
        std::size_t max_iterations, double tolerance,
        double feasibility_tolerance) noexcept {
    constexpr std::size_t NZ = MOVES * NU;
    constexpr std::size_t MAX_LINEAR = HORIZON * NY;
    PredictiveResult<NU> result;
    std::array<double, NZ * NZ> hessian{};
    std::array<double, NZ> linear{};
    double constant = 0.0;
    std::array<double, NX> free = initial_state;
    std::array<double, NX * NZ> response{};
    std::array<double, NU * NZ> prior_response{};
    std::array<double, NU> prior_free = previous;
    std::array<double, MAX_LINEAR * NZ> constraints{};
    std::array<double, MAX_LINEAR> constraint_lower{}, constraint_upper{};
    std::size_t constraint_rows = 0;
    for (std::size_t stage = 0; stage < HORIZON; ++stage) {
        const auto& model = models[stage];
        const std::size_t move = std::min(stage, MOVES - 1);
        std::array<double, NX> next_free{};
        std::array<double, NX * NZ> next_response{};
        for (std::size_t row = 0; row < NX; ++row) {
            next_free[row] = model.offset[row];
            for (std::size_t inner = 0; inner < NX; ++inner) {
                next_free[row] += model.a[row * NX + inner] * free[inner];
                for (std::size_t column = 0; column < NZ; ++column)
                    next_response[row * NZ + column]
                        += model.a[row * NX + inner] * response[inner * NZ + column];
            }
            for (std::size_t input = 0; input < NU; ++input)
                next_response[row * NZ + move * NU + input]
                    += model.b[row * NU + input];
        }
        free = next_free;
        response = next_response;
        std::array<double, NY> output_free{};
        std::array<double, NY * NZ> output_response{};
        for (std::size_t row = 0; row < NY; ++row)
            for (std::size_t inner = 0; inner < NX; ++inner) {
                output_free[row] += model.c[row * NX + inner] * free[inner];
                for (std::size_t column = 0; column < NZ; ++column)
                    output_response[row * NZ + column]
                        += model.c[row * NX + inner] * response[inner * NZ + column];
            }
        auto add_cost = [&](const auto& mapping, const auto& offset,
                            const auto& weight, std::size_t width) {
            for (std::size_t left = 0; left < NZ; ++left) {
                for (std::size_t right = 0; right < NZ; ++right)
                    for (std::size_t row = 0; row < width; ++row)
                        for (std::size_t column = 0; column < width; ++column)
                            hessian[left * NZ + right] += 2.0
                                * mapping[row * NZ + left]
                                * weight[row * width + column]
                                * mapping[column * NZ + right];
                for (std::size_t row = 0; row < width; ++row)
                    for (std::size_t column = 0; column < width; ++column)
                        linear[left] += 2.0 * mapping[row * NZ + left]
                            * weight[row * width + column] * offset[column];
            }
            for (std::size_t row = 0; row < width; ++row)
                for (std::size_t column = 0; column < width; ++column)
                    constant += offset[row] * weight[row * width + column]
                        * offset[column];
        };
        std::array<double, NY> tracking_offset{};
        for (std::size_t row = 0; row < NY; ++row)
            tracking_offset[row] = output_free[row] - references[stage][row];
        add_cost(output_response, tracking_offset, tracking_weight, NY);
        std::array<double, NU * NZ> selection{};
        for (std::size_t row = 0; row < NU; ++row)
            selection[row * NZ + move * NU + row] = 1.0;
        std::array<double, NU> zero{};
        add_cost(selection, zero, input_weight, NU);
        std::array<double, NU * NZ> delta{};
        std::array<double, NU> delta_offset{};
        for (std::size_t row = 0; row < NU; ++row) {
            delta_offset[row] = -prior_free[row];
            for (std::size_t column = 0; column < NZ; ++column)
                delta[row * NZ + column] = selection[row * NZ + column]
                    - prior_response[row * NZ + column];
        }
        add_cost(delta, delta_offset, rate_weight, NU);
        prior_response = selection;
        prior_free.fill(0.0);
        if (std::isfinite(output_min) || std::isfinite(output_max))
            for (std::size_t row = 0; row < NY; ++row) {
                for (std::size_t column = 0; column < NZ; ++column)
                    constraints[constraint_rows * NZ + column]
                        = output_response[row * NZ + column];
                constraint_lower[constraint_rows] = output_min - output_free[row];
                constraint_upper[constraint_rows] = output_max - output_free[row];
                ++constraint_rows;
            }
    }
    std::array<double, NZ> lower{}, upper{}, solution{};
    lower.fill(input_min);
    upper.fill(input_max);
    for (std::size_t move = 0; move < MOVES; ++move)
        for (std::size_t input = 0; input < NU; ++input)
            solution[move * NU + input]
                = std::clamp(previous[input], input_min, input_max);
    std::size_t iterations = 0;
    result.success = predictive_qp<NZ, MAX_LINEAR>(
        hessian, linear, lower, upper, constraints, constraint_lower, constraint_upper,
        constraint_rows, solution, max_iterations, tolerance,
        feasibility_tolerance, iterations);
    result.iterations = iterations;
    result.evaluations = iterations;
    if (!result.success) return result;
    for (std::size_t input = 0; input < NU; ++input)
        result.action[input] = solution[input];
    double objective = constant;
    for (std::size_t row = 0; row < NZ; ++row) {
        objective += linear[row] * solution[row];
        for (std::size_t column = 0; column < NZ; ++column)
            objective += 0.5 * solution[row] * hessian[row * NZ + column]
                * solution[column];
    }
    result.objective = objective;
    return result;
}

template <std::size_t NX, std::size_t NU, std::size_t NY,
          std::size_t HORIZON, std::size_t MOVES, std::size_t ACTIONS>
PredictiveResult<NU> solve_fcs_mpc(
        const std::array<PredictiveModel<NX, NU, NY>, HORIZON>& models,
        const std::array<double, NX>& initial_state,
        const std::array<std::array<double, NY>, HORIZON>& references,
        const std::array<double, NU>& previous,
        const std::array<std::array<double, NU>, ACTIONS>& actions,
        const std::array<double, NY * NY>& tracking_weight,
        const std::array<double, NU * NU>& input_weight,
        const std::array<double, NU * NU>& rate_weight,
        double input_min, double input_max,
        double output_min, double output_max,
        std::size_t max_evaluations, double feasibility_tolerance) noexcept {
    PredictiveResult<NU> result;
    std::array<std::size_t, ACTIONS> allowed{};
    std::size_t allowed_count = 0;
    for (std::size_t index = 0; index < ACTIONS; ++index) {
        bool admissible = true;
        for (double value : actions[index])
            admissible = admissible && value >= input_min && value <= input_max;
        if (admissible) allowed[allowed_count++] = index;
    }
    if (allowed_count == 0) return result;
    std::size_t plans = 1;
    for (std::size_t move = 0; move < MOVES; ++move) {
        if (plans > max_evaluations / allowed_count) {
            result.evaluations = max_evaluations;
            return result;
        }
        plans *= allowed_count;
    }
    result.evaluations = plans;
    double best_cost = std::numeric_limits<double>::infinity();
    double best_distance = std::numeric_limits<double>::infinity();
    std::array<double, NU> action_scale{};
    for (std::size_t input = 0; input < NU; ++input) {
        double low = actions[allowed[0]][input], high = actions[allowed[0]][input];
        for (std::size_t index = 0; index < allowed_count; ++index) {
            const auto& action = actions[allowed[index]];
            low = std::min(low, action[input]);
            high = std::max(high, action[input]);
        }
        action_scale[input] = high > low ? high - low : 1.0;
    }
    for (std::size_t plan = 0; plan < plans; ++plan) {
        std::array<std::size_t, MOVES> choices{};
        std::size_t divisor = plans;
        for (std::size_t move = 0; move < MOVES; ++move) {
            divisor /= allowed_count;
            choices[move] = allowed[(plan / divisor) % allowed_count];
        }
        std::array<double, NX> state = initial_state;
        std::array<double, NU> prior = previous;
        double cost = 0.0;
        bool feasible = true;
        for (std::size_t stage = 0; stage < HORIZON && feasible; ++stage) {
            const auto& action = actions[choices[std::min(stage, MOVES - 1)]];
            const auto& model = models[stage];
            std::array<double, NX> next = model.offset;
            for (std::size_t row = 0; row < NX; ++row) {
                for (std::size_t column = 0; column < NX; ++column)
                    next[row] += model.a[row * NX + column] * state[column];
                for (std::size_t input = 0; input < NU; ++input)
                    next[row] += model.b[row * NU + input] * action[input];
            }
            std::array<double, NY> output{};
            for (std::size_t row = 0; row < NY; ++row) {
                for (std::size_t column = 0; column < NX; ++column)
                    output[row] += model.c[row * NX + column] * next[column];
                feasible = feasible
                    && output[row] >= output_min - feasibility_tolerance
                    && output[row] <= output_max + feasibility_tolerance;
            }
            if (!feasible) break;
            for (std::size_t row = 0; row < NY; ++row)
                for (std::size_t column = 0; column < NY; ++column)
                    cost += (output[row] - references[stage][row])
                        * tracking_weight[row * NY + column]
                        * (output[column] - references[stage][column]);
            for (std::size_t row = 0; row < NU; ++row)
                for (std::size_t column = 0; column < NU; ++column) {
                    cost += action[row] * input_weight[row * NU + column]
                        * action[column];
                    cost += (action[row] - prior[row])
                        * rate_weight[row * NU + column]
                        * (action[column] - prior[column]);
                }
            state = next;
            prior = action;
        }
        if (!feasible || !std::isfinite(cost)) continue;
        double distance = 0.0;
        const auto& first = actions[choices[0]];
        for (std::size_t input = 0; input < NU; ++input)
            distance += std::abs(first[input] - previous[input]) / action_scale[input];
        const double tie = 32.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(cost), std::abs(best_cost)});
        const bool better = !result.success || cost < best_cost - tie;
        const bool tied = result.success && std::abs(cost - best_cost) <= tie;
        if (better || (tied && distance < best_distance)) {
            result.success = true;
            result.action = first;
            result.objective = cost;
            best_cost = cost;
            best_distance = distance;
        }
    }
    return result;
}
'''


def _native_weight(value: Any, width: int) -> tuple[Any, ...]:
    """Normalize one validated scalar, diagonal or full MPC weight without losing parameters."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return tuple(value if row == column else 0.0
                     for row in range(width) for column in range(width))
    values = tuple(value)
    nested = any(isinstance(item, Sequence) and not isinstance(item, (str, bytes))
                 for item in values)
    if not nested:
        return tuple(values[row] if row == column else 0.0
                     for row in range(width) for column in range(width))
    return tuple(item for row in values for item in row)


def _native_array(generator: Any, values: Iterable[Any]) -> str:
    return "{{" + ", ".join(
        value if isinstance(value, str) else generator.number(value) for value in values
    ) + "}}"


class PredictiveLoopGenerator:
    """Lower predictive loops to the target-independent fixed-size MPC kernel."""

    shared_code = MPC_KERNEL

    def members(self, generator: Any, loop_name: str, node: Any) -> list[str]:
        lines = []
        for local, value in node.get_state().items():
            lines.append(
                f"{generator.cpp_type(value)} "
                f"{generator.node(loop_name, local)} = {generator.literal(value)};"
            )
        for field, value in node.observe().items():
            lines.append(f"double {generator.g(loop_name + '.' + field)} = {generator.number(value)};")
        return lines

    @staticmethod
    def reset(_generator: Any, _loop_name: str, _node: Any) -> list[str]:
        return []

    @staticmethod
    def _bound(value: Any, lower: bool) -> Any:
        if value is not None:
            return value
        return -math.inf if lower else math.inf

    def _common(self, generator: Any, loop_name: str, node: Any) -> tuple[list[str], dict[str, str]]:
        horizon = int(node.cfg.prediction_horizon)
        moves = (int(node.cfg.control_horizon)
                 if node.cfg.control_horizon is not None else horizon)
        nc, nu, nx = node.n_current, node.n_input, node.n_current + node.n_input
        prefix = generator.m("mpc_" + loop_name)
        previous = [generator._state(loop_name, f"previous_input.{index}")
                    for index in range(nu)]
        names = {
            "prefix": prefix,
            "models": prefix + "_models",
            "current": prefix + "_current",
            "initial": prefix + "_initial",
            "references": prefix + "_references",
            "previous": prefix + "_previous",
            "result": prefix + "_result",
            "action": prefix + "_action",
            "valid": prefix + "_valid",
            "enabled": prefix + "_enabled",
        }
        q = _native_weight(node.cfg.tracking_weight, nc)
        r = _native_weight(node.cfg.input_weight, nu)
        s = _native_weight(node.cfg.rate_weight, nu)
        lines = [
            f"std::array<PredictiveModel<{nx}, {nu}, {nc}>, {horizon}> {names['models']}{{}};",
            f"const std::array<double, {nu}> {names['previous']}{_native_array(generator, previous)};",
            f"std::array<double, {nx}> {names['initial']}{{}};",
            *[f"{names['initial']}[{index}] = {value};"
              for index, value in enumerate((*(["0.0"] * nc), *previous))],
            f"std::array<std::array<double, {nc}>, {horizon}> {names['references']}{{}};",
            f"const std::array<double, {nc * nc}> {prefix}_q{_native_array(generator, q)};",
            f"const std::array<double, {nu * nu}> {prefix}_r{_native_array(generator, r)};",
            f"const std::array<double, {nu * nu}> {prefix}_s{_native_array(generator, s)};",
        ]
        return lines, names

    def _dc_prediction(self, generator: Any, loop_name: str, node: Any,
                       names: Mapping[str, str]) -> list[str]:
        current = generator.input(loop_name, "current")
        voltage = generator.input(loop_name, "voltage")
        vdc = generator.input(loop_name, "vin")
        reference = generator.input(loop_name, "reference")
        prefix = names["prefix"]
        plant = node.plant
        horizon = int(node.cfg.prediction_horizon)
        previous_enabled = generator._state(loop_name, "previous_enabled")
        return [
            f"const std::array<double, 1> {names['current']}{{{current}}};",
            f"{names['initial']}[0] = {current};",
            f"const double {prefix}_before = {voltage} * predictive_integral("
            f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.delay)});",
            f"const double {prefix}_after = {voltage} * predictive_integral("
            f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.period - plant.delay)});",
            *[line for stage in range(horizon) for line in (
                f"{names['references']}[{stage}][0] = {reference} * {generator.m('startup_value')};",
                f"{names['models']}[{stage}] = make_rl_prediction<1, 1>("
                f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.period)}, "
                f"{generator.number(plant.delay)}, {vdc}, {names['current']}, "
                f"std::array<double, 1>{{{prefix}_before}}, "
                f"std::array<double, 1>{{{prefix}_after}}, "
                f"std::array<double, 1>{{1.0}}, "
                f"{'true' if stage else previous_enabled}, true, std::array<double, 1>{{0.0}});",
            )],
        ]

    def _ac_prediction(self, generator: Any, loop_name: str, node: Any,
                       names: Mapping[str, str]) -> list[str]:
        prefix = names["prefix"]
        plant = node.plant
        horizon = int(node.cfg.prediction_horizon)
        current = generator.input(loop_name, "i")
        voltage = generator.input(loop_name, "v")
        frame = generator.input(loop_name, "frame")
        omega = generator.input(loop_name, "omega")
        id_ref = generator.input(loop_name, "id_ref")
        iq_ref = generator.input(loop_name, "iq_ref")
        vdc = generator.input(loop_name, "u_dc")
        previous_enabled = generator._state(loop_name, "previous_enabled")
        voltage_map = tuple(float(value) for value in plant.voltage_map.ravel())
        lines = [
            f"const Complex {prefix}_i = {current} * {generator.number(node.i_base)};",
            f"const std::array<double, 2> {names['current']}{{std::real({prefix}_i), std::imag({prefix}_i)}};",
            f"{names['initial']}[0] = std::real({prefix}_i);",
            f"{names['initial']}[1] = std::imag({prefix}_i);",
            f"const Complex {prefix}_target = Complex{{{id_ref}, {iq_ref}}} "
            f"* {generator.number(node.i_base)} * {generator.m('startup_value')};",
            f"const double {prefix}_vdc = {vdc} * {generator.number(node.vdc_base)};",
            f"const Complex {prefix}_denominator{{{generator.number(plant.R)}, {omega} * {generator.number(plant.L)}}};",
            f"auto {prefix}_disturbance = [&](double start, double duration) {{",
            f"    const Complex factor = std::abs({prefix}_denominator) == 0.0",
            f"        ? Complex{{duration / {generator.number(plant.L)}, 0.0}}",
            f"        : (std::exp(Complex{{0.0, {omega} * duration}}) "
            f"- std::exp(-{generator.number(plant.R)} * duration / {generator.number(plant.L)})) / {prefix}_denominator;",
            f"    const Complex value = {voltage} * {generator.number(node.v_base)} "
            f"* std::exp(Complex{{0.0, {omega} * start}}) * factor;",
            "    return std::array<double, 2>{std::real(value), std::imag(value)};",
            "};",
        ]
        for stage in range(horizon):
            before = f"{prefix}_before_{stage}"
            after = f"{prefix}_after_{stage}"
            lines += [
                f"const Complex {prefix}_reference_{stage} = {prefix}_target * std::exp("
                f"Complex{{0.0, {frame} + {omega} * {generator.number((stage + 1) * plant.period)}}});",
                f"{names['references']}[{stage}] = {{std::real({prefix}_reference_{stage}), "
                f"std::imag({prefix}_reference_{stage})}};",
                f"const auto {before} = {prefix}_disturbance("
                f"{generator.number(stage * plant.period)}, {generator.number(plant.delay)});",
                f"const auto {after} = {prefix}_disturbance("
                f"{generator.number(stage * plant.period + plant.delay)}, "
                f"{generator.number(plant.period - plant.delay)});",
                f"{names['models']}[{stage}] = make_rl_prediction<2, 3>("
                f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.period)}, "
                f"{generator.number(plant.delay)}, {prefix}_vdc, {names['current']}, {before}, {after}, "
                f"std::array<double, 6>{_native_array(generator, voltage_map)}, "
                f"{'true' if stage else previous_enabled}, false, std::array<double, 2>{{}});",
            ]
        return lines

    def _solver(self, generator: Any, node: Any, names: Mapping[str, str]) -> list[str]:
        horizon = int(node.cfg.prediction_horizon)
        moves = (int(node.cfg.control_horizon)
                 if node.cfg.control_horizon is not None else horizon)
        nc, nu, nx = node.n_current, node.n_input, node.n_current + node.n_input
        prefix = names["prefix"]
        output_min = self._bound(node.cfg.current_min, True)
        output_max = self._bound(node.cfg.current_max, False)
        common = (
            f"{names['models']}, {names['initial']}, {names['references']}, "
            f"{names['previous']}, {prefix}_q, {prefix}_r, {prefix}_s, "
            f"{generator.number(node.cfg.input_min)}, {generator.number(node.cfg.input_max)}, "
            f"{generator.number(output_min)}, {generator.number(output_max)}, "
        )
        if node.cfg.algorithm == "ccs":
            call = (
                f"solve_ccs_mpc<{nx}, {nu}, {nc}, {horizon}, {moves}>("
                f"{common}static_cast<std::size_t>({generator.number(node.cfg.max_iterations)}), "
                f"{generator.number(node.cfg.tolerance)}, {generator.number(node.cfg.feasibility_tolerance)})"
            )
            return [f"{names['result']} = {call};"]
        actions = tuple(itertools.product((0.0, 1.0), repeat=nu))
        action_values = ", ".join(_native_array(generator, action) for action in actions)
        call = (
            f"solve_fcs_mpc<{nx}, {nu}, {nc}, {horizon}, {moves}, {len(actions)}>("
            f"{names['models']}, {names['initial']}, {names['references']}, "
            f"{names['previous']}, {prefix}_actions, {prefix}_q, {prefix}_r, {prefix}_s, "
            f"{generator.number(node.cfg.input_min)}, {generator.number(node.cfg.input_max)}, "
            f"{generator.number(output_min)}, {generator.number(output_max)}, "
            f"static_cast<std::size_t>({generator.number(node.cfg.max_evaluations)}), "
            f"{generator.number(node.cfg.feasibility_tolerance)})"
        )
        return [
            f"const std::array<std::array<double, {nu}>, {len(actions)}> "
            f"{prefix}_actions{{{{{action_values}}}}};",
            f"{names['result']} = {call};",
        ]

    def _fixed_action(self, generator: Any, node: Any, prefix: str) -> tuple[list[str], str]:
        name = prefix + "_fixed"
        if node.cfg.fallback_input is not None:
            return ([f"const std::array<double, {node.n_input}> {name}"
                     f"{_native_array(generator, node.cfg.fallback_input)};"], name)
        if node.cfg.algorithm == "ccs":
            return ([f"std::array<double, {node.n_input}> {name}{{}};",
                     f"{name}.fill({generator.number(node.cfg.input_min)});"], name)
        default = (f"({generator.number(node.cfg.input_min)} <= 0.0 && "
                   f"{generator.number(node.cfg.input_max)} >= 0.0 ? 0.0 : 1.0)")
        return ([f"std::array<double, {node.n_input}> {name}{{}};",
                 f"{name}.fill({default});"], name)

    def lines(self, generator: Any, loop_name: str, node: Any) -> list[str]:
        lines, names = self._common(generator, loop_name, node)
        fixed_lines, fixed = self._fixed_action(generator, node, names["prefix"])
        lines += [*fixed_lines, f"PredictiveResult<{node.n_input}> {names['result']}{{}};"]
        prediction = (
            self._dc_prediction(generator, loop_name, node, names)
            if node.n_current == 1
            else self._ac_prediction(generator, loop_name, node, names)
        )
        solve = [*prediction, *self._solver(generator, node, names)]
        lines += [f"if ({generator.m('startup_run')}) {{",
                  *("    " + line for line in solve), "}"]
        previous_enabled = generator._state(loop_name, "previous_enabled")
        solves = generator._state(loop_name, "solve_count")
        fallbacks = generator._state(loop_name, "fallback_count")
        lines += [
            f"std::array<double, {node.n_input}> {names['action']} = {fixed};",
            f"bool {names['enabled']} = {generator.m('startup_run')};",
            f"bool {names['valid']} = false;",
            f"if ({generator.m('startup_run')}) {{",
            f"    ++{solves};",
            f"    {names['valid']} = {names['result']}.success;",
            f"    for (double value : {names['result']}.action)",
            f"        {names['valid']} = {names['valid']} && std::isfinite(value) "
            f"&& value >= {generator.number(node.cfg.input_min)} && value <= {generator.number(node.cfg.input_max)};",
            f"    if ({names['valid']}) {names['action']} = {names['result']}.action;",
            "    else {",
            f"        ++{fallbacks};",
        ]
        if node.cfg.fallback == "hold":
            hold_valid = (f"std::all_of({names['previous']}.begin(), {names['previous']}.end(), "
                          f"[&](double value) {{ return std::isfinite(value) "
                          f"&& value >= {generator.number(node.cfg.input_min)} "
                          f"&& value <= {generator.number(node.cfg.input_max)}; }})")
            lines += [f"        if ({hold_valid}) {names['action']} = {names['previous']};"]
        if node.cfg.fallback == "block":
            lines.append(f"        {names['enabled']} = false;")
        lines += [
            "    }",
            "}",
            *[f"{generator._state(loop_name, f'previous_input.{index}')} = "
              f"{names['action']}[{index}];" for index in range(node.n_input)],
            f"{previous_enabled} = {names['enabled']};",
        ]
        output_ports = tuple(port for port in node.outputs if port != "enable")
        lines += [f"{generator.g(loop_name + '.' + port)} = {names['action']}[{index}];"
                  for index, port in enumerate(output_ports)]
        lines += [
            f"{generator.g(loop_name + '.enable')} = {names['enabled']} ? 1.0 : 0.0;",
            f"{generator.g(loop_name + '.success')} = {names['valid']} ? 1.0 : 0.0;",
            f"{generator.g(loop_name + '.objective')} = "
            f"{names['result']}.success ? {names['result']}.objective : 0.0;",
            f"{generator.g(loop_name + '.iterations')} = "
            f"static_cast<double>({names['result']}.iterations);",
            f"{generator.g(loop_name + '.evaluations')} = "
            f"static_cast<double>({names['result']}.evaluations);",
            f"{generator.g(loop_name + '.solves')} = {solves};",
            f"{generator.g(loop_name + '.fallbacks')} = {fallbacks};",
            f"{generator.g(loop_name + '.delay')} = {generator.number(node.delay)};",
        ]
        return lines
