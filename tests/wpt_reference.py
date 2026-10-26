"""Independent reference for a compensated link fed on its primary port and rectified by an
ideal diode bridge on its secondary port into a stiff DC voltage.

Written from the circuit equations in coil coordinates (full inductance matrix, no modal
transformation) and integrated with SciPy DOP853 between located diode transitions. A blocked
bridge keeps its port current at zero: the port voltage then follows from the linearity of the
port-current rate in that voltage. Conventions follow PESLite: a port current flows from the
converter into the link; a conducting bridge applies ``-direction * V2`` with ``direction`` the
sign of its port current.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


class Circuit:
    """Coupled coils, coil capacitors, and optional LCC front ends (``lcc[side]``)."""

    def __init__(self, inductance, capacitance, resistance, coil_sides, lcc=None,
                 primary_series_resistance=0.0):
        self.L = np.asarray(inductance, dtype=float)
        self.C = np.asarray(capacitance, dtype=float)
        self.R = np.asarray(resistance, dtype=float)
        self.sides = tuple(coil_sides)
        self.n = len(self.sides)
        self.lcc = dict(lcc or {})  # side -> (Lf, Rf, Cf, Rcf)
        self.front = [side for side in ("primary", "secondary") if side in self.lcc]
        self.r_series = primary_series_resistance
        self.size = 2 * len(self.front) + 2 * self.n

    def _split(self, x):
        k = len(self.front)
        comp = dict(zip(self.front, x[:k]))
        shunt = dict(zip(self.front, x[k:2 * k]))
        i = x[2 * k:2 * k + self.n]
        uc = x[2 * k + self.n:]
        return comp, shunt, i, uc

    def side_sum(self, i, side):
        return sum(i[j] for j in range(self.n) if self.sides[j] == side)

    def port_current(self, x, side):
        comp, _shunt, i, _uc = self._split(x)
        return comp[side] if side in self.lcc else self.side_sum(i, side)

    def derivative(self, x, u_primary, u_secondary):
        comp, shunt, i, uc = self._split(x)
        if "primary" not in self.lcc and self.r_series:
            u_primary = u_primary - self.r_series * self.side_sum(i, "primary")
        elif "primary" in self.lcc and self.r_series:
            u_primary = u_primary - self.r_series * comp["primary"]
        ports = {"primary": u_primary, "secondary": u_secondary}
        node, d_comp, d_shunt = {}, [], []
        for side in ("primary", "secondary"):
            if side in self.lcc:
                lf, rf, cf, rcf = self.lcc[side]
                branch = comp[side] - self.side_sum(i, side)
                node[side] = shunt[side] + rcf * branch
                d_comp.append((ports[side] - node[side] - rf * comp[side]) / lf)
                d_shunt.append(branch / cf)
            else:
                node[side] = ports[side]
        v = np.array([node[self.sides[j]] for j in range(self.n)]) - self.R * i - uc
        di = np.linalg.solve(self.L, v)
        return np.concatenate([d_comp, d_shunt, di, i / self.C])

    def hold_voltage(self, x, u_primary, side="secondary"):
        """Secondary port voltage with zero rate of change of its port current."""
        def rate(u):
            dx = self.derivative(x, u_primary, u)
            return self.port_current(dx, side)
        r0 = rate(0.0)
        return -r0 / (rate(1.0) - r0)


def rectify(circuit, u_primary, V2, t_end, t_samples, rtol=1e-11, atol=1e-13, max_events=200000,
            parallel_capacitance=0.0, on_resistance=0.0, breaks=None):
    """Integrate from rest; returns sampled circuit states, the DC current samples (into the DC
    side) and the transition log [(t, direction)].

    With ``parallel_capacitance`` Cp > 0, positive ``on_resistance`` gives finite diode current;
    capacitor voltage is appended as the last state. Without Cp the diodes are ideal.

    Without Cp, ``breaks`` (times of source discontinuities) makes ``u_primary`` piecewise constant between
    them and evaluated inside each segment; a blocked bridge whose hold voltage jumps beyond the
    DC voltage at a break starts conducting there.
    """
    cp = float(parallel_capacitance)
    if cp > 0.0:
        if on_resistance <= 0.0:
            raise ValueError("finite capacitance requires positive diode resistance")
        n = circuit.size

        def current(voltage):
            return (np.maximum(-V2 - voltage, 0.0) - np.maximum(voltage - V2, 0.0)) / (2 * on_resistance)

        def rhs(tt, xx):
            voltage = xx[n]
            injected = current(voltage) - circuit.port_current(xx[:n], "secondary")
            return np.append(circuit.derivative(xx[:n], u_primary(tt), voltage), injected / cp)

        sol = solve_ivp(rhs, (0.0, t_end), np.zeros(n + 1), method="DOP853", rtol=rtol,
                        atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        out = sol.sol(t_samples).T
        return out, -np.abs(current(out[:, n])), []
    size = circuit.size
    x = np.zeros(size)
    t, d = 0.0, 0.0
    log = []
    samples = np.asarray(t_samples, dtype=float)
    out = np.full((samples.size, size), np.nan)
    directions = np.zeros(samples.size)
    n = circuit.size
    edges = np.asarray(sorted(breaks), dtype=float) if breaks is not None else None
    source = u_primary
    segment_end = t_end
    for _ in range(max_events):
        if t >= t_end:
            break
        if edges is not None:
            k = int(np.searchsorted(edges, t * (1.0 + 1e-14) + 1e-18, side="right"))
            segment_end = min(t_end, edges[k]) if k < edges.size else t_end
            value = u_primary(0.5 * (t + segment_end))
            source = lambda _tt, value=value: value  # noqa: E731
            if d == 0.0:
                u_hold = circuit.hold_voltage(x, value)
                if abs(u_hold) > V2:
                    d = -math.copysign(1.0, u_hold)
                    log.append((t, d))
        u_source = source
        if d == 0.0:
            def f(tt, xx):
                u1 = u_source(tt)
                return circuit.derivative(xx, u1, circuit.hold_voltage(xx, u1))

            def event(tt, xx):
                return V2 - abs(circuit.hold_voltage(xx, u_source(tt)))
        else:
            def f(tt, xx, d=d):
                dx = circuit.derivative(xx[:n], u_source(tt), -d * V2)
                return dx

            def event(tt, xx, d=d):
                return d * circuit.port_current(xx[:n], "secondary")
        event.terminal, event.direction = True, -1
        sol = solve_ivp(f, (t, segment_end), x, method="DOP853", rtol=rtol, atol=atol,
                        events=event, dense_output=True)
        t_stop = sol.t[-1]
        mask = (samples >= t) & (samples <= t_stop)
        if mask.any():
            out[mask] = sol.sol(samples[mask]).T
            directions[mask] = d
        x = sol.y[:, -1].copy()
        if sol.status != 1:
            t = t_stop
            continue
        t = t_stop
        u_hold = circuit.hold_voltage(x, u_primary(t))
        if d == 0.0:
            d = -math.copysign(1.0, u_hold)
        else:
            d = -d if abs(u_hold) > V2 else 0.0
        log.append((t, d))
    port = np.array([circuit.port_current(sample[:n], "secondary") for sample in out])
    i_dc = -np.abs(port) * (directions != 0.0)
    return out, i_dc, log
