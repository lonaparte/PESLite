"""Independent reference for square-wave phase-shift modulation, by time stepping.

The reference never solves for an edge: it integrates the angle axis on a fine time grid and
evaluates the bridge level pointwise, so it shares no edge arithmetic with the modulator under
test. Levels are +1 (positive pulse), -1 (negative pulse) and 0.

Register semantics: ``width`` is the pulse width in half periods, ``phase`` the angle of the
positive pulse centre, ``frequency`` the angle-axis frequency (acting at its load), ``mode`` one
of 0 (bipolar pulses), 1 (positive pulse only), 3 (zero level) and 4 (negative pulse only,
centred half a period after ``phase``).

With pulse synchronisation a register change never cuts a pulse: a pulse in progress at the load
continues under the old registers until it ends; the output then stays at zero until the next
pulse of the new registers starts. The output before the first load is zero, so the first pulse
is also whole.
"""

from __future__ import annotations

import math

import numpy as np


def level(mode: float, width: float, y: float) -> int:
    """Bridge level at angle ``y`` (rad) measured from the phase register."""
    h = 0.5 * math.pi * min(1.0, max(0.0, width))
    if mode == 3 or h <= 0.0:
        return 0
    a = (y + math.pi) % (2.0 * math.pi) - math.pi            # in [-pi, pi)
    b = y % (2.0 * math.pi) - math.pi                         # y - pi in [-pi, pi)
    if mode in (0, 1) and -h <= a < h:
        return 1
    if mode in (0, 4) and -h <= b < h:
        return -1
    return 0


def waveform(loads, t_end: float, dt: float, pulse: bool):
    """Sampled level on ``t = t0 + k dt`` for register loads ``[(t, (w, phi, f, mode)), ...]``.

    The first load starts the run with the angle axis at ``2 pi f t0``. Returns ``(t, level)``.
    """
    loads = sorted(loads, key=lambda item: item[0])
    t0 = loads[0][0]
    n = int(math.floor((t_end - t0) / dt)) + 1
    times = t0 + dt * np.arange(n)
    out = np.zeros(n)
    theta = 2.0 * math.pi * loads[0][1][2] * t0
    k_load = 0
    frequency = loads[0][1][2]
    governing = None            # (w, phi, mode) producing the output
    target = None
    stage = "follow"            # follow | finish | wait
    previous_level = 0
    previous_target_level = 0
    for i, t in enumerate(times):
        if i:
            theta += 2.0 * math.pi * frequency * dt
        while k_load < len(loads) and loads[k_load][0] <= t + 1e-15:
            w, phi, f, mode = loads[k_load][1]
            frequency = f
            new = (w, phi, mode)
            if governing is None and pulse:
                governing = target = (0.0, 0.0, 3)     # zero output before the first load
            if not pulse:
                governing, target, stage = new, new, "follow"
            elif new != target or stage != "follow":
                current = (0 if stage == "wait"
                           else level(governing[2], governing[0], theta - governing[1]))
                target = new
                stage = "finish" if current != 0 else "wait"
                previous_target_level = level(new[2], new[0], theta - new[1])
            k_load += 1
        if stage == "finish":
            value = level(governing[2], governing[0], theta - governing[1])
            if value != previous_level:
                stage = "wait"
                previous_target_level = level(target[2], target[0], theta - target[1])
        if stage == "wait":
            value = level(target[2], target[0], theta - target[1])
            if value != 0 and value != previous_target_level:
                governing, stage = target, "follow"
            elif target[0] <= 0.0 or target[2] == 3:
                governing, stage = target, "follow"
            previous_target_level = value
        if stage == "follow":
            value = level(governing[2], governing[0], theta - governing[1])
        elif stage == "wait":
            value = 0
        out[i] = value
        previous_level = value
    return times, out
