"""Simple pendulum simulation using a fourth-order Runge-Kutta integrator.

This module provides utilities to simulate the motion of a simple pendulum
subject to gravity and (optionally) linear damping.  The integration results can
be plotted or visualised with a 2D animation when the module is executed as a
script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "numpy es un requisito para ejecutar la simulación del péndulo. "
        "Instálalo con 'pip install numpy'."
    ) from exc

try:  # pragma: no cover - matplotlib is optional at runtime
    import matplotlib.pyplot as plt
    from matplotlib import animation
except ModuleNotFoundError:  # pragma: no cover - handled gracefully at runtime
    plt = None
    animation = None


@dataclass(frozen=True)
class PendulumParams:
    """Container for the physical properties of the pendulum.

    Attributes
    ----------
    length:
        Length of the pendulum (meters).
    gravity:
        Gravitational acceleration (m/s^2).
    damping:
        Linear damping coefficient applied to the angular velocity (1/s).
    """

    length: float = 1.0
    gravity: float = 9.81
    damping: float = 0.0


def _derivatives(state: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Return the time derivatives of the pendulum state.

    The state vector is ``[theta, omega]`` where ``theta`` is the angle from the
    vertical and ``omega`` its angular velocity.  The returned vector contains
    ``[d(theta)/dt, d(omega)/dt]``.
    """

    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(params.gravity / params.length) * np.sin(theta) - params.damping * omega
    return np.array([dtheta_dt, domega_dt])


def rk4_step(state: np.ndarray, dt: float, params: PendulumParams) -> np.ndarray:
    """Advance the pendulum state by one time step using RK4 integration."""

    k1 = _derivatives(state, params)
    k2 = _derivatives(state + 0.5 * dt * k1, params)
    k3 = _derivatives(state + 0.5 * dt * k2, params)
    k4 = _derivatives(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_pendulum(
    theta0: float,
    omega0: float,
    total_time: float,
    dt: float,
    params: PendulumParams | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a pendulum and return the time, angle, and angular velocity.

    Parameters
    ----------
    theta0:
        Initial angle (radians).
    omega0:
        Initial angular velocity (radians per second).
    total_time:
        Total duration of the simulation (seconds).
    dt:
        Time step size (seconds).
    params:
        Optional :class:`PendulumParams` instance describing the pendulum.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Time vector, angular displacement, and angular velocity arrays.
    """

    if params is None:
        params = PendulumParams()

    steps = int(total_time / dt) + 1
    times = np.linspace(0.0, total_time, steps)
    state = np.array([theta0, omega0], dtype=float)

    angles = np.empty(steps)
    velocities = np.empty(steps)

    for idx, _ in enumerate(times):
        angles[idx], velocities[idx] = state
        state = rk4_step(state, dt, params)

    return times, angles, velocities


def _plot_simulation(times: np.ndarray, angles: np.ndarray) -> None:
    """Plot the angular displacement of the pendulum over time."""

    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting but is not installed. "
            "Install it with 'pip install matplotlib'."
        )

    plt.figure(figsize=(8, 4))
    plt.plot(times, angles)
    plt.title("Simple Pendulum Simulation")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_example_simulation() -> Tuple[np.ndarray, np.ndarray, np.ndarray, PendulumParams]:
    """Execute a demonstration simulation with typical parameters."""

    params = PendulumParams(length=1.0, gravity=9.81, damping=0.05)
    results = simulate_pendulum(
        theta0=np.deg2rad(20.0),
        omega0=0.0,
        total_time=10.0,
        dt=0.01,
        params=params,
    )
    return (*results, params)


def _animate_pendulum(
    times: np.ndarray,
    angles: np.ndarray,
    params: PendulumParams,
) -> None:
    """Display a 2D animation of the pendulum bob over time."""

    if plt is None or animation is None:
        raise RuntimeError(
            "matplotlib es necesario para la animación 2D pero no está instalado. "
            "Instálalo con 'pip install matplotlib'."
        )

    x_positions = params.length * np.sin(angles)
    y_positions = -params.length * np.cos(angles)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")
    limit = params.length * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Animación de péndulo simple")
    ax.grid(True, linestyle="--", linewidth=0.5)

    rod_line, = ax.plot([], [], lw=2, color="tab:blue")
    bob_marker, = ax.plot([], [], "o", markersize=12, color="tab:orange")
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def init() -> Tuple:
        rod_line.set_data([], [])
        bob_marker.set_data([], [])
        time_text.set_text("")
        return rod_line, bob_marker, time_text

    def update(frame: int) -> Tuple:
        rod_line.set_data([0.0, x_positions[frame]], [0.0, y_positions[frame]])
        bob_marker.set_data(x_positions[frame], y_positions[frame])
        time_text.set_text(f"t = {times[frame]:.2f} s")
        return rod_line, bob_marker, time_text

    if len(times) > 1:
        interval_ms = max(int(1000 * (times[1] - times[0])), 1)
    else:
        interval_ms = 20

    animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(times),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    plt.show()


def _parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    """Return command-line arguments for the script interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Simula un péndulo simple y muestra una animación 2D. "
            "Utiliza --show-plot para incluir la gráfica tradicional."
        )
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Muestra la gráfica del ángulo vs tiempo además de la animación.",
    )
    parsed_args = list(args) if args is not None else None
    return parser.parse_args(parsed_args)


def main(argv: Iterable[str] | None = None) -> None:
    """Run the example simulation and display an animation if possible."""

    args = _parse_args(argv)
    times, angles, _, params = run_example_simulation()

    if plt is None:
        print(
            "Simulación completada. Instala matplotlib para ver la animación 2D "
            "o la gráfica de resultados."
        )
        return

    if animation is None:
        print(
            "matplotlib está disponible pero su módulo de animación no pudo "
            "importarse. Mostrando solo la gráfica tradicional."
        )
        _plot_simulation(times, angles)
        return

    _animate_pendulum(times, angles, params)

    if args.show_plot:
        _plot_simulation(times, angles)


if __name__ == "__main__":
    main()
