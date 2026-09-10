import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elph_grid import ElphGrid, generate_regular_q_grid
from elph_interpolation import (
    displacement_gauge_phase,
    elph_to_real_space,
    evaluate_real_space,
    fourier_interpolate_elph,
    from_periodic_gauge,
    interpolate_elph,
    to_periodic_gauge,
)


def make_scalar_grid(mesh=(2, 2, 2), shift=(0.0, 0.0, 0.0)):
    qpoints, _ = generate_regular_q_grid(mesh, shift)
    values = np.zeros((len(qpoints), 1, 1), dtype=complex)
    # Smooth periodic function exactly representable on the coarse mesh.
    values[:, 0, 0] = (
        2.0
        + 0.5 * np.exp(2.0j * np.pi * qpoints[:, 0])
        + 0.25 * np.exp(-2.0j * np.pi * qpoints[:, 1])
    )
    return ElphGrid(qpoints=qpoints, matrices=values, mesh=mesh, shift=shift)


def test_displacement_gauge_phase_has_expected_atom_blocks():
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    phase = displacement_gauge_phase([0.5, 0.0, 0.0], tau)
    assert phase.shape == (6, 6)
    np.testing.assert_allclose(phase[:3, :3], 1.0)
    expected = np.exp(+2.0j * np.pi * 0.5 * 0.25)
    np.testing.assert_allclose(phase[:3, 3:], expected)
    np.testing.assert_allclose(phase[3:, :3], expected.conjugate())


def test_periodic_gauge_round_trip():
    qpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    tau = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]])
    rng = np.random.default_rng(7)
    values = rng.normal(size=(2, 6, 6)) + 1.0j * rng.normal(size=(2, 6, 6))
    periodic = to_periodic_gauge(values, qpoints, tau)
    restored = from_periodic_gauge(periodic, qpoints, tau)
    np.testing.assert_allclose(restored, values, atol=1.0e-13)


def test_q_to_r_to_original_q_round_trip_gamma_centered():
    grid = make_scalar_grid(mesh=(3, 2, 2))
    real_space = elph_to_real_space(grid)
    recovered = evaluate_real_space(real_space, grid.qpoints)
    np.testing.assert_allclose(recovered, grid.matrices, atol=1.0e-12)


def test_q_to_r_to_original_q_round_trip_shifted_mesh():
    grid = make_scalar_grid(mesh=(3, 2, 2), shift=(0.5, 0.5, 0.0))
    real_space = elph_to_real_space(grid)
    recovered = evaluate_real_space(real_space, grid.qpoints)
    np.testing.assert_allclose(recovered, grid.matrices, atol=1.0e-12)


def test_round_trip_with_atomic_gauge():
    mesh = (2, 2, 2)
    qpoints, _ = generate_regular_q_grid(mesh)
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
    rng = np.random.default_rng(11)

    # Construct a periodic-gauge field, then convert it to the SolveME/input
    # gauge before passing it through the interpolation machinery.
    periodic = rng.normal(size=(len(qpoints), 6, 6)) + 1.0j * rng.normal(
        size=(len(qpoints), 6, 6)
    )
    input_gauge = from_periodic_gauge(periodic, qpoints, tau)
    grid = ElphGrid(qpoints=qpoints, matrices=input_gauge, mesh=mesh)

    real_space = elph_to_real_space(grid, atom_positions=tau)
    recovered = evaluate_real_space(real_space, qpoints)
    np.testing.assert_allclose(recovered, input_gauge, atol=1.0e-12)


def test_interpolated_grid_recovers_original_coarse_points():
    coarse = make_scalar_grid(mesh=(2, 2, 2))
    real_space = elph_to_real_space(coarse)
    fine = interpolate_elph(real_space, mesh=(4, 4, 4))

    # Every coarse point appears exactly on the doubled grid.
    fine_lookup = {
        tuple(np.rint(q * np.array(fine.mesh)).astype(int) % np.array(fine.mesh)): i
        for i, q in enumerate(fine.qpoints)
    }
    for iq, q in enumerate(coarse.qpoints):
        key = tuple(np.rint(q * np.array(fine.mesh)).astype(int) % np.array(fine.mesh))
        np.testing.assert_allclose(fine.matrices[fine_lookup[key]], coarse.matrices[iq], atol=1.0e-12)


def test_convenience_interpolator_matches_direct_evaluation():
    coarse = make_scalar_grid(mesh=(2, 2, 2))
    fine = fourier_interpolate_elph(coarse, target_mesh=(4, 3, 2))
    real_space = elph_to_real_space(coarse)
    direct = evaluate_real_space(real_space, fine.qpoints)
    np.testing.assert_allclose(fine.matrices, direct, atol=1.0e-12)
