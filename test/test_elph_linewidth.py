import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elph_linewidth import PhononLinewidthPath, _degeneracy_safe_mode_deformation


def test_fwhm_property_is_twice_hwhm():
    result = PhononLinewidthPath(
        qpoints_fractional=np.zeros((1, 3)),
        frequencies=np.array([[1.0, 2.0]]),
        mode_deformation=np.array([[0.1, 0.2]]),
        lambdas=np.array([[0.3, 0.4]]),
        gamma_hwhm=np.array([[0.5, 0.7]]),
    )
    np.testing.assert_allclose(result.gamma_fwhm, [[1.0, 1.4]])


def test_epw_eq25_matches_solvemode_lambda_identity():
    # SolveME uses lambda = M / (2 N_F omega^2). Combining this with
    # EPW Eq. 25 gives gamma = pi M / 2 (hbar=1 in native energy units).
    deformation = np.array([0.04, 0.11, 0.20])
    omega = np.array([0.03, 0.07, 0.15])
    dos = 2.3

    lam = deformation / (2.0 * dos * omega**2)
    gamma = 0.5 * np.pi * deformation
    reconstructed = gamma / (np.pi * dos * omega**2)

    np.testing.assert_allclose(reconstructed, lam)


def test_degenerate_subspace_is_basis_invariant():
    freq = np.array([1.0, 1.0, 2.0])
    coupling = np.array(
        [[2.0, 0.5, 0.0], [0.5, 4.0, 0.0], [0.0, 0.0, 7.0]], dtype=complex
    )
    expected_pair = np.linalg.eigvalsh(coupling[:2, :2])

    angle = 0.37
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed = coupling.copy()
    transformed[:2, :2] = rot.T @ coupling[:2, :2] @ rot

    values = _degeneracy_safe_mode_deformation(
        transformed, freq, atol=1.0e-10, rtol=1.0e-10
    )
    np.testing.assert_allclose(values[:2], expected_pair)
    np.testing.assert_allclose(values[2], 7.0)
