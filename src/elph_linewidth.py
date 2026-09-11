"""Phonon linewidths from interpolated electron-phonon deformation matrices.

For the isotropic SolveME input, the projected mode quantity M(q,nu) is the
same Fermi-surface-integrated deformation-potential object used in the
existing mode-resolved coupling

    lambda(q,nu) = M(q,nu) / [2 N_F omega(q,nu)^2].

Combining this with EPW Eq. (25),

    lambda(q,nu) = gamma(q,nu) / [pi hbar N_F omega(q,nu)^2],

gives, in CellConstructor/SolveME energy-frequency units (hbar = 1),

    gamma(q,nu) = pi M(q,nu) / 2,

where gamma is the HWHM. The FWHM is 2*gamma.

At exact or near phonon degeneracies, individual phonon eigenvectors are not
unique. The code therefore diagonalizes the Hermitian projected e-ph matrix
inside each degenerate frequency subspace before assigning mode linewidths.
This makes the linewidth eigenvalues invariant to arbitrary rotations of the
phonon eigenvectors inside the degenerate manifold.
"""

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np

from elph_dense import iter_dense_elph_qpoints
from elph_grid import canonicalize_qpoints


@dataclass
class PhononLinewidthPath:
    qpoints_fractional: np.ndarray
    frequencies: np.ndarray
    mode_deformation: np.ndarray
    lambdas: np.ndarray
    gamma_hwhm: np.ndarray

    @property
    def gamma_fwhm(self):
        return 2.0 * self.gamma_hwhm


def _as_fractional_qpoints(tc, qpoints, coordinates):
    qpoints = np.asarray(qpoints, dtype=float)
    if qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise ValueError("qpoints must have shape (nq, 3)")

    if coordinates == "fractional":
        qfrac = qpoints
    elif coordinates == "cartesian":
        qfrac = np.dot(qpoints, np.linalg.inv(np.asarray(tc.reciprocal_lattice)))
    else:
        raise ValueError("coordinates must be 'fractional' or 'cartesian'")
    return canonicalize_qpoints(qfrac)


def _degenerate_groups(freq, atol, rtol):
    freq = np.asarray(freq, dtype=float)
    groups = []
    start = 0
    for i in range(1, len(freq)):
        scale = max(abs(freq[i]), abs(freq[i - 1]), 1.0e-14)
        if abs(freq[i] - freq[i - 1]) > atol + rtol * scale:
            groups.append(slice(start, i))
            start = i
    groups.append(slice(start, len(freq)))
    return groups


def _degeneracy_safe_mode_deformation(mode_matrix, freq, atol, rtol):
    """Return basis-invariant mode couplings inside degenerate subspaces."""
    matrix = np.asarray(mode_matrix, dtype=complex)
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values = np.empty(len(freq), dtype=float)

    for group in _degenerate_groups(freq, atol, rtol):
        block = hermitian[group, group]
        if block.shape == (1, 1):
            values[group] = block[0, 0].real
        else:
            values[group] = np.linalg.eigvalsh(block)
    return values


def _clip_small_negative_deformation(values, atol=1.0e-10, rtol=1.0e-5):
    """Clip small negative interpolation noise while rejecting large PSD violations.

    The deformation matrix represents a Fermi-surface |g|^2 covariance and is
    therefore positive semidefinite in the exact theory. Fourier interpolation
    and floating-point roundoff can nevertheless produce tiny negative mode
    eigenvalues. A negative value is accepted as numerical noise when

        |value_min| <= max(atol, rtol * max(|values|)).

    Accepted negatives are clipped to zero and reported with ``RuntimeWarning``.
    Larger negative values still raise ``RuntimeError``.
    """
    values = np.asarray(values, dtype=float)
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("negative tolerances must be non-negative")
    if values.size == 0:
        return values.copy()

    minimum = float(np.min(values))
    if minimum >= 0.0:
        return values.copy()

    scale = max(float(np.max(np.abs(values))), 1.0e-14)
    tolerance = max(float(atol), float(rtol) * scale)

    if minimum < -tolerance:
        raise RuntimeError(
            "interpolated mode deformation matrix has a significantly negative "
            "eigenvalue (minimum %.6e, tolerance %.6e, scale %.6e)"
            % (minimum, tolerance, scale)
        )

    warnings.warn(
        "Clipping small negative interpolated e-ph mode deformation values "
        "(minimum %.6e, tolerance %.6e, scale %.6e)"
        % (minimum, tolerance, scale),
        RuntimeWarning,
        stacklevel=2,
    )
    return np.where(values < 0.0, 0.0, values)


def calculate_solver_linewidth_path(
    solver,
    qpoints: Sequence[Sequence[float]],
    *,
    smear_id: int = 0,
    coordinates: str = "cartesian",
    scattering_mesh=(10, 10, 10),
    a2f_smearing: float = 0.5,
    block_size: int = 64,
    validate_symmetry: bool = True,
    elph_inverse_symmetry: bool = False,
    elph_reciprocal_gauge: bool = False,
    frequency_tol: float = 1.0e-12,
    degeneracy_atol: float = 1.0e-8,
    degeneracy_rtol: float = 1.0e-5,
    negative_atol: float = 1.0e-10,
    negative_rtol: float = 1.0e-5,
):
    """Calculate electron-phonon phonon linewidths along an arbitrary q path.

    ``elph_inverse_symmetry`` and ``elph_reciprocal_gauge`` are experimental
    e-ph-only convention switches. Neither changes the CellConstructor
    dynamical-matrix symmetry validation.

    Small negative interpolated mode-deformation values are treated as numerical
    noise when their magnitude is below
    ``max(negative_atol, negative_rtol * max(abs(mode_deformation)))`` at that q.
    They are clipped to zero with a RuntimeWarning. Larger negative values still
    raise because the exact Fermi-surface |g|^2 covariance is positive semidefinite.
    """
    if solver.multiband:
        raise NotImplementedError("linewidth path currently supports isotropic input only")
    if smear_id < 0 or smear_id >= solver.ep_nsmear:
        raise IndexError("smear_id is outside the available electron-phonon smearings")

    tc = solver._build_coarse_tc(scattering_mesh, a2f_smearing)
    real_space = solver._prepare_dense_real_space(
        tc,
        validate_symmetry,
        elph_inverse_symmetry=elph_inverse_symmetry,
        elph_reciprocal_gauge=elph_reciprocal_gauge,
    )
    qfrac = _as_fractional_qpoints(tc, qpoints, coordinates)

    frequencies = []
    mode_deformation = []
    lambdas = []
    gamma = []
    dos = float(solver.elph_dos[smear_id])

    for block in iter_dense_elph_qpoints(
        real_space,
        tc,
        qfrac,
        block_size=block_size,
        phonon_evaluator=solver._mass_scaled_phonon_evaluator,
    ):
        for iloc in range(len(block.qpoints)):
            freq = np.asarray(block.frequencies[iloc], dtype=float)
            mode_matrix = np.asarray(block.mode_matrices[iloc])[smear_id]
            mat = _degeneracy_safe_mode_deformation(
                mode_matrix,
                freq,
                degeneracy_atol,
                degeneracy_rtol,
            )
            mat = _clip_small_negative_deformation(
                mat,
                atol=negative_atol,
                rtol=negative_rtol,
            )

            valid = freq > frequency_tol
            lam = np.full(freq.shape, np.nan, dtype=float)
            gam = np.full(freq.shape, np.nan, dtype=float)
            lam[valid] = mat[valid] / (2.0 * dos * freq[valid] ** 2)
            gam[valid] = 0.5 * np.pi * mat[valid]

            frequencies.append(freq)
            mode_deformation.append(mat)
            lambdas.append(lam)
            gamma.append(gam)

    return PhononLinewidthPath(
        qpoints_fractional=qfrac,
        frequencies=np.asarray(frequencies),
        mode_deformation=np.asarray(mode_deformation),
        lambdas=np.asarray(lambdas),
        gamma_hwhm=np.asarray(gamma),
    )
