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
unique.  The code therefore diagonalizes the Hermitian projected e-ph matrix
inside each degenerate frequency subspace before assigning mode linewidths.
This makes the linewidth eigenvalues invariant to arbitrary rotations of the
phonon eigenvectors inside the degenerate manifold.
"""

from dataclasses import dataclass
from typing import Sequence

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
        # CellConstructor convention: k_points = qpoints @ reciprocal_lattice.
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
            # The eigenvalues of this restricted self-energy/coupling block do
            # not depend on the arbitrary phonon basis chosen by diagonalization.
            values[group] = np.linalg.eigvalsh(block)
    return values


def calculate_solver_linewidth_path(
    solver,
    qpoints: Sequence[Sequence[float]],
    *,
    smear_id: int = 0,
    coordinates: str = "cartesian",
    scattering_mesh=(10, 10, 10),
    a2f_smearing: float = 0.5,
    block_size: int = 64,
    validate_symmetry: bool = False,
    frequency_tol: float = 1.0e-12,
    degeneracy_atol: float = 1.0e-8,
    degeneracy_rtol: float = 1.0e-5,
    negative_tol: float = 1.0e-10,
):
    """Calculate electron-phonon phonon linewidths along an arbitrary q path.

    Parameters
    ----------
    solver
        A loaded ``dense_mesolver.mesolver`` instance.
    qpoints
        Path points. With ``coordinates='cartesian'`` these must use the same
        CellConstructor reciprocal convention as ``tc.k_points`` and
        ``ForceTensor.get_phonons_in_qpath`` (reciprocal lattice without an
        extra 2*pi). With ``coordinates='fractional'`` they are fractional
        reciprocal coordinates.
    smear_id
        Electronic smearing/DOS index used for the deformation matrix and DOS.
    degeneracy_atol, degeneracy_rtol
        Frequency tolerances for grouping degenerate/near-degenerate phonons.

    Returns
    -------
    PhononLinewidthPath
        Frequencies, projected mode deformation values, lambda(q,nu), and
        linewidth HWHM gamma(q,nu), all in native CellConstructor energy units
        except lambda, which is dimensionless. Unstable/zero modes are NaN.
    """
    if solver.multiband:
        raise NotImplementedError("linewidth path currently supports isotropic input only")
    if smear_id < 0 or smear_id >= solver.ep_nsmear:
        raise IndexError("smear_id is outside the available electron-phonon smearings")

    tc = solver._build_coarse_tc(scattering_mesh, a2f_smearing)
    real_space = solver._prepare_dense_real_space(tc, validate_symmetry)
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

            # A Fermi-surface |g|^2 integral is positive semidefinite.  Permit
            # tiny negative interpolation/roundoff noise, but fail loudly for
            # a significant negative eigenvalue of the coupling block.
            if np.any(mat < -negative_tol):
                raise RuntimeError(
                    "interpolated mode deformation matrix has a significantly "
                    "negative eigenvalue (minimum %.6e)" % float(np.min(mat))
                )
            mat = np.where(mat < 0.0, 0.0, mat)

            valid = freq > frequency_tol
            lam = np.full(freq.shape, np.nan, dtype=float)
            gam = np.full(freq.shape, np.nan, dtype=float)
            lam[valid] = mat[valid] / (2.0 * dos * freq[valid] ** 2)
            # EPW gamma is the half-width. This identity follows from Eq. 25
            # and the SolveME definition of the mode-resolved lambda above.
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
