"""Phonon linewidths from interpolated electron-phonon deformation matrices.

For the isotropic SolveME input, the projected mode quantity M(q,nu) is the
same Fermi-surface-integrated deformation-potential object used in the
existing mode-resolved coupling

    lambda(q,nu) = M(q,nu) / [2 N_F omega(q,nu)^2].

Combining this with EPW Eq. (25),

    lambda(q,nu) = gamma(q,nu) / [pi hbar N_F omega(q,nu)^2],

gives, in CellConstructor/SolveME energy-frequency units (hbar = 1),

    gamma(q,nu) = pi M(q,nu) / 2,

where gamma is the HWHM.  The FWHM is 2*gamma.
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
):
    """Calculate electron-phonon phonon linewidths along an arbitrary q path.

    Parameters
    ----------
    solver
        A loaded ``dense_mesolver.mesolver`` instance.
    qpoints
        Path points.  With ``coordinates='cartesian'`` these must use the same
        CellConstructor reciprocal convention as ``tc.k_points`` and
        ``ForceTensor.get_phonons_in_qpath`` (reciprocal lattice without an
        extra 2*pi).  With ``coordinates='fractional'`` they are fractional
        reciprocal coordinates.
    smear_id
        Electronic smearing/DOS index used for the deformation matrix and DOS.

    Returns
    -------
    PhononLinewidthPath
        Frequencies, projected mode deformation values, lambda(q,nu), and
        linewidth HWHM gamma(q,nu), all in the native CellConstructor energy
        units except lambda, which is dimensionless.
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
            mode_matrix = np.asarray(block.mode_matrices[iloc])
            # Isotropic input shape at one q is (nsmear, nmode, nmode).
            mat = np.diagonal(mode_matrix, axis1=-2, axis2=-1)[smear_id].real

            valid = freq > frequency_tol
            lam = np.full(freq.shape, np.nan, dtype=float)
            gam = np.full(freq.shape, np.nan, dtype=float)
            lam[valid] = mat[valid] / (2.0 * dos * freq[valid] ** 2)
            # EPW gamma is the half-width.  This identity follows from Eq. 25
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
