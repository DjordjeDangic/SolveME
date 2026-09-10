"""Electron-phonon q-grid data structures and regular-grid utilities.

Conventions
-----------
q-points are stored in fractional reciprocal-lattice coordinates and are
canonicalized to [0, 1).  Electron-phonon matrices use q as their leading
axis.  In SolveME the matrix trailing axes are Cartesian atomic-displacement
indices (atom-major, x/y/z within each atom) before projection onto phonon
modes.  Additional axes (for example smearing or electronic-band indices)
may appear between q and the final two displacement axes.

No Fourier/gauge convention is imposed in this module.  In particular,
atomic-position phase factors are deliberately left to the later Fourier
interpolation milestone.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


_Q_TOL = 1.0e-8


def canonicalize_qpoints(qpoints: np.ndarray, tol: float = _Q_TOL) -> np.ndarray:
    """Return fractional q-points in [0, 1), removing round-off at 0/1."""
    q = np.mod(np.asarray(qpoints, dtype=float), 1.0)
    q[np.isclose(q, 1.0, atol=tol)] = 0.0
    q[np.isclose(q, 0.0, atol=tol)] = 0.0
    return q


def generate_regular_q_grid(
    mesh: Sequence[int], shift: Sequence[float] = (0.0, 0.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a regular reciprocal-space grid.

    Parameters
    ----------
    mesh
        Number of points along each reciprocal-lattice direction.
    shift
        Fractional shift in units of one mesh step.  ``(0, 0, 0)`` is
        Gamma-centered; ``(0.5, 0.5, 0.5)`` is the common half-grid shift.

    Returns
    -------
    qpoints, indices
        q-points in fractional reciprocal coordinates and their integer
        mesh coordinates.  The first axis varies slowest (C ordering).
    """
    mesh = np.asarray(mesh, dtype=int)
    shift = np.asarray(shift, dtype=float)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError("mesh must contain three positive integers")
    if shift.shape != (3,):
        raise ValueError("shift must contain three values")

    indices = np.indices(tuple(mesh), dtype=int).reshape(3, -1).T
    qpoints = canonicalize_qpoints((indices + shift) / mesh)
    return qpoints, indices


def qpoint_to_mesh_index(
    qpoint: Sequence[float],
    mesh: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
    tol: float = 1.0e-7,
) -> Tuple[int, int, int]:
    """Convert a regular-grid q-point to integer mesh coordinates.

    Raises ``ValueError`` when the point is not on the requested mesh.
    """
    mesh = np.asarray(mesh, dtype=int)
    shift = np.asarray(shift, dtype=float)
    q = canonicalize_qpoints(np.asarray(qpoint, dtype=float))
    raw = q * mesh - shift
    nearest = np.rint(raw).astype(int)
    residual = raw - nearest
    # Account for equivalent points differing by a reciprocal lattice vector.
    residual -= np.rint(residual / mesh) * mesh
    if np.max(np.abs(residual)) > tol:
        raise ValueError("q-point is not commensurate with the requested mesh")
    return tuple((nearest % mesh).tolist())


def mesh_index_lookup(
    qpoints: np.ndarray,
    mesh: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
    tol: float = 1.0e-7,
):
    """Build an integer-key lookup for q-points on a regular mesh."""
    lookup = {}
    for iq, qpoint in enumerate(np.asarray(qpoints)):
        key = qpoint_to_mesh_index(qpoint, mesh, shift, tol)
        if key in lookup:
            raise ValueError("duplicate q-point on regular mesh: %s" % (key,))
        lookup[key] = iq
    return lookup


@dataclass(frozen=True)
class QPointMapping:
    """Recipe for reconstructing one full-grid q-point from an irreducible one."""

    target_index: int
    irred_index: int
    symmetry_index: Optional[int]
    time_reversal: bool
    reciprocal_shift: np.ndarray


@dataclass
class ElphGrid:
    """Electron-phonon matrices together with their reciprocal-space grid."""

    qpoints: np.ndarray
    matrices: np.ndarray
    mesh: Optional[Tuple[int, int, int]] = None
    weights: Optional[np.ndarray] = None
    smearings: Optional[np.ndarray] = None
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        self.qpoints = canonicalize_qpoints(self.qpoints)
        self.matrices = np.asarray(self.matrices)
        if self.qpoints.ndim != 2 or self.qpoints.shape[1] != 3:
            raise ValueError("qpoints must have shape (nq, 3)")
        if self.matrices.shape[0] != self.qpoints.shape[0]:
            raise ValueError("matrices and qpoints must have the same leading dimension")
        if self.mesh is not None:
            self.mesh = tuple(int(n) for n in self.mesh)
            expected = int(np.prod(self.mesh))
            if expected != len(self.qpoints):
                raise ValueError("a full regular grid must contain prod(mesh) q-points")
            # Validate commensurability and uniqueness immediately.
            mesh_index_lookup(self.qpoints, self.mesh, self.shift)
        if self.weights is not None and len(self.weights) != len(self.qpoints):
            raise ValueError("weights and qpoints must have the same length")

    @property
    def nq(self) -> int:
        return self.qpoints.shape[0]
