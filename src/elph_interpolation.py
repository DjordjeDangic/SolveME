"""Fourier interpolation of electron-phonon matrices on regular q grids.

The default interpolation convention is the same periodic Bloch displacement
convention used by CellConstructor's symmetry machinery.  In this convention
no additional basis-position phase is inserted before the q-to-R transform.

An older/experimental basis-phase conversion remains available explicitly via
``gauge='basis'``:

    D_periodic(q) = exp[-2 pi i q.(tau_kappa - tau_kappa')] D_input(q)

This is not applied by default because the matrices reconstructed with
``ThermalConductivity.construct_symmetry_matrix`` already transform in the
CellConstructor Bloch convention.  Applying a second basis phase can therefore
double-count the positional phase.

The Fourier convention is

    D(R) = 1/Nq sum_q D(q) exp[-2 pi i q.R]
    D(q) = sum_R D(R) exp[+2 pi i q.R]

with integer Born-von-Karman lattice vectors R.  Shifted regular meshes are
supported explicitly.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from elph_grid import ElphGrid, canonicalize_qpoints, generate_regular_q_grid, mesh_index_lookup


VALID_GAUGES = ("cellconstructor", "basis")


def _validate_atom_positions(atom_positions: np.ndarray, matrix_size: int) -> np.ndarray:
    tau = np.asarray(atom_positions, dtype=float)
    if tau.ndim != 2 or tau.shape[1] != 3:
        raise ValueError("atom_positions must have shape (natom, 3)")
    if matrix_size != 3 * len(tau):
        raise ValueError(
            "matrix trailing dimensions must be (3*natom, 3*natom) when "
            "atom_positions are supplied"
        )
    return tau


def displacement_gauge_phase(qpoint: Sequence[float], atom_positions: np.ndarray) -> np.ndarray:
    """Return the optional basis-position phase matrix for one q-point."""
    q = np.asarray(qpoint, dtype=float)
    tau = np.asarray(atom_positions, dtype=float)
    if q.shape != (3,):
        raise ValueError("qpoint must contain three fractional coordinates")
    if tau.ndim != 2 or tau.shape[1] != 3:
        raise ValueError("atom_positions must have shape (natom, 3)")

    atom_phase = np.exp(-2.0j * np.pi * np.einsum("ad,d->a", tau, q))
    cart_phase = np.repeat(atom_phase, 3)
    return cart_phase[:, None] * cart_phase[None, :].conj()


def to_periodic_gauge(
    matrices: np.ndarray,
    qpoints: np.ndarray,
    atom_positions: Optional[np.ndarray],
) -> np.ndarray:
    """Apply the optional legacy basis-position phase conversion."""
    values = np.asarray(matrices, dtype=complex)
    qpoints = np.asarray(qpoints, dtype=float)
    if values.shape[0] != len(qpoints):
        raise ValueError("matrices and qpoints must have the same leading dimension")
    if atom_positions is None:
        return values.copy()
    if values.ndim < 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError("matrices must have square trailing matrix dimensions")

    tau = _validate_atom_positions(atom_positions, values.shape[-1])
    out = values.copy()
    for iq, qpoint in enumerate(qpoints):
        out[iq] *= displacement_gauge_phase(qpoint, tau)
    return out


def from_periodic_gauge(
    matrices: np.ndarray,
    qpoints: np.ndarray,
    atom_positions: Optional[np.ndarray],
) -> np.ndarray:
    """Undo the optional legacy basis-position phase conversion."""
    values = np.asarray(matrices, dtype=complex)
    qpoints = np.asarray(qpoints, dtype=float)
    if values.shape[0] != len(qpoints):
        raise ValueError("matrices and qpoints must have the same leading dimension")
    if atom_positions is None:
        return values.copy()
    if values.ndim < 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError("matrices must have square trailing matrix dimensions")

    tau = _validate_atom_positions(atom_positions, values.shape[-1])
    out = values.copy()
    for iq, qpoint in enumerate(qpoints):
        out[iq] *= displacement_gauge_phase(qpoint, tau).conj()
    return out


def _grid_ordered_values(grid: ElphGrid) -> np.ndarray:
    if grid.mesh is None:
        raise ValueError("Fourier interpolation requires a complete regular q grid")
    lookup = mesh_index_lookup(grid.qpoints, grid.mesh, grid.shift)
    ordered = np.empty(tuple(grid.mesh) + grid.matrices.shape[1:], dtype=complex)
    for index in np.ndindex(*grid.mesh):
        ordered[index] = grid.matrices[lookup[index]]
    return ordered


def real_space_vectors(mesh: Sequence[int]) -> np.ndarray:
    """Return signed integer BvK real-space vectors in FFT ordering."""
    mesh = tuple(int(n) for n in mesh)
    if len(mesh) != 3 or any(n <= 0 for n in mesh):
        raise ValueError("mesh must contain three positive integers")
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in mesh]
    rr = np.meshgrid(*axes, indexing="ij")
    return np.stack(rr, axis=-1)


@dataclass
class ElphRealSpace:
    """Real-space representation used to evaluate/interpolate e-ph matrices."""

    matrices: np.ndarray
    mesh: Tuple[int, int, int]
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    atom_positions: Optional[np.ndarray] = None
    gauge: str = "cellconstructor"

    def __post_init__(self):
        self.mesh = tuple(int(n) for n in self.mesh)
        self.shift = tuple(float(x) for x in self.shift)
        self.matrices = np.asarray(self.matrices, dtype=complex)
        if self.matrices.shape[:3] != self.mesh:
            raise ValueError("real-space matrix leading dimensions must equal mesh")
        if self.gauge not in VALID_GAUGES:
            raise ValueError("gauge must be one of %s" % (VALID_GAUGES,))
        if self.atom_positions is not None:
            self.atom_positions = np.asarray(self.atom_positions, dtype=float)

    @property
    def r_vectors(self) -> np.ndarray:
        return real_space_vectors(self.mesh)


def elph_to_real_space(
    grid: ElphGrid,
    atom_positions: Optional[np.ndarray] = None,
    gauge: str = "cellconstructor",
) -> ElphRealSpace:
    """Transform a complete coarse q grid to real space.

    ``gauge='cellconstructor'`` is the default and applies no extra basis phase.
    ``gauge='basis'`` enables the older explicit basis-position conversion and
    requires fractional direct ``atom_positions``.
    """
    if grid.mesh is None:
        raise ValueError("grid.mesh is required for a q-to-R transform")
    if gauge not in VALID_GAUGES:
        raise ValueError("gauge must be one of %s" % (VALID_GAUGES,))
    if gauge == "basis" and atom_positions is None:
        raise ValueError("atom_positions are required when gauge='basis'")

    gauge_positions = atom_positions if gauge == "basis" else None
    periodic = to_periodic_gauge(grid.matrices, grid.qpoints, gauge_positions)
    periodic_grid = ElphGrid(
        qpoints=grid.qpoints,
        matrices=periodic,
        mesh=grid.mesh,
        shift=grid.shift,
    )
    ordered = _grid_ordered_values(periodic_grid)
    nq = float(np.prod(grid.mesh))
    d_r = np.fft.fftn(ordered, axes=(0, 1, 2)) / nq

    r = real_space_vectors(grid.mesh)
    shift = np.asarray(grid.shift, dtype=float)
    mesh = np.asarray(grid.mesh, dtype=float)
    shift_phase = np.exp(-2.0j * np.pi * np.einsum("...d,d->...", r, shift / mesh))
    d_r *= shift_phase[(...,) + (None,) * (d_r.ndim - 3)]

    return ElphRealSpace(
        matrices=d_r,
        mesh=grid.mesh,
        shift=grid.shift,
        atom_positions=gauge_positions,
        gauge=gauge,
    )


def evaluate_real_space(
    real_space: ElphRealSpace,
    qpoints: np.ndarray,
    restore_gauge: bool = True,
) -> np.ndarray:
    """Evaluate a real-space representation at arbitrary fractional q-points."""
    qpoints = canonicalize_qpoints(np.asarray(qpoints, dtype=float))
    if qpoints.ndim == 1:
        qpoints = qpoints[None, :]
    if qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise ValueError("qpoints must have shape (nq, 3)")

    r = real_space.r_vectors.reshape(-1, 3)
    d_r = real_space.matrices.reshape((-1,) + real_space.matrices.shape[3:])
    phase = np.exp(2.0j * np.pi * np.einsum("qd,rd->qr", qpoints, r))
    periodic = np.einsum("qr,r...->q...", phase, d_r, optimize=True)

    if restore_gauge and real_space.gauge == "basis":
        return from_periodic_gauge(periodic, qpoints, real_space.atom_positions)
    return periodic


def interpolate_elph(
    real_space: ElphRealSpace,
    mesh: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> ElphGrid:
    """Evaluate a real-space e-ph representation on a denser regular q mesh."""
    target_q, _ = generate_regular_q_grid(mesh, shift)
    values = evaluate_real_space(real_space, target_q, restore_gauge=True)
    return ElphGrid(
        qpoints=target_q,
        matrices=values,
        mesh=tuple(int(n) for n in mesh),
        shift=tuple(float(x) for x in shift),
    )


def fourier_interpolate_elph(
    coarse_grid: ElphGrid,
    target_mesh: Sequence[int],
    target_shift: Sequence[float] = (0.0, 0.0, 0.0),
    atom_positions: Optional[np.ndarray] = None,
    gauge: str = "cellconstructor",
) -> ElphGrid:
    """Convenience q-grid -> real-space -> dense-q interpolation pipeline."""
    real_space = elph_to_real_space(
        coarse_grid,
        atom_positions=atom_positions,
        gauge=gauge,
    )
    return interpolate_elph(real_space, target_mesh, target_shift)
