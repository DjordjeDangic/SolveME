"""Dense-q electron-phonon evaluation utilities.

This module connects the Cartesian electron-phonon Fourier interpolator to
phonon eigenmodes. The interpolation remains in Cartesian atomic-displacement
space until the target q point is reached; only then is the matrix projected
onto phonon eigenvectors.

The public iterator API processes q points in blocks so dense meshes need not
materialize the complete interpolated electron-phonon tensor in memory.
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence, Tuple

import numpy as np

from elph_grid import generate_regular_q_grid
from elph_interpolation import ElphRealSpace, evaluate_real_space


@dataclass
class DenseElphBlock:
    """One block of dense-q electron-phonon data."""

    qpoints: np.ndarray
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    cartesian_matrices: np.ndarray
    mode_matrices: np.ndarray
    start: int
    stop: int


def _normalize_eigenvectors(eigenvectors: np.ndarray, ndof: int) -> np.ndarray:
    """Return phonon eigenvectors with shape ``(ndof, nmode)``."""
    pol = np.asarray(eigenvectors, dtype=complex)
    if pol.ndim != 2:
        raise ValueError("phonon eigenvectors must be a 2-D array")
    if pol.shape[0] != ndof:
        if pol.shape[1] == ndof:
            pol = pol.T
        else:
            raise ValueError("phonon eigenvector dimension does not match e-ph matrix")
    return pol


def evaluate_phonons_at_q(
    phonons,
    qpoint: Sequence[float],
    phonon_evaluator: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate phonon frequencies/eigenvectors at one fractional q point.

    The q-vector accepted by this public helper is always in fractional
    reciprocal-lattice coordinates, matching the e-ph interpolation layer.

    For a ``ThermalConductivity`` object, ``get_frequency_at_q`` expects the
    corresponding Cartesian reciprocal vector (the same convention as
    ``tc.k_points``).  Therefore this helper converts

        q_cart = q_fractional @ tc.reciprocal_lattice

    before calling it.  ``DyagDinQ`` is deliberately not used for arbitrary
    q-vectors because CellConstructor's ``Phonons.DyagDinQ`` expects an integer
    index into the stored q-point list.

    A custom ``phonon_evaluator(phonons, qpoint)`` receives the original
    fractional q-vector and may return either ``(frequencies, eigenvectors)``
    or a longer tuple whose first two entries have those meanings.
    """
    q = np.asarray(qpoint, dtype=float)
    if q.shape != (3,):
        raise ValueError("qpoint must contain three fractional coordinates")

    if phonon_evaluator is not None:
        result = phonon_evaluator(phonons, q)
    elif hasattr(phonons, "get_frequency_at_q"):
        if not hasattr(phonons, "reciprocal_lattice"):
            raise AttributeError(
                "get_frequency_at_q is available but reciprocal_lattice is missing; "
                "cannot convert fractional q to the expected Cartesian vector"
            )
        q_cart = np.dot(q, np.asarray(phonons.reciprocal_lattice, dtype=float))
        result = phonons.get_frequency_at_q(q_cart)
    elif hasattr(phonons, "DiagonalizeQPoint"):
        result = phonons.DiagonalizeQPoint(q)
    elif hasattr(phonons, "diagonalize_qpoint"):
        result = phonons.diagonalize_qpoint(q)
    else:
        raise AttributeError(
            "phonon object has no recognized arbitrary-q diagonalization method; "
            "pass a ThermalConductivity object or provide phonon_evaluator"
        )

    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise ValueError("phonon evaluator must return at least (frequencies, eigenvectors)")
    frequencies = np.asarray(result[0], dtype=float)
    eigenvectors = np.asarray(result[1], dtype=complex)
    if frequencies.ndim != 1:
        raise ValueError("phonon frequencies must be one-dimensional")
    return frequencies, eigenvectors


def project_cartesian_to_modes(
    matrices: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project Cartesian displacement matrices onto phonon eigenmodes."""
    values = np.asarray(matrices, dtype=complex)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError("Cartesian matrices must have square trailing dimensions")
    ndof = values.shape[-1]
    pol = _normalize_eigenvectors(eigenvectors, ndof)
    return np.einsum("am,...ab,bn->...mn", pol.conj(), values, pol, optimize=True)


def evaluate_dense_qpoint(
    real_space: ElphRealSpace,
    phonons,
    qpoint: Sequence[float],
    phonon_evaluator: Optional[Callable] = None,
):
    """Interpolate Cartesian e-ph data and project it at one target q point."""
    q = np.asarray(qpoint, dtype=float)
    cart = evaluate_real_space(real_space, q[None, :], restore_gauge=True)[0]
    freq, pol = evaluate_phonons_at_q(phonons, q, phonon_evaluator=phonon_evaluator)
    pol = _normalize_eigenvectors(pol, cart.shape[-1])
    if len(freq) != pol.shape[1]:
        raise ValueError("number of phonon frequencies and eigenmodes differ")
    mode = project_cartesian_to_modes(cart, pol)
    return freq, pol, cart, mode


def iter_dense_elph_qpoints(
    real_space: ElphRealSpace,
    phonons,
    qpoints: np.ndarray,
    block_size: int = 64,
    phonon_evaluator: Optional[Callable] = None,
) -> Iterator[DenseElphBlock]:
    """Yield interpolated/projected electron-phonon data in q-point blocks.

    Each ``qpoint`` remains in fractional reciprocal coordinates throughout
    the e-ph interpolation.  The exact same fractional vector is passed to the
    phonon-evaluation adapter; the adapter performs any API-specific coordinate
    conversion internally.  This keeps the e-ph and phonon data locked to the
    same physical q-point.
    """
    qpoints = np.asarray(qpoints, dtype=float)
    if qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise ValueError("qpoints must have shape (nq, 3)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    for start in range(0, len(qpoints), block_size):
        stop = min(start + block_size, len(qpoints))
        qblock = qpoints[start:stop]
        cart_block = evaluate_real_space(real_space, qblock, restore_gauge=True)

        frequencies = []
        eigenvectors = []
        mode_matrices = []
        for local_i, qpoint in enumerate(qblock):
            freq, pol = evaluate_phonons_at_q(
                phonons, qpoint, phonon_evaluator=phonon_evaluator
            )
            pol = _normalize_eigenvectors(pol, cart_block.shape[-1])
            if len(freq) != pol.shape[1]:
                raise ValueError("number of phonon frequencies and eigenmodes differ")
            frequencies.append(freq)
            eigenvectors.append(pol)
            mode_matrices.append(project_cartesian_to_modes(cart_block[local_i], pol))

        try:
            frequency_array = np.stack(frequencies, axis=0)
            eigenvector_array = np.stack(eigenvectors, axis=0)
            mode_array = np.stack(mode_matrices, axis=0)
        except ValueError as exc:
            raise ValueError("phonon mode count changed within a q block") from exc

        yield DenseElphBlock(
            qpoints=qblock.copy(),
            frequencies=frequency_array,
            eigenvectors=eigenvector_array,
            cartesian_matrices=cart_block,
            mode_matrices=mode_array,
            start=start,
            stop=stop,
        )


def iter_dense_elph_mesh(
    real_space: ElphRealSpace,
    phonons,
    mesh: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
    block_size: int = 64,
    phonon_evaluator: Optional[Callable] = None,
) -> Iterator[DenseElphBlock]:
    """Generate a regular target mesh and yield dense e-ph data by blocks."""
    qpoints, _ = generate_regular_q_grid(mesh, shift)
    yield from iter_dense_elph_qpoints(
        real_space,
        phonons,
        qpoints,
        block_size=block_size,
        phonon_evaluator=phonon_evaluator,
    )
