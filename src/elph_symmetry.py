"""Symmetry expansion of electron-phonon matrices from irreducible q-points.

The transformation convention follows the pre-existing SolveME implementation:
for a unitary space-group operation S, a Cartesian displacement-space matrix is
transformed as Gamma(S,q) D(q) Gamma(S,q)^dagger.  Mapping through -S q uses
the complex-conjugated transformed matrix (time reversal).

The module intentionally does not implement Fourier interpolation; it only
reconstructs a complete coarse q grid.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from elph_grid import ElphGrid, QPointMapping, canonicalize_qpoints


DEFAULT_Q_TOL = 1.0e-6


def _periodic_difference(q1, q2):
    diff = np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)
    return diff - np.rint(diff)


def equivalent_qpoints(q1, q2, tol=DEFAULT_Q_TOL):
    return np.linalg.norm(_periodic_difference(q1, q2)) < tol


def _find_qpoint(qpoint, qpoints, tol=DEFAULT_Q_TOL):
    for iq, candidate in enumerate(qpoints):
        if equivalent_qpoints(qpoint, candidate, tol):
            return iq
    return None


def build_qpoint_symmetry_map(tc, irred_qpoints, tol=DEFAULT_Q_TOL) -> List[QPointMapping]:
    """Map every q-point in ``tc.qpoints`` to an irreducible e-ph q-point.

    The method uses the same reciprocal-space rotations as the legacy
    ``mesolver.get_elph_on_full_grid`` implementation and records whether time
    reversal is needed.  A reciprocal-lattice shift is retained explicitly for
    future phase/gauge handling.
    """
    irred_qpoints = canonicalize_qpoints(irred_qpoints)
    full_qpoints = canonicalize_qpoints(tc.qpoints)
    mappings: List[Optional[QPointMapping]] = [None] * len(full_qpoints)

    rotations = np.asarray(tc.rotations)

    for ifull, qtarget in enumerate(full_qpoints):
        candidates = []
        for iirr, qsource in enumerate(irred_qpoints):
            diff = np.asarray(qsource) - np.asarray(qtarget)
            G = np.rint(diff).astype(int)
            if np.linalg.norm(diff - G) < tol:
                candidates.append(QPointMapping(ifull, iirr, None, False, G))

            diff = -np.asarray(qsource) - np.asarray(qtarget)
            G = np.rint(diff).astype(int)
            if np.linalg.norm(diff - G) < tol:
                candidates.append(QPointMapping(ifull, iirr, None, True, G))

            for isym, rotation in enumerate(rotations):
                qrot = np.dot(rotation.T, qsource)

                diff = qrot - qtarget
                G = np.rint(diff).astype(int)
                if np.linalg.norm(diff - G) < tol:
                    candidates.append(QPointMapping(ifull, iirr, isym, False, G))

                diff = -qrot - qtarget
                G = np.rint(diff).astype(int)
                if np.linalg.norm(diff - G) < tol:
                    candidates.append(QPointMapping(ifull, iirr, isym, True, G))

        if not candidates:
            raise RuntimeError(
                "Could not map full-grid q-point %s to any irreducible q-point"
                % np.asarray(qtarget)
            )

        candidates.sort(
            key=lambda m: (
                m.symmetry_index is not None,
                m.time_reversal,
                m.irred_index,
                -1 if m.symmetry_index is None else m.symmetry_index,
            )
        )
        mappings[ifull] = candidates[0]

    return mappings


def _default_gamma_builder(tc, symmetry_index, qrot_cart):
    import cellconstructor as CC

    rotations_cart, translations_cart = tc.get_sg_in_cartesian()
    return CC.ThermalConductivity.construct_symmetry_matrix(
        rotations_cart[symmetry_index],
        translations_cart[symmetry_index],
        qrot_cart,
        tc.dyn.structure.coords,
        tc.atom_map[symmetry_index],
        tc.unitcell,
    )


def construct_gamma_cache(
    tc,
    irred_qpoints,
    mappings: Sequence[QPointMapping],
    gamma_builder: Optional[Callable] = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Construct each needed displacement representation matrix only once."""
    if gamma_builder is None:
        gamma_builder = _default_gamma_builder

    cache: Dict[Tuple[int, int], np.ndarray] = {}
    irred_qpoints = np.asarray(irred_qpoints, dtype=float)
    for mapping in mappings:
        if mapping.symmetry_index is None:
            continue
        key = (mapping.irred_index, mapping.symmetry_index)
        if key in cache:
            continue
        rotation = np.asarray(tc.rotations[mapping.symmetry_index])
        qrot_frac = np.dot(rotation.T, irred_qpoints[mapping.irred_index])
        qrot_cart = np.dot(qrot_frac, tc.reciprocal_lattice)
        cache[key] = gamma_builder(tc, mapping.symmetry_index, qrot_cart)
    return cache


def transform_elph_matrix(matrix, gamma=None, time_reversal=False):
    """Apply displacement-space symmetry and optional time reversal."""
    transformed = np.asarray(matrix)
    if gamma is not None:
        transformed = np.einsum(
            "ij,...jk,lk->...il", gamma, transformed, gamma.conj(), optimize=True
        )
    if time_reversal:
        transformed = transformed.conj()
    return transformed


def expand_irreducible_elph(
    tc,
    irred_qpoints,
    elph,
    *,
    mappings: Optional[Sequence[QPointMapping]] = None,
    gamma_builder: Optional[Callable] = None,
    validate_collisions: bool = False,
    collision_tol: float = 1.0e-7,
) -> ElphGrid:
    """Reconstruct a full coarse q-grid from irreducible e-ph matrices.

    ``elph`` may contain arbitrary axes between q and the final two Cartesian
    displacement axes.  The first dimension must match ``irred_qpoints``.
    """
    irred_qpoints = np.asarray(irred_qpoints, dtype=float)
    elph = np.asarray(elph)
    if elph.shape[0] != len(irred_qpoints):
        raise ValueError("elph and irred_qpoints must have the same leading dimension")
    if elph.ndim < 3 or elph.shape[-1] != elph.shape[-2]:
        raise ValueError("e-ph matrices must have square trailing matrix axes")

    if mappings is None:
        mappings = build_qpoint_symmetry_map(tc, irred_qpoints)
    if len(mappings) != len(tc.qpoints):
        raise ValueError("mapping must contain one entry per full-grid q-point")

    gamma_cache = construct_gamma_cache(tc, irred_qpoints, mappings, gamma_builder)
    output = np.empty((len(tc.qpoints),) + elph.shape[1:], dtype=np.result_type(elph, complex))

    for mapping in mappings:
        gamma = None
        if mapping.symmetry_index is not None:
            gamma = gamma_cache[(mapping.irred_index, mapping.symmetry_index)]
        output[mapping.target_index] = transform_elph_matrix(
            elph[mapping.irred_index], gamma, mapping.time_reversal
        )

    if validate_collisions:
        validate_symmetry_collisions(
            tc,
            irred_qpoints,
            elph,
            mappings,
            gamma_builder=gamma_builder,
            tol=collision_tol,
        )

    return ElphGrid(qpoints=np.asarray(tc.qpoints), matrices=output)


def validate_symmetry_collisions(
    tc,
    irred_qpoints,
    elph,
    preferred_mappings,
    *,
    gamma_builder=None,
    tol=1.0e-7,
):
    """Check alternate symmetry paths from the selected irreducible source.

    For each target q-point, the production mapping first selects one
    irreducible representative.  Validation must therefore compare alternate
    symmetry operations that start from that *same* representative.  Mixing
    paths from different irreducible input points can spuriously compare data
    that are independent representatives in the input file and was the source
    of false collision failures in real calculations.

    This expensive diagnostic is intended for tests/debugging rather than the
    production hot path.
    """
    rotations = np.asarray(tc.rotations)
    irred_qpoints = np.asarray(irred_qpoints, dtype=float)
    full_qpoints = np.asarray(tc.qpoints, dtype=float)
    if gamma_builder is None:
        gamma_builder = _default_gamma_builder

    if len(preferred_mappings) != len(full_qpoints):
        raise ValueError("preferred_mappings must contain one entry per full-grid q-point")

    cache = {}

    def get_gamma(iirr, isym, qsource):
        if isym is None:
            return None
        key = (iirr, isym)
        if key not in cache:
            qrot = np.dot(rotations[isym].T, qsource)
            qrot_cart = np.dot(qrot, tc.reciprocal_lattice)
            cache[key] = gamma_builder(tc, isym, qrot_cart)
        return cache[key]

    for target_index, qtarget in enumerate(full_qpoints):
        preferred = preferred_mappings[target_index]
        iirr = preferred.irred_index
        qsource = irred_qpoints[iirr]

        preferred_gamma = get_gamma(iirr, preferred.symmetry_index, qsource)
        reference = transform_elph_matrix(
            elph[iirr], preferred_gamma, preferred.time_reversal
        )
        scale = max(float(np.linalg.norm(reference)), 1.0)

        paths = [(None, qsource, False), (None, -qsource, True)]
        for isym, rotation in enumerate(rotations):
            qrot = np.dot(rotation.T, qsource)
            paths.extend(((isym, qrot, False), (isym, -qrot, True)))

        for isym, qcand, tr in paths:
            if not equivalent_qpoints(qcand, qtarget):
                continue
            if isym == preferred.symmetry_index and tr == preferred.time_reversal:
                continue

            gamma = get_gamma(iirr, isym, qsource)
            value = transform_elph_matrix(elph[iirr], gamma, tr)
            error = float(np.linalg.norm(value - reference)) / scale
            if error > tol:
                raise RuntimeError(
                    "Inconsistent symmetry paths for q-point %d from irreducible "
                    "point %d: relative error %.3e (preferred symmetry=%s, "
                    "preferred time_reversal=%s; alternate symmetry=%s, "
                    "alternate time_reversal=%s)"
                    % (
                        target_index,
                        iirr,
                        error,
                        str(preferred.symmetry_index),
                        str(preferred.time_reversal),
                        str(isym),
                        str(tr),
                    )
                )


def get_elph_on_full_grid(tc, elph, tc_qpt_id, star_id):
    """Backward-compatible wrapper for the legacy SolveME API.

    ``tc_qpt_id`` identifies the full-grid representative corresponding to each
    irreducible e-ph matrix.  ``star_id`` is retained for API compatibility but
    is no longer required because the mapping is derived directly from q-points
    and symmetry operations.
    """
    del star_id
    irred_qpoints = np.asarray([tc.qpoints[i] for i in tc_qpt_id], dtype=float)
    return expand_irreducible_elph(tc, irred_qpoints, elph).matrices
