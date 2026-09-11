"""Symmetry expansion of electron-phonon matrices from irreducible q-points.

The transformation convention follows the pre-existing SolveME implementation:
for a unitary space-group operation S, a Cartesian displacement-space matrix is
transformed as Gamma(S,q) D(q) Gamma(S,q)^dagger. Mapping through -S q uses
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


def match_irreducible_qpoints_to_tc(tc, irred_qpoints, tol=DEFAULT_Q_TOL):
    """Match each input irreducible e-ph q point to one unique ``tc.qpoints`` index.

    Matching is purely geometric and never relies on array ordering. Two points
    are considered identical when they differ by an integer reciprocal-lattice
    vector within ``tol``.

    Returns
    -------
    numpy.ndarray
        Integer array ``matched[iirr] = itc``.
    """
    irred = canonicalize_qpoints(irred_qpoints)
    full = canonicalize_qpoints(tc.qpoints)
    matched = np.empty(len(irred), dtype=int)
    used = {}

    for iirr, q in enumerate(irred):
        candidates = [
            itc for itc, qtc in enumerate(full)
            if equivalent_qpoints(q, qtc, tol)
        ]
        if len(candidates) == 0:
            distances = np.array([
                np.linalg.norm(_periodic_difference(q, qtc)) for qtc in full
            ])
            nearest = int(np.argmin(distances))
            raise RuntimeError(
                "Electron-phonon irreducible q-point %d %s does not match any "
                "CellConstructor q-point within %.1e. Nearest is tc.qpoints[%d]="
                "%s with periodic distance %.3e"
                % (iirr, np.asarray(q), tol, nearest, full[nearest], distances[nearest])
            )
        if len(candidates) > 1:
            raise RuntimeError(
                "Electron-phonon irreducible q-point %d %s matches multiple "
                "CellConstructor q-points %s; the CC full grid is not unique modulo G"
                % (iirr, np.asarray(q), candidates)
            )

        itc = candidates[0]
        if itc in used:
            raise RuntimeError(
                "Electron-phonon irreducible q-points %d and %d both match "
                "tc.qpoints[%d]=%s. The input irreducible q list contains duplicate "
                "points modulo reciprocal lattice vectors."
                % (used[itc], iirr, itc, full[itc])
            )
        used[itc] = iirr
        matched[iirr] = itc

    return matched


def build_qpoint_symmetry_map(tc, irred_qpoints, tol=DEFAULT_Q_TOL) -> List[QPointMapping]:
    """Map every q-point in ``tc.qpoints`` to an irreducible e-ph q-point.

    First, every input irreducible e-ph q point is matched one-to-one to the
    CellConstructor full q grid by coordinate. The matched ``tc.qpoints`` values
    are then used as the symmetry sources. Thus the reconstruction never assumes
    the input q points and the CellConstructor q points have the same ordering.
    """
    irred_qpoints = canonicalize_qpoints(irred_qpoints)
    full_qpoints = canonicalize_qpoints(tc.qpoints)

    matched_tc_indices = match_irreducible_qpoints_to_tc(tc, irred_qpoints, tol)
    source_qpoints = full_qpoints[matched_tc_indices]

    mappings: List[Optional[QPointMapping]] = [None] * len(full_qpoints)
    rotations = np.asarray(tc.rotations)

    for ifull, qtarget in enumerate(full_qpoints):
        candidates = []
        for iirr, qsource in enumerate(source_qpoints):
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
                "Could not map CellConstructor full-grid q-point %d %s to any "
                "matched irreducible e-ph q-point"
                % (ifull, np.asarray(qtarget))
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
    matched_tc_indices = match_irreducible_qpoints_to_tc(tc, irred_qpoints)
    source_qpoints = canonicalize_qpoints(tc.qpoints)[matched_tc_indices]

    for mapping in mappings:
        if mapping.symmetry_index is None:
            continue
        key = (mapping.irred_index, mapping.symmetry_index)
        if key in cache:
            continue
        rotation = np.asarray(tc.rotations[mapping.symmetry_index])
        qrot_frac = np.dot(rotation.T, source_qpoints[mapping.irred_index])
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
    """Reconstruct a full coarse q-grid from irreducible e-ph matrices."""
    irred_qpoints = np.asarray(irred_qpoints, dtype=float)
    elph = np.asarray(elph)
    if elph.shape[0] != len(irred_qpoints):
        raise ValueError("elph and irred_qpoints must have the same leading dimension")
    if elph.ndim < 3 or elph.shape[-1] != elph.shape[-2]:
        raise ValueError("e-ph matrices must have square trailing matrix axes")

    # Always enforce the explicit one-to-one source-grid match, even when a
    # precomputed symmetry mapping is supplied.
    match_irreducible_qpoints_to_tc(tc, irred_qpoints)

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

    If a failing comparison involves time reversal, the diagnostic also reports
    the relative error obtained when the same spatial symmetry transformation is
    applied without complex conjugation.  This does not change the production
    transformation; it is only intended to identify the time-reversal convention
    of the stored e-ph matrix.
    """
    rotations = np.asarray(tc.rotations)
    irred_qpoints = np.asarray(irred_qpoints, dtype=float)
    full_qpoints = canonicalize_qpoints(tc.qpoints)
    matched_tc_indices = match_irreducible_qpoints_to_tc(tc, irred_qpoints)
    source_qpoints = full_qpoints[matched_tc_indices]

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
        qsource = source_qpoints[iirr]

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
                diagnostic = ""

                if tr:
                    value_no_tr = transform_elph_matrix(
                        elph[iirr], gamma, time_reversal=False
                    )
                    error_no_tr = float(np.linalg.norm(value_no_tr - reference)) / scale
                    diagnostic += (
                        "; alternate-route error without TR conjugation=%.3e"
                        % error_no_tr
                    )

                if preferred.time_reversal:
                    reference_no_tr = transform_elph_matrix(
                        elph[iirr], preferred_gamma, time_reversal=False
                    )
                    alt_scale = max(float(np.linalg.norm(reference_no_tr)), 1.0)
                    error_preferred_no_tr = (
                        float(np.linalg.norm(value - reference_no_tr)) / alt_scale
                    )
                    diagnostic += (
                        "; error with preferred-route TR conjugation removed=%.3e"
                        % error_preferred_no_tr
                    )

                raise RuntimeError(
                    "Inconsistent symmetry paths for tc.qpoints[%d]=%s from matched "
                    "e-ph irreducible point %d=%s (matched tc index %d): relative "
                    "error %.3e (preferred symmetry=%s, preferred TR=%s; alternate "
                    "symmetry=%s, alternate TR=%s)%s"
                    % (
                        target_index,
                        qtarget,
                        iirr,
                        irred_qpoints[iirr],
                        matched_tc_indices[iirr],
                        error,
                        str(preferred.symmetry_index),
                        str(preferred.time_reversal),
                        str(isym),
                        str(tr),
                        diagnostic,
                    )
                )


def get_elph_on_full_grid(tc, elph, tc_qpt_id, star_id):
    """Backward-compatible wrapper for the legacy SolveME API."""
    del star_id
    irred_qpoints = np.asarray([tc.qpoints[i] for i in tc_qpt_id], dtype=float)
    return expand_irreducible_elph(tc, irred_qpoints, elph).matrices
