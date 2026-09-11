"""Symmetry expansion of electron-phonon matrices from irreducible q-points.

The transformation convention follows the pre-existing SolveME implementation:
for a unitary space-group operation S, a Cartesian displacement-space matrix is
transformed as Gamma(S,q) D(q) Gamma(S,q)^dagger. Mapping through -S q uses
the complex-conjugated transformed matrix (time reversal).
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import warnings

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
    """Match each input irreducible e-ph q point to one unique tc.qpoints index."""
    irred = canonicalize_qpoints(irred_qpoints)
    full = canonicalize_qpoints(tc.qpoints)
    matched = np.empty(len(irred), dtype=int)
    used = {}

    for iirr, q in enumerate(irred):
        candidates = [itc for itc, qtc in enumerate(full) if equivalent_qpoints(q, qtc, tol)]
        if not candidates:
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


def build_qpoint_symmetry_candidates(tc, irred_qpoints, tol=DEFAULT_Q_TOL):
    """Return all symmetry/TR routes from irreducible sources to every full-grid q."""
    irred_qpoints = canonicalize_qpoints(irred_qpoints)
    full_qpoints = canonicalize_qpoints(tc.qpoints)
    matched_tc_indices = match_irreducible_qpoints_to_tc(tc, irred_qpoints, tol)
    source_qpoints = full_qpoints[matched_tc_indices]
    rotations = np.asarray(tc.rotations)
    all_candidates = []

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
                "matched irreducible e-ph q-point" % (ifull, np.asarray(qtarget))
            )
        all_candidates.append(candidates)

    return all_candidates


def _mapping_tiebreak_key(mapping):
    return (
        mapping.irred_index,
        mapping.time_reversal,
        mapping.symmetry_index is not None,
        -1 if mapping.symmetry_index is None else mapping.symmetry_index,
        tuple(np.asarray(mapping.reciprocal_shift, dtype=int)),
    )


def build_qpoint_symmetry_map(tc, irred_qpoints, tol=DEFAULT_Q_TOL) -> List[QPointMapping]:
    """Build a deterministic geometry-only map.

    Production expansion uses select_continuous_qpoint_symmetry_map instead.
    """
    candidate_sets = build_qpoint_symmetry_candidates(tc, irred_qpoints, tol)
    mappings = []
    for candidates in candidate_sets:
        candidates = sorted(candidates, key=_mapping_tiebreak_key)
        anchors = [m for m in candidates if m.symmetry_index is None and not m.time_reversal]
        mappings.append(anchors[0] if anchors else candidates[0])
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


def _fractional_atom_positions(tc):
    coords = np.asarray(tc.dyn.structure.coords, dtype=float)
    cell = np.asarray(tc.unitcell, dtype=float)
    return np.dot(coords, np.linalg.inv(cell))


def apply_reciprocal_gauge(matrix, tc, reciprocal_shift, sign=-1.0):
    """Move a displacement-space matrix between q and q+G Bloch gauges."""
    G = np.asarray(reciprocal_shift, dtype=float)
    if G.shape != (3,):
        raise ValueError("reciprocal_shift must have shape (3,)")
    if np.all(G == 0):
        return np.asarray(matrix)

    tau = _fractional_atom_positions(tc)
    atom_phase = np.exp(sign * 2.0j * np.pi * np.dot(tau, G))
    phase = np.repeat(atom_phase, 3)
    value = np.asarray(matrix)
    return phase[:, None] * value * phase.conj()[None, :]


def _route_value(matrix, mapping, gamma_cache, tc, reciprocal_gauge=False):
    gamma = None
    if mapping.symmetry_index is not None:
        gamma = gamma_cache[(mapping.irred_index, mapping.symmetry_index)]
    value = transform_elph_matrix(matrix, gamma, mapping.time_reversal)
    if reciprocal_gauge:
        value = apply_reciprocal_gauge(value, tc, mapping.reciprocal_shift)
    return value


def _relative_norm_difference(a, b, floor=1.0e-14):
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), floor)
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) / scale


def _regular_mesh_neighbors(tc, tol=1.0e-6):
    """Return periodic nearest-neighbor q-point indices."""
    qpoints = canonicalize_qpoints(tc.qpoints)
    if not hasattr(tc, "kpoint_grid"):
        neighbors = [[] for _ in qpoints]
        for iq, q in enumerate(qpoints):
            distances = np.array([
                np.inf if jq == iq else np.linalg.norm(_periodic_difference(q, qp))
                for jq, qp in enumerate(qpoints)
            ])
            dmin = float(np.min(distances))
            if not np.isfinite(dmin):
                continue
            neighbors[iq] = [
                jq for jq, distance in enumerate(distances) if abs(distance - dmin) <= tol
            ]
        return neighbors

    mesh = tuple(int(x) for x in tc.kpoint_grid)
    lookup = {}
    for iq, q in enumerate(qpoints):
        scaled = q * np.asarray(mesh, dtype=float)
        rounded = np.rint(scaled)
        if np.linalg.norm(scaled - rounded) > tol:
            raise RuntimeError("tc.qpoints is not commensurate with tc.kpoint_grid")
        index = tuple(np.mod(rounded.astype(int), mesh))
        if index in lookup:
            raise RuntimeError("tc.qpoints contains duplicate regular-mesh indices")
        lookup[index] = iq

    neighbors = [[] for _ in qpoints]
    for index, iq in lookup.items():
        for axis in range(3):
            for step in (-1, 1):
                candidate = list(index)
                candidate[axis] = (candidate[axis] + step) % mesh[axis]
                jq = lookup[tuple(candidate)]
                if jq != iq and jq not in neighbors[iq]:
                    neighbors[iq].append(jq)
    return neighbors


def select_continuous_qpoint_symmetry_map(
    tc,
    irred_qpoints,
    elph,
    *,
    gamma_builder: Optional[Callable] = None,
    reciprocal_gauge: bool = False,
    tol: float = DEFAULT_Q_TOL,
):
    """Choose one route per q by continuity with already-selected nearest neighbors.

    Exact irreducible source q points are fixed as identity-route anchors. The
    periodic mesh is then grown outward. At each step every valid symmetry/TR
    route is reconstructed and scored by the mean relative Frobenius distance
    to already-selected nearest neighbors. Symmetry indices are used only as a
    deterministic final tie-break.
    """
    irred_qpoints = np.asarray(irred_qpoints, dtype=float)
    elph = np.asarray(elph)
    candidate_sets = build_qpoint_symmetry_candidates(tc, irred_qpoints, tol)
    flat_candidates = [m for candidates in candidate_sets for m in candidates]
    gamma_cache = construct_gamma_cache(tc, irred_qpoints, flat_candidates, gamma_builder)
    neighbors = _regular_mesh_neighbors(tc, tol)

    selected = [None] * len(candidate_sets)
    values = [None] * len(candidate_sets)
    frontier = set()

    for target_index, candidates in enumerate(candidate_sets):
        anchors = [m for m in candidates if m.symmetry_index is None and not m.time_reversal]
        if not anchors:
            continue
        anchor = sorted(anchors, key=_mapping_tiebreak_key)[0]
        selected[target_index] = anchor
        values[target_index] = _route_value(
            elph[anchor.irred_index], anchor, gamma_cache, tc, reciprocal_gauge
        )
        frontier.update(j for j in neighbors[target_index] if selected[j] is None)

    if not any(mapping is not None for mapping in selected):
        first = sorted(candidate_sets[0], key=_mapping_tiebreak_key)[0]
        selected[0] = first
        values[0] = _route_value(
            elph[first.irred_index], first, gamma_cache, tc, reciprocal_gauge
        )
        frontier.update(neighbors[0])

    while any(mapping is None for mapping in selected):
        available = sorted(i for i in frontier if selected[i] is None)
        if not available:
            available = [i for i, mapping in enumerate(selected) if mapping is None]

        best_target = None
        best_mapping = None
        best_value = None
        best_key = None

        for target_index in available:
            selected_neighbors = [j for j in neighbors[target_index] if selected[j] is not None]
            if not selected_neighbors and frontier:
                continue

            for mapping in candidate_sets[target_index]:
                value = _route_value(
                    elph[mapping.irred_index], mapping, gamma_cache, tc, reciprocal_gauge
                )
                score = 0.0
                if selected_neighbors:
                    score = float(np.mean([
                        _relative_norm_difference(value, values[j]) for j in selected_neighbors
                    ]))
                key = (score, _mapping_tiebreak_key(mapping), target_index)
                if best_key is None or key < best_key:
                    best_key = key
                    best_target = target_index
                    best_mapping = mapping
                    best_value = value

        if best_target is None:
            best_target = available[0]
            best_mapping = sorted(candidate_sets[best_target], key=_mapping_tiebreak_key)[0]
            best_value = _route_value(
                elph[best_mapping.irred_index], best_mapping, gamma_cache, tc, reciprocal_gauge
            )

        selected[best_target] = best_mapping
        values[best_target] = best_value
        frontier.discard(best_target)
        frontier.update(j for j in neighbors[best_target] if selected[j] is None)

    return selected


def expand_irreducible_elph(
    tc,
    irred_qpoints,
    elph,
    *,
    mappings: Optional[Sequence[QPointMapping]] = None,
    gamma_builder: Optional[Callable] = None,
    validate_collisions: bool = False,
    collision_tol: float = 1.0e-7,
    reciprocal_gauge: bool = False,
    continuity_gauge: bool = True,
) -> ElphGrid:
    """Reconstruct a full coarse q grid from irreducible e-ph matrices.

    Equivalent routes are selected by neighbor continuity by default. Pass
    continuity_gauge=False to use the geometry-only deterministic mapping.
    """
    irred_qpoints = np.asarray(irred_qpoints, dtype=float)
    elph = np.asarray(elph)
    if elph.shape[0] != len(irred_qpoints):
        raise ValueError("elph and irred_qpoints must have the same leading dimension")
    if elph.ndim < 3 or elph.shape[-1] != elph.shape[-2]:
        raise ValueError("e-ph matrices must have square trailing matrix axes")

    match_irreducible_qpoints_to_tc(tc, irred_qpoints)

    if mappings is None:
        if continuity_gauge:
            mappings = select_continuous_qpoint_symmetry_map(
                tc,
                irred_qpoints,
                elph,
                gamma_builder=gamma_builder,
                reciprocal_gauge=reciprocal_gauge,
            )
        else:
            mappings = build_qpoint_symmetry_map(tc, irred_qpoints)
    if len(mappings) != len(tc.qpoints):
        raise ValueError("mapping must contain one entry per full-grid q-point")

    gamma_cache = construct_gamma_cache(tc, irred_qpoints, mappings, gamma_builder)
    output = np.empty((len(tc.qpoints),) + elph.shape[1:], dtype=np.result_type(elph, complex))
    for mapping in mappings:
        output[mapping.target_index] = _route_value(
            elph[mapping.irred_index], mapping, gamma_cache, tc, reciprocal_gauge
        )

    if validate_collisions:
        validate_symmetry_collisions(
            tc,
            irred_qpoints,
            elph,
            mappings,
            gamma_builder=gamma_builder,
            tol=collision_tol,
            reciprocal_gauge=reciprocal_gauge,
        )

    return ElphGrid(qpoints=np.asarray(tc.qpoints), matrices=output)


def _collision_invariant_diagnostics(tc, target_index, reference, value):
    """Compare basis-insensitive and phonon-projected diagnostics for two routes."""
    reference = np.asarray(reference, dtype=complex)
    value = np.asarray(value, dtype=complex)
    ref_h = 0.5 * (reference + np.swapaxes(reference.conj(), -1, -2))
    val_h = 0.5 * (value + np.swapaxes(value.conj(), -1, -2))
    ref_herm = _relative_norm_difference(reference, ref_h)
    val_herm = _relative_norm_difference(value, val_h)
    trace_error = _relative_norm_difference(
        np.trace(reference, axis1=-2, axis2=-1),
        np.trace(value, axis1=-2, axis2=-1),
    )
    spectrum_error = _relative_norm_difference(np.linalg.eigvalsh(ref_h), np.linalg.eigvalsh(val_h))

    mode_diag_error = np.nan
    mode_abs_error = np.nan
    eigvecs = getattr(tc, "eigvecs", None)
    if eigvecs is not None:
        eig = np.asarray(eigvecs[target_index], dtype=complex)
        ncart = reference.shape[-1]
        if eig.shape != (ncart, ncart):
            eig = eig.T
        if eig.shape == (ncart, ncart):
            ref_mode = np.einsum("ia,...ij,jb->...ab", eig.conj(), ref_h, eig, optimize=True)
            val_mode = np.einsum("ia,...ij,jb->...ab", eig.conj(), val_h, eig, optimize=True)
            ref_diag = np.diagonal(ref_mode, axis1=-2, axis2=-1).real
            val_diag = np.diagonal(val_mode, axis1=-2, axis2=-1).real
            mode_diag_error = _relative_norm_difference(ref_diag, val_diag)
            mode_abs_error = _relative_norm_difference(np.abs(ref_mode), np.abs(val_mode))

    return (
        "; invariants: hermiticity(preferred)=%.3e, hermiticity(alternate)=%.3e, "
        "trace error=%.3e, eigenvalue-spectrum error=%.3e, "
        "target-mode diagonal error=%.3e, target-mode |M| error=%.3e"
        % (ref_herm, val_herm, trace_error, spectrum_error, mode_diag_error, mode_abs_error)
    )


def validate_symmetry_collisions(
    tc,
    irred_qpoints,
    elph,
    preferred_mappings,
    *,
    gamma_builder=None,
    tol=1.0e-7,
    reciprocal_gauge=False,
):
    """Check alternate symmetry paths and warn, rather than abort, on mismatch."""
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

    def route_value(iirr, gamma, tr, G, gauge_sign=-1.0):
        value = transform_elph_matrix(elph[iirr], gamma, tr)
        if reciprocal_gauge:
            value = apply_reciprocal_gauge(value, tc, G, sign=gauge_sign)
        return value

    failures = []
    worst_error = -1.0
    worst_message = None

    for target_index, qtarget in enumerate(full_qpoints):
        preferred = preferred_mappings[target_index]
        iirr = preferred.irred_index
        qsource = source_qpoints[iirr]
        preferred_gamma = get_gamma(iirr, preferred.symmetry_index, qsource)
        reference = route_value(
            iirr, preferred_gamma, preferred.time_reversal, preferred.reciprocal_shift
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

            G = np.rint(np.asarray(qcand) - np.asarray(qtarget)).astype(int)
            gamma = get_gamma(iirr, isym, qsource)
            value = route_value(iirr, gamma, tr, G)
            error = float(np.linalg.norm(value - reference)) / scale
            if error <= tol:
                continue

            diagnostic = _collision_invariant_diagnostics(tc, target_index, reference, value)
            diagnostic += (
                "; reciprocal shifts preferred=%s alternate=%s"
                % (np.asarray(preferred.reciprocal_shift), G)
            )

            if reciprocal_gauge:
                reference_opposite = route_value(
                    iirr, preferred_gamma, preferred.time_reversal,
                    preferred.reciprocal_shift, gauge_sign=+1.0,
                )
                value_opposite = route_value(iirr, gamma, tr, G, gauge_sign=+1.0)
                opposite_scale = max(float(np.linalg.norm(reference_opposite)), 1.0)
                opposite_error = float(np.linalg.norm(value_opposite - reference_opposite)) / opposite_scale
                diagnostic += "; opposite reciprocal-gauge sign error=%.3e" % opposite_error

            if tr:
                value_no_tr = route_value(iirr, gamma, False, G)
                error_no_tr = float(np.linalg.norm(value_no_tr - reference)) / scale
                diagnostic += "; alternate-route error without TR conjugation=%.3e" % error_no_tr
                diagnostic += _collision_invariant_diagnostics(
                    tc, target_index, reference, value_no_tr
                ).replace("; invariants:", "; no-TR invariants:")

            if preferred.time_reversal:
                reference_no_tr = route_value(
                    iirr, preferred_gamma, False, preferred.reciprocal_shift
                )
                alt_scale = max(float(np.linalg.norm(reference_no_tr)), 1.0)
                error_preferred_no_tr = float(np.linalg.norm(value - reference_no_tr)) / alt_scale
                diagnostic += (
                    "; error with preferred-route TR conjugation removed=%.3e"
                    % error_preferred_no_tr
                )

            message = (
                "Inconsistent symmetry paths for tc.qpoints[%d]=%s from matched "
                "e-ph irreducible point %d=%s (matched tc index %d): relative "
                "error %.3e (preferred symmetry=%s, preferred TR=%s; alternate "
                "symmetry=%s, alternate TR=%s)%s"
                % (
                    target_index, qtarget, iirr, irred_qpoints[iirr],
                    matched_tc_indices[iirr], error,
                    str(preferred.symmetry_index), str(preferred.time_reversal),
                    str(isym), str(tr), diagnostic,
                )
            )
            failures.append(message)
            if error > worst_error:
                worst_error = error
                worst_message = message

    if failures:
        warnings.warn(
            "Electron-phonon symmetry validation found %d raw Cartesian route "
            "mismatches above %.1e. Reconstruction continues using the selected "
            "continuity gauge. Worst mismatch:\n%s"
            % (len(failures), tol, worst_message),
            RuntimeWarning,
            stacklevel=2,
        )
    return failures


def get_elph_on_full_grid(tc, elph, tc_qpt_id, star_id):
    """Backward-compatible wrapper for the legacy SolveME API."""
    del star_id
    irred_qpoints = np.asarray([tc.qpoints[i] for i in tc_qpt_id], dtype=float)
    return expand_irreducible_elph(tc, irred_qpoints, elph).matrices
