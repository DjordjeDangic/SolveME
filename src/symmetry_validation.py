"""Validation of coarse-grid dynamical-matrix symmetry covariance.

The primary check compares dynamical matrices directly,

    D(q') = Gamma(S,q) D(q) Gamma(S,q)^dagger,

with complex conjugation when q' = -S q + G.  This comparison is independent
of the arbitrary choice of eigenvectors inside degenerate subspaces.

An optional eigenspace check is included as a diagnostic.  It groups nearly
degenerate eigenvalues and compares the corresponding projectors instead of
individual eigenvectors, which are not uniquely defined at degeneracies.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from elph_symmetry import (
    DEFAULT_Q_TOL,
    _default_gamma_builder,
    equivalent_qpoints,
    transform_elph_matrix,
)


@dataclass
class DynamicalSymmetryCheck:
    source_index: int
    target_index: int
    symmetry_index: int
    time_reversal: bool
    matrix_relative_error: float
    eigenvalue_relative_error: float
    subspace_error: float
    passed: bool


def _hermitian(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    return 0.5 * (matrix + matrix.conj().T)


def _relative_error(a, b, floor=1.0e-14):
    scale = max(float(np.linalg.norm(b)), floor)
    return float(np.linalg.norm(a - b)) / scale


def _degenerate_groups(eigenvalues, atol=1.0e-8, rtol=1.0e-6):
    """Return contiguous groups of numerically degenerate eigenvalues."""
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if len(values) == 0:
        return []

    groups = []
    start = 0
    for i in range(1, len(values)):
        scale = max(abs(values[i]), abs(values[i - 1]), 1.0)
        if abs(values[i] - values[i - 1]) > atol + rtol * scale:
            groups.append(slice(start, i))
            start = i
    groups.append(slice(start, len(values)))
    return groups


def _projector(vectors):
    vectors = np.asarray(vectors, dtype=complex)
    return vectors @ vectors.conj().T


def _subspace_error(reference_matrix, transformed_matrix, atol, rtol):
    """Compare eigenspaces using projectors, robust to degeneracies."""
    eval_ref, evec_ref = np.linalg.eigh(_hermitian(reference_matrix))
    eval_tr, evec_tr = np.linalg.eigh(_hermitian(transformed_matrix))

    # Eigenvalues are sorted by eigh.  If the spectra disagree substantially,
    # projector matching is not meaningful and the eigenvalue error will catch it.
    groups = _degenerate_groups(eval_ref, atol=atol, rtol=rtol)
    worst = 0.0
    for group in groups:
        pref = _projector(evec_ref[:, group])
        ptr = _projector(evec_tr[:, group])
        denom = max(float(np.linalg.norm(pref)), 1.0)
        worst = max(worst, float(np.linalg.norm(pref - ptr)) / denom)
    return worst


def validate_dynamical_matrix_symmetry(
    tc,
    *,
    matrix_rtol: float = 1.0e-7,
    eigen_rtol: float = 1.0e-7,
    degeneracy_atol: float = 1.0e-8,
    degeneracy_rtol: float = 1.0e-6,
    q_tol: float = DEFAULT_Q_TOL,
    include_time_reversal: bool = True,
    gamma_builder=None,
    raise_on_failure: bool = True,
) -> List[DynamicalSymmetryCheck]:
    """Validate coarse-grid dynamical matrices under all available symmetries.

    Parameters
    ----------
    tc
        A configured ``ThermalConductivity`` object after
        ``setup_harmonic_properties``.  It must expose ``qpoints`` (or
        ``k_points``), ``dynmats``, ``rotations``, ``reciprocal_lattice`` and
        the structure/symmetry metadata required by ``construct_symmetry_matrix``.
    matrix_rtol
        Relative tolerance for direct matrix covariance.  This is the primary
        correctness criterion and does not depend on eigenvector choices.
    eigen_rtol
        Relative tolerance for eigenvalue agreement.
    degeneracy_atol, degeneracy_rtol
        Tolerances used to group degenerate/near-degenerate eigenvalues before
        projector comparison.
    include_time_reversal
        Also check q' = -S q + G mappings using complex conjugation.

    Returns
    -------
    list[DynamicalSymmetryCheck]
        One result per valid source/symmetry/target mapping.
    """
    if gamma_builder is None:
        gamma_builder = _default_gamma_builder

    qpoints = np.asarray(getattr(tc, "qpoints", getattr(tc, "k_points", None)))
    if qpoints is None or qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise ValueError("ThermalConductivity object has no usable q-point grid")

    dynmats = np.asarray(tc.dynmats, dtype=complex)
    if dynmats.shape[0] != len(qpoints):
        raise ValueError("tc.dynmats and q-point grid have inconsistent sizes")

    rotations = np.asarray(tc.rotations)
    results: List[DynamicalSymmetryCheck] = []
    failures = []
    gamma_cache = {}

    for isource, qsource in enumerate(qpoints):
        for isym, rotation in enumerate(rotations):
            qrot = np.dot(rotation.T, qsource)
            qrot_cart = np.dot(qrot, tc.reciprocal_lattice)
            key = (isource, isym)
            if key not in gamma_cache:
                gamma_cache[key] = gamma_builder(tc, isym, qrot_cart)
            gamma = gamma_cache[key]

            candidates = [(qrot, False)]
            if include_time_reversal:
                candidates.append((-qrot, True))

            for qcand, time_reversal in candidates:
                target = None
                for itarget, qtarget in enumerate(qpoints):
                    if equivalent_qpoints(qcand, qtarget, q_tol):
                        target = itarget
                        break
                if target is None:
                    continue

                transformed = transform_elph_matrix(
                    dynmats[isource], gamma, time_reversal=time_reversal
                )
                reference = dynmats[target]

                matrix_error = _relative_error(transformed, reference)

                eval_ref = np.linalg.eigvalsh(_hermitian(reference))
                eval_tr = np.linalg.eigvalsh(_hermitian(transformed))
                eigen_error = _relative_error(eval_tr, eval_ref)
                subspace_error = _subspace_error(
                    reference,
                    transformed,
                    atol=degeneracy_atol,
                    rtol=degeneracy_rtol,
                )

                passed = matrix_error <= matrix_rtol and eigen_error <= eigen_rtol
                result = DynamicalSymmetryCheck(
                    source_index=isource,
                    target_index=target,
                    symmetry_index=isym,
                    time_reversal=time_reversal,
                    matrix_relative_error=matrix_error,
                    eigenvalue_relative_error=eigen_error,
                    subspace_error=subspace_error,
                    passed=passed,
                )
                results.append(result)
                if not passed:
                    failures.append(result)

    if not results:
        raise RuntimeError("no symmetry-related q-point pairs were found on the coarse grid")

    if failures and raise_on_failure:
        worst = max(failures, key=lambda item: item.matrix_relative_error)
        raise RuntimeError(
            "Dynamical-matrix symmetry validation failed for %d/%d mappings; "
            "worst matrix relative error %.3e (q %d -> %d, symmetry %d, time_reversal=%s)"
            % (
                len(failures),
                len(results),
                worst.matrix_relative_error,
                worst.source_index,
                worst.target_index,
                worst.symmetry_index,
                worst.time_reversal,
            )
        )

    return results


def summarize_dynamical_symmetry_checks(checks: Sequence[DynamicalSymmetryCheck]):
    """Return compact diagnostics suitable for printing/logging."""
    checks = list(checks)
    if not checks:
        return {
            "nchecks": 0,
            "nfailed": 0,
            "max_matrix_relative_error": 0.0,
            "max_eigenvalue_relative_error": 0.0,
            "max_subspace_error": 0.0,
        }
    return {
        "nchecks": len(checks),
        "nfailed": sum(not check.passed for check in checks),
        "max_matrix_relative_error": max(check.matrix_relative_error for check in checks),
        "max_eigenvalue_relative_error": max(check.eigenvalue_relative_error for check in checks),
        "max_subspace_error": max(check.subspace_error for check in checks),
    }
