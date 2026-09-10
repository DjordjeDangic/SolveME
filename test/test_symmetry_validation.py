import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from symmetry_validation import (
    _degenerate_groups,
    summarize_dynamical_symmetry_checks,
    validate_dynamical_matrix_symmetry,
)


class MockTC:
    def __init__(self, bad_target=False):
        self.qpoints = np.array(
            [
                [0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.75, 0.0, 0.0],
                [0.0, 0.75, 0.0],
            ]
        )
        self.k_points = self.qpoints.copy()
        self.rotations = np.array(
            [
                np.eye(3),
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        self.reciprocal_lattice = np.eye(3)

        # Two exactly-degenerate modes followed by a non-degenerate one.
        source = np.diag([2.0, 2.0, 5.0]).astype(complex)
        gamma = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )
        rotated = gamma @ source @ gamma.conj().T
        if bad_target:
            rotated = rotated.copy()
            rotated[2, 2] += 0.2

        self.dynmats = np.array(
            [
                source,
                rotated,
                source.conj(),
                rotated.conj(),
            ]
        )


def gamma_builder(tc, symmetry_index, qrot_cart):
    del tc, qrot_cart
    if symmetry_index == 0:
        return np.eye(3, dtype=complex)
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )


def test_degenerate_groups_are_clustered():
    groups = _degenerate_groups(np.array([1.0, 1.0 + 1.0e-10, 3.0]))
    assert [(g.start, g.stop) for g in groups] == [(0, 2), (2, 3)]


def test_dynamical_symmetry_validation_passes_with_degeneracy():
    checks = validate_dynamical_matrix_symmetry(
        MockTC(),
        gamma_builder=gamma_builder,
        matrix_rtol=1.0e-12,
        eigen_rtol=1.0e-12,
    )
    assert checks
    assert all(check.passed for check in checks)

    summary = summarize_dynamical_symmetry_checks(checks)
    assert summary["nfailed"] == 0
    assert summary["max_matrix_relative_error"] < 1.0e-12
    assert summary["max_eigenvalue_relative_error"] < 1.0e-12
    assert summary["max_subspace_error"] < 1.0e-12


def test_dynamical_symmetry_validation_detects_bad_matrix():
    with pytest.raises(RuntimeError, match="Dynamical-matrix symmetry validation failed"):
        validate_dynamical_matrix_symmetry(
            MockTC(bad_target=True),
            gamma_builder=gamma_builder,
            matrix_rtol=1.0e-10,
        )


def test_nonraising_mode_reports_failures():
    checks = validate_dynamical_matrix_symmetry(
        MockTC(bad_target=True),
        gamma_builder=gamma_builder,
        matrix_rtol=1.0e-10,
        raise_on_failure=False,
    )
    summary = summarize_dynamical_symmetry_checks(checks)
    assert summary["nfailed"] > 0
