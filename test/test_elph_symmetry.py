import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elph_symmetry import (
    build_qpoint_symmetry_map,
    expand_irreducible_elph,
    transform_elph_matrix,
    validate_symmetry_collisions,
)


class MockTC:
    def __init__(self):
        self.qpoints = np.array(
            [
                [0.25, 0.00, 0.0],
                [0.00, 0.25, 0.0],
                [0.75, 0.00, 0.0],
                [0.00, 0.75, 0.0],
            ]
        )
        # Identity and a 90-degree rotation about z.  The legacy SolveME
        # convention applies rotation.T to fractional q coordinates.
        self.rotations = np.array(
            [
                np.eye(3),
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        self.reciprocal_lattice = np.eye(3)


def gamma_builder(tc, symmetry_index, qrot_cart):
    del tc, qrot_cart
    if symmetry_index == 0:
        return np.eye(2)
    # Mock displacement representation: exchange the two coordinates.
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def test_transform_identity_and_time_reversal():
    matrix = np.array([[1.0, 2.0j], [-3.0j, 4.0]], dtype=complex)
    np.testing.assert_allclose(transform_elph_matrix(matrix), matrix)
    np.testing.assert_allclose(
        transform_elph_matrix(matrix, time_reversal=True), matrix.conj()
    )


def test_transform_applies_gamma_d_gamma_dagger():
    matrix = np.diag([2.0, 5.0]).astype(complex)
    gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    transformed = transform_elph_matrix(matrix, gamma)
    np.testing.assert_allclose(transformed, np.diag([5.0, 2.0]))


def test_mapping_is_deterministic_and_covers_full_grid():
    tc = MockTC()
    mappings = build_qpoint_symmetry_map(tc, np.array([[0.25, 0.0, 0.0]]))
    assert len(mappings) == len(tc.qpoints)
    assert mappings[0].irred_index == 0
    assert mappings[0].symmetry_index is None
    assert mappings[0].time_reversal is False
    assert mappings[2].time_reversal is True


def test_expand_irreducible_elph_reconstructs_star_and_time_reverse():
    tc = MockTC()
    source = np.array([[[2.0 + 0.0j, 1.0j], [-1.0j, 5.0 + 0.0j]]])
    grid = expand_irreducible_elph(
        tc,
        np.array([[0.25, 0.0, 0.0]]),
        source,
        gamma_builder=gamma_builder,
    )

    expected_rotated = np.array([[5.0, -1.0j], [1.0j, 2.0]], dtype=complex)
    np.testing.assert_allclose(grid.matrices[0], source[0])
    np.testing.assert_allclose(grid.matrices[1], expected_rotated)
    np.testing.assert_allclose(grid.matrices[2], source[0].conj())
    np.testing.assert_allclose(grid.matrices[3], expected_rotated.conj())


def test_extra_axes_between_q_and_matrix_are_supported():
    tc = MockTC()
    source = np.zeros((1, 3, 2, 2), dtype=complex)
    source[0, 0] = np.diag([1.0, 2.0])
    source[0, 1] = np.diag([3.0, 4.0])
    source[0, 2] = np.diag([5.0, 6.0])

    grid = expand_irreducible_elph(
        tc,
        np.array([[0.25, 0.0, 0.0]]),
        source,
        gamma_builder=gamma_builder,
    )
    assert grid.matrices.shape == (4, 3, 2, 2)
    np.testing.assert_allclose(grid.matrices[1, 1], np.diag([4.0, 3.0]))


def test_collision_validator_accepts_consistent_duplicate_paths():
    tc = MockTC()
    source = np.array([np.eye(2, dtype=complex)])
    mappings = build_qpoint_symmetry_map(tc, np.array([[0.25, 0.0, 0.0]]))
    validate_symmetry_collisions(
        tc,
        np.array([[0.25, 0.0, 0.0]]),
        source,
        mappings,
        gamma_builder=gamma_builder,
    )


def test_collision_validator_rejects_inconsistent_duplicate_paths():
    tc = MockTC()
    # Add a second identity reciprocal-space operation.  It reaches the same
    # q-point by a different symmetry path, so the displacement representation
    # must produce the same transformed matrix.
    tc.rotations = np.concatenate([tc.rotations, np.eye(3)[None, :, :]], axis=0)
    source = np.array([np.diag([1.0, 3.0]).astype(complex)])
    mappings = build_qpoint_symmetry_map(tc, np.array([[0.25, 0.0, 0.0]]))

    def inconsistent_gamma_builder(tc, symmetry_index, qrot_cart):
        del tc, qrot_cart
        if symmetry_index in (0,):
            return np.eye(2)
        if symmetry_index == 2:
            return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    with pytest.raises(RuntimeError, match="Inconsistent symmetry paths"):
        validate_symmetry_collisions(
            tc,
            np.array([[0.25, 0.0, 0.0]]),
            source,
            mappings,
            gamma_builder=inconsistent_gamma_builder,
        )
