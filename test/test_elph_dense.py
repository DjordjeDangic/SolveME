import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elph_dense import (
    evaluate_dense_qpoint,
    evaluate_phonons_at_q,
    iter_dense_elph_mesh,
    iter_dense_elph_qpoints,
    project_cartesian_to_modes,
)
from elph_grid import ElphGrid, generate_regular_q_grid
from elph_interpolation import elph_to_real_space


class MockPhonons:
    def DyagDinQ(self, qpoint):
        qpoint = np.asarray(qpoint)
        # Two non-degenerate modes with q-dependent frequencies; identity
        # eigenvectors make the expected projection transparent.
        frequencies = np.array([1.0 + qpoint[0], 2.0 + qpoint[1]])
        eigenvectors = np.eye(2, dtype=complex)
        return frequencies, eigenvectors


def _make_real_space():
    qpoints, _ = generate_regular_q_grid((2, 1, 1))
    matrices = np.empty((2, 2, 2), dtype=complex)
    matrices[0] = np.array([[2.0, 0.25], [0.25, 5.0]], dtype=complex)
    matrices[1] = np.array([[4.0, -0.25j], [0.25j, 7.0]], dtype=complex)
    grid = ElphGrid(qpoints=qpoints, matrices=matrices, mesh=(2, 1, 1))
    return elph_to_real_space(grid, atom_positions=None)


def test_project_cartesian_to_modes_identity():
    matrix = np.array([[2.0, 1.0j], [-1.0j, 5.0]], dtype=complex)
    np.testing.assert_allclose(
        project_cartesian_to_modes(matrix, np.eye(2, dtype=complex)), matrix
    )


def test_project_cartesian_to_modes_rotation():
    matrix = np.diag([2.0, 5.0]).astype(complex)
    invsqrt2 = 1.0 / np.sqrt(2.0)
    eigenvectors = np.array(
        [[invsqrt2, invsqrt2], [invsqrt2, -invsqrt2]], dtype=complex
    )
    expected = eigenvectors.conj().T @ matrix @ eigenvectors
    np.testing.assert_allclose(
        project_cartesian_to_modes(matrix, eigenvectors), expected
    )


def test_phonon_adapter_uses_cellconstructor_style_method():
    freq, pol = evaluate_phonons_at_q(MockPhonons(), [0.25, 0.5, 0.0])
    np.testing.assert_allclose(freq, [1.25, 2.5])
    np.testing.assert_allclose(pol, np.eye(2))


def test_custom_phonon_evaluator_is_supported():
    def evaluator(phonons, qpoint):
        del phonons
        return np.array([3.0 + qpoint[2], 4.0]), np.eye(2)

    freq, pol = evaluate_phonons_at_q(object(), [0.0, 0.0, 0.25], evaluator)
    np.testing.assert_allclose(freq, [3.25, 4.0])
    np.testing.assert_allclose(pol, np.eye(2))


def test_evaluate_dense_qpoint_interpolates_before_projection():
    real_space = _make_real_space()
    freq, pol, cart, mode = evaluate_dense_qpoint(
        real_space, MockPhonons(), [0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(freq, [1.0, 2.0])
    np.testing.assert_allclose(pol, np.eye(2))
    np.testing.assert_allclose(cart, [[2.0, 0.25], [0.25, 5.0]])
    np.testing.assert_allclose(mode, cart)


def test_streaming_preserves_q_order_and_block_boundaries():
    real_space = _make_real_space()
    qpoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.125, 0.0, 0.0],
        ]
    )
    blocks = list(
        iter_dense_elph_qpoints(real_space, MockPhonons(), qpoints, block_size=2)
    )
    assert [(block.start, block.stop) for block in blocks] == [(0, 2), (2, 4), (4, 5)]
    recovered_q = np.concatenate([block.qpoints for block in blocks], axis=0)
    np.testing.assert_allclose(recovered_q, qpoints)
    assert all(block.cartesian_matrices.shape[0] <= 2 for block in blocks)
    assert all(block.mode_matrices.shape[0] <= 2 for block in blocks)


def test_streamed_projection_matches_pointwise_evaluation():
    real_space = _make_real_space()
    qpoints = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )
    blocks = list(
        iter_dense_elph_qpoints(real_space, MockPhonons(), qpoints, block_size=2)
    )
    streamed = np.concatenate([block.mode_matrices for block in blocks], axis=0)
    expected = []
    for qpoint in qpoints:
        expected.append(evaluate_dense_qpoint(real_space, MockPhonons(), qpoint)[3])
    np.testing.assert_allclose(streamed, np.stack(expected, axis=0))


def test_regular_mesh_iterator_generates_requested_number_of_points():
    real_space = _make_real_space()
    blocks = list(
        iter_dense_elph_mesh(
            real_space,
            MockPhonons(),
            mesh=(4, 2, 1),
            block_size=3,
        )
    )
    assert sum(len(block.qpoints) for block in blocks) == 8
    assert max(len(block.qpoints) for block in blocks) <= 3
