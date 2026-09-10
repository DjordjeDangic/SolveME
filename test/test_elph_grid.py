import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elph_grid import ElphGrid, generate_regular_q_grid, mesh_index_lookup, qpoint_to_mesh_index


def test_generate_gamma_centered_grid_has_integer_indices():
    qpoints, indices = generate_regular_q_grid((2, 3, 2))
    assert qpoints.shape == (12, 3)
    assert indices.shape == (12, 3)
    lookup = mesh_index_lookup(qpoints, (2, 3, 2))
    assert len(lookup) == 12
    for iq, qpoint in enumerate(qpoints):
        assert lookup[qpoint_to_mesh_index(qpoint, (2, 3, 2))] == iq


def test_generate_shifted_grid_round_trip():
    mesh = (4, 2, 2)
    shift = (0.5, 0.5, 0.5)
    qpoints, indices = generate_regular_q_grid(mesh, shift)
    for qpoint, index in zip(qpoints, indices):
        assert qpoint_to_mesh_index(qpoint, mesh, shift) == tuple(index)


def test_off_grid_point_is_rejected():
    with pytest.raises(ValueError, match="commensurate"):
        qpoint_to_mesh_index((0.11, 0.0, 0.0), (4, 4, 4))


def test_elph_grid_validates_leading_dimension_and_regular_mesh():
    qpoints, _ = generate_regular_q_grid((2, 1, 1))
    matrices = np.zeros((2, 3, 3), dtype=complex)
    grid = ElphGrid(qpoints, matrices, mesh=(2, 1, 1))
    assert grid.nq == 2

    with pytest.raises(ValueError, match="leading dimension"):
        ElphGrid(qpoints, np.zeros((3, 3, 3)))

    with pytest.raises(ValueError, match="prod"):
        ElphGrid(qpoints, matrices, mesh=(3, 1, 1))
