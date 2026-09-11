import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elph_grid import QPointMapping
from elph_symmetry import validate_symmetry_collisions


class MockTC:
    def __init__(self):
        self.qpoints = np.array([[0.25, 0.0, 0.0]])
        self.rotations = np.array([np.eye(3)])
        self.reciprocal_lattice = np.eye(3)


def gamma_builder(tc, symmetry_index, qrot_cart):
    del tc, symmetry_index, qrot_cart
    return np.eye(2, dtype=complex)


def test_collision_validation_does_not_mix_independent_irreducible_sources():
    tc = MockTC()
    irred_qpoints = np.array(
        [
            [0.25, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ]
    )
    elph = np.array(
        [
            np.diag([1.0, 2.0]),
            np.diag([10.0, 20.0]),
        ],
        dtype=complex,
    )

    preferred = [
        QPointMapping(
            target_index=0,
            irred_index=0,
            symmetry_index=None,
            time_reversal=False,
            reciprocal_shift=np.zeros(3, dtype=int),
        )
    ]

    # The second input matrix is intentionally very different.  Validation
    # must only compare alternate symmetry routes from irred_index=0, because
    # that is the representative selected by the production mapping.
    validate_symmetry_collisions(
        tc,
        irred_qpoints,
        elph,
        preferred,
        gamma_builder=gamma_builder,
    )
