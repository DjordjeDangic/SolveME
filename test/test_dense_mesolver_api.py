import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dense_mesolver
from mesolver import mesolver as BaseMesolver


def _set_minimal_dense_state(obj):
    obj.multiband = False
    obj.ep_nsmear = 1
    obj.elph_supercell = np.array([2, 2, 2])
    obj.elph_weights = np.array([1.0, 7.0])
    obj.elph_dos = np.array([1.0])
    obj.a2f = None
    obj.a2f_omega = None
    obj.delta = []
    obj.indices = []


def test_calculate_a2f_without_interpolation_delegates_to_legacy(monkeypatch):
    called = {}

    def fake_legacy(self, **kwargs):
        called.update(kwargs)
        self.a2f_omega = np.array([1.0, 2.0])
        self.a2f = np.array([[0.1, 0.2]])

    monkeypatch.setattr(BaseMesolver, "calculate_a2f", fake_legacy)
    obj = dense_mesolver.mesolver()
    _set_minimal_dense_state(obj)
    obj.calculate_a2f(nom=321)
    assert called["nom"] == 321
    assert np.allclose(obj.a2f[0], [0.1, 0.2])


def test_calculate_a2f_with_interpolation_dispatches_dense(monkeypatch):
    obj = dense_mesolver.mesolver()
    _set_minimal_dense_state(obj)
    called = {}

    def fake_dense(**kwargs):
        called.update(kwargs)
        obj.a2f_omega = np.array([1.0, 2.0])
        obj.a2f = np.array([[0.2, 0.3]])

    monkeypatch.setattr(obj, "_calculate_a2f_dense_isotropic", fake_dense)
    obj.calculate_a2f(interpolation_mesh=(4, 5, 6), interpolation_block_size=17)
    assert called["mesh"] == (4, 5, 6)
    assert called["block_size"] == 17


def test_dense_path_rejects_unsupported_physics():
    obj = dense_mesolver.mesolver()
    _set_minimal_dense_state(obj)

    try:
        obj.calculate_a2f(interpolation_mesh=(4, 4, 4), anharmonic=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("anharmonic dense interpolation must be rejected")

    obj.multiband = True
    try:
        obj.calculate_a2f(interpolation_mesh=(4, 4, 4))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("multiband dense interpolation must be rejected")


def test_validate_coarse_grid_equivalence_reports_lambda_error(monkeypatch):
    obj = dense_mesolver.mesolver()
    _set_minimal_dense_state(obj)

    def fake_legacy(self, **kwargs):
        self.a2f_omega = np.linspace(1.0, 4.0, 4)
        self.a2f = np.array([[1.0, 2.0, 3.0, 4.0]])

    monkeypatch.setattr(BaseMesolver, "calculate_a2f", fake_legacy)

    def fake_dense(**kwargs):
        obj.a2f_omega = np.linspace(1.0, 4.0, 4)
        obj.a2f = np.array([[1.0, 2.0, 3.0, 4.0]])

    monkeypatch.setattr(obj, "_calculate_a2f_dense_isotropic", fake_dense)
    report = obj.validate_coarse_grid_equivalence(nom=4, rtol=1.0e-12)
    assert report["a2f_close"]
    np.testing.assert_allclose(report["lambda_relative_error"], 0.0, atol=1.0e-12)


def test_converge_q_meshes_collects_metrics(monkeypatch):
    obj = dense_mesolver.mesolver()
    _set_minimal_dense_state(obj)

    def fake_calculate_a2f(*, interpolation_mesh=None, **kwargs):
        scale = float(np.prod(interpolation_mesh))
        obj.a2f_omega = np.array([1.0, 2.0, 3.0])
        obj.a2f = np.array([[scale, scale, scale]])

    monkeypatch.setattr(obj, "calculate_a2f", fake_calculate_a2f)
    monkeypatch.setattr(obj, "get_lambda", lambda smear_id=0: float(obj.a2f[0, 0]))
    monkeypatch.setattr(obj, "get_omega_log", lambda units="meV", smear_id=0: 11.0)
    monkeypatch.setattr(obj, "get_tot_a2f", lambda smear_id=0: obj.a2f[0])

    rows = obj.converge_q_meshes([(2, 2, 2), (3, 3, 3)])
    assert [row["nq"] for row in rows] == [8, 27]
    assert [row["lambda"] for row in rows] == [8.0, 27.0]
    assert all(row["omega_log_meV"] == 11.0 for row in rows)
