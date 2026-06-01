"""Tests for the SchNetPack model build, forward pass, and output modules.

Require the ``ml`` extra (torch + schnetpack) and are skipped automatically when
it is not installed, keeping the default test run lightweight.
"""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
spk = pytest.importorskip("schnetpack")

from ase import Atoms  # noqa: E402

from casnet.model.losses import symm_matrix_mse  # noqa: E402
from casnet.model.network import create_orbital_model  # noqa: E402

BASIS = 4


@pytest.fixture
def water_input():
    """A converted 3-atom input batch for the model."""
    mol = Atoms("OH2", positions=[[0, 0, 0], [0.758, 0.587, 0], [-0.758, 0.587, 0]])
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=spk.transform.ASENeighborList(cutoff=5.0), dtype=torch.float32
    )
    return converter(mol)


def _build(property_key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_orbital_model(
            loss_function=symm_matrix_mse,
            output_property_key=property_key,
            basis_set_size=BASIS,
            cutoff=5.0,
        )


def test_inference_module_imports_without_phisnet():
    # Regression: the phisnet_fork import is lazy, so importing inference (and the
    # F/MO-model functions) must not require the phisnet_fork submodule.
    import casnet.model.inference as inference

    assert hasattr(inference, "infer_orbitals_from_F_model")
    assert hasattr(inference, "infer_orbitals_from_mo_model")


def test_f_model_builds_and_predicts_symmetric_matrix(water_input):
    task = _build("F")
    task.eval()
    out = task.model(water_input)
    assert out["F"].shape == (BASIS * BASIS,)
    matrix = out["F"].detach().numpy().reshape(BASIS, BASIS)
    # HamiltonianOutput enforces symmetry (H + Hᵀ).
    assert np.allclose(matrix, matrix.T, atol=1e-5)


def test_mo_model_builds_and_predicts_right_shape(water_input):
    task = _build("mo_coeffs")
    task.eval()
    out = task.model(water_input)
    assert out["mo_coeffs"].shape == (BASIS * BASIS,)


def test_forward_pass_is_deterministic(water_input):
    torch.manual_seed(0)
    task = _build("F")
    task.eval()
    a = task.model(water_input)["F"].detach().clone()
    b = task.model(water_input)["F"].detach().clone()
    assert torch.allclose(a, b)
