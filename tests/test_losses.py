"""Tests for the matrix loss functions.

These require PyTorch (the ``ml`` extra) and are skipped automatically when it is
not installed, so they do not slow down the lightweight default test run.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from casnet.model.losses import mean_squared_error, symm_matrix_mse  # noqa: E402


def test_symm_matrix_mse_zero_on_symmetric_target():
    n = 4
    base = torch.arange(float(n * n)).reshape(n, n)
    sym = 0.5 * (base + base.T)
    pred = sym.reshape(1, -1)
    target = sym.reshape(1, -1)
    loss = symm_matrix_mse(pred, target, basis_set_size=n)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_symm_matrix_mse_symmetrizes_prediction():
    # A prediction whose symmetric part equals the (symmetric) target should
    # incur zero loss even though the raw prediction is not symmetric.
    n = 3
    sym_target = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    asym_part = torch.tensor([[0.0, 1.0, -2.0], [-1.0, 0.0, 4.0], [2.0, -4.0, 0.0]])
    pred = (sym_target + asym_part).reshape(1, -1)
    target = sym_target.reshape(1, -1)
    loss = symm_matrix_mse(pred, target, basis_set_size=n)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_mean_squared_error_matches_numpy():
    n = 3
    rng = np.random.default_rng(0)
    pred_np = rng.standard_normal((1, n * n)).astype("float32")
    target_np = rng.standard_normal((1, n * n)).astype("float32")
    loss = mean_squared_error(torch.tensor(pred_np), torch.tensor(target_np), basis_set_size=n)
    expected = np.sum((pred_np - target_np) ** 2) / (n * n)
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)
