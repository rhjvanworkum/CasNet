"""Tests for the CasscfResult container (NumPy npz serialisation)."""

import numpy as np

from casnet.data.utils import CasscfResult, find_all_files_in_output_folder


def _make_result(index=None):
    rng = np.random.default_rng(0)
    return CasscfResult(
        converged=True,
        basis="sto_6g",
        e_tot=-1.23,
        mo_energies=rng.standard_normal(4),
        mo_coeffs=rng.standard_normal((4, 4)),
        S=rng.standard_normal((4, 4)),
        F=rng.standard_normal((4, 4)),
        imacro=7,
        index=index,
    )


def test_store_and_load_roundtrip(tmp_path):
    result = _make_result()
    path = tmp_path / "calc_5.npz"
    result.store_as_npz(str(path))

    loaded = CasscfResult.load_from_npz(str(path))
    assert bool(loaded.converged) is True
    assert str(loaded.basis) == "sto_6g"
    assert np.isclose(float(loaded.e_tot), -1.23)
    assert int(loaded.imacro) == 7
    assert np.allclose(loaded.mo_coeffs, result.mo_coeffs)
    # index parsed from the filename
    assert loaded.index == 5


def test_mo_coeffs_adjusted_defaults_none():
    assert _make_result().mo_coeffs_adjusted is None


def test_find_all_files_in_output_folder(tmp_path):
    for i in range(3):
        _make_result(index=i).store_as_npz(str(tmp_path / f"calc_{i}.npz"))
    (tmp_path / "ignore.txt").write_text("nope")

    results = find_all_files_in_output_folder(str(tmp_path) + "/")
    assert len(results) == 3
    assert sorted(r.index for r in results) == [0, 1, 2]
