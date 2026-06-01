"""Tests for configuration constants and environment-variable path helpers."""

import casnet.config as config


def test_base_dir_prefers_casnet_env(monkeypatch):
    monkeypatch.setenv("CASNET_BASE_DIR", "/tmp/casnet_exp")
    monkeypatch.delenv("base_dir", raising=False)
    assert config.get_base_dir() == "/tmp/casnet_exp"


def test_base_dir_legacy_env(monkeypatch):
    monkeypatch.delenv("CASNET_BASE_DIR", raising=False)
    monkeypatch.setenv("base_dir", "/legacy/path")
    assert config.get_base_dir() == "/legacy/path"


def test_base_dir_falls_back_to_cwd(monkeypatch):
    monkeypatch.delenv("CASNET_BASE_DIR", raising=False)
    monkeypatch.delenv("base_dir", raising=False)
    # Falls back to a real path instead of raising KeyError (the old behaviour).
    assert config.get_base_dir()


def test_molcas_path_default(monkeypatch):
    monkeypatch.delenv("MOLCAS_PATH", raising=False)
    assert config.get_molcas_path() == "pymolcas"


def test_active_space_constants():
    assert config.NCAS == 6
    assert config.NELECAS == 6
    assert config.N_STATES == 3
    assert config.SORT_MO_INDICES == [19, 20, 21, 22, 23, 24]
    assert config.BASIS_SET_SIZES["sto_6g"] == config.DEFAULT_BASIS_SET_SIZE
