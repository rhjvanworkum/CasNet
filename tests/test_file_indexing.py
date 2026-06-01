"""Tests for filename index parsing and geometry-file sorting."""

import pytest

from casnet.data.utils import (
    find_all_geometry_files_in_folder,
    parse_index_from_filename,
    sort_geometry_files_by_idx,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("geometry_0.xyz", 0),
        ("geometry_199.xyz", 199),
        ("/abs/path/calc_42.npz", 42),
        ("fulvene_md_traj_13.xyz", 13),
    ],
)
def test_parse_index_valid(filename, expected):
    assert parse_index_from_filename(filename) == expected


@pytest.mark.parametrize("filename", ["nodigits.xyz", "geometry_abc.xyz", "plain"])
def test_parse_index_invalid_returns_none(filename):
    assert parse_index_from_filename(filename) is None


def test_sort_geometry_files_by_idx_orders_numerically():
    files = ["d/geometry_10.xyz", "d/geometry_2.xyz", "d/geometry_1.xyz"]
    assert sort_geometry_files_by_idx(files) == [
        "d/geometry_1.xyz",
        "d/geometry_2.xyz",
        "d/geometry_10.xyz",
    ]


def test_sort_geometry_files_raises_on_unparseable():
    with pytest.raises(ValueError):
        sort_geometry_files_by_idx(["d/bad.xyz"])


def test_find_all_geometry_files(tmp_path):
    (tmp_path / "geometry_0.xyz").write_text("0\n\n")
    (tmp_path / "geometry_1.xyz").write_text("0\n\n")
    (tmp_path / "notes.txt").write_text("ignore me")
    found = find_all_geometry_files_in_folder(str(tmp_path) + "/")
    assert sorted(f.split("/")[-1] for f in found) == ["geometry_0.xyz", "geometry_1.xyz"]
