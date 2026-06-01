"""Generate a geometry scan by linearly interpolating between two endpoints.

Previously this script duplicated the ``Atom`` / XYZ-I/O helpers inline and hard-coded
absolute paths; it now reuses :mod:`casnet.data.utils` and takes paths as CLI arguments.
"""

from __future__ import annotations

import argparse

from casnet.data.utils import interpolate_geometry, read_xyz_file, write_xyz_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="XYZ file for the first endpoint")
    parser.add_argument("--end", required=True, help="XYZ file for the last endpoint")
    parser.add_argument("--n", type=int, required=True, help="Number of interpolated geometries")
    parser.add_argument(
        "--out-pattern",
        required=True,
        help="Output path pattern containing '{idx}', e.g. './geom_scan/geometry_{idx}.xyz'",
    )
    args = parser.parse_args()

    start = read_xyz_file(args.start)
    end = read_xyz_file(args.end)
    geometries = interpolate_geometry(start, end, args.n)

    for idx, geometry in enumerate(geometries):
        write_xyz_file(geometry, args.out_pattern.format(idx=idx))


if __name__ == "__main__":
    main()
