"""Build a geometry scan by interpolating between two endpoint geometries.

Run with:

    uv run examples/02_interpolate_geometries.py

Reads the bundled water start/end geometries, interpolates a short scan, writes
the frames to a temporary directory, and reads one back to confirm the round
trip. Uses only the lightweight core (NumPy).
"""

import tempfile
from pathlib import Path

from casnet.data.utils import interpolate_geometry, read_xyz_file, write_xyz_file

DATA = Path(__file__).parent / "data"


def main() -> None:
    start = read_xyz_file(str(DATA / "water_start.xyz"))
    end = read_xyz_file(str(DATA / "water_end.xyz"))
    print(f"start O-H_x = {start[1].x:.5f}, end O-H_x = {end[1].x:.5f}")

    n_frames = 5
    scan = interpolate_geometry(start, end, n_frames)
    print(f"interpolated {len(scan)} frames")
    print("H1 x-coordinate per frame:", [round(frame[1].x, 5) for frame in scan])

    with tempfile.TemporaryDirectory() as tmp:
        for idx, frame in enumerate(scan):
            write_xyz_file(frame, str(Path(tmp) / f"geometry_{idx}.xyz"))
        roundtrip = read_xyz_file(str(Path(tmp) / "geometry_2.xyz"))
        print(f"wrote {n_frames} frames; midpoint H1 x = {roundtrip[1].x:.5f}")


if __name__ == "__main__":
    main()
