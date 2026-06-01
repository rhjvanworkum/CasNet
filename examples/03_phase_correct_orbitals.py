"""Demonstrate orbital phase correction along a sequence of geometries.

Run with:

    uv run examples/03_phase_correct_orbitals.py

CASSCF orbital coefficients can flip sign arbitrarily between nearby geometries.
:func:`casnet.data.utils.phase_correct_orbitals` aligns each set of orbitals to
the previous one so the ML targets vary smoothly. Uses only the core (NumPy).
"""

import numpy as np

from casnet.data.utils import phase_correct_orbitals


def main() -> None:
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((4, 4))

    # Simulate the "same" orbitals at the next geometry, but with two columns
    # sign-flipped (as a real solver might return them).
    target = reference.copy()
    target[:, 1] *= -1
    target[:, 3] *= -1

    corrected = phase_correct_orbitals(reference, target)

    flipped_before = [j for j in range(4) if np.dot(reference[:, j], target[:, j]) < 0]
    flipped_after = [j for j in range(4) if np.dot(reference[:, j], corrected[:, j]) < 0]
    print(f"columns misaligned before correction: {flipped_before}")
    print(f"columns misaligned after correction:  {flipped_after}")
    print(f"corrected orbitals match reference: {np.allclose(corrected, reference)}")


if __name__ == "__main__":
    main()
