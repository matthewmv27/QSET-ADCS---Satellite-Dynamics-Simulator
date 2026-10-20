"""Random stowed end-mass placement inside the spool."""
from __future__ import annotations

import numpy as np

from edtsim.core.rotations import rotation_matrix_from_a_to_b

__all__ = ["sample_point_in_annular_cylinder"]


def sample_point_in_annular_cylinder(r_i: float, r_o: float, h: float, axis_s_B, com_B,
                                     rng: np.random.Generator) -> np.ndarray:
    """Draw one point uniformly (by volume) from an annular cylinder, in body-frame coordinates.

    The cylinder has inner/outer radius ``r_i``/``r_o``, height ``h``, symmetry axis ``axis_s_B``
    and centre ``com_B``. The order of RNG calls (radius, angle, height) matches the legacy
    notebook, so the same seed gives the same point.
    """
    u_rand = rng.random()
    r = np.sqrt(r_i**2 + u_rand * (r_o**2 - r_i**2))
    phi = 2.0 * np.pi * rng.random()
    z = (rng.random() - 0.5) * h

    p_local = np.array([r * np.cos(phi), r * np.sin(phi), z], dtype=float)
    R = rotation_matrix_from_a_to_b(np.array([0.0, 0.0, 1.0]), axis_s_B)
    return (R @ p_local) + np.asarray(com_B, dtype=float).reshape(3)
