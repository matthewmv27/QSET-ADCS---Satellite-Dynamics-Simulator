import numpy as np
import pytest

from edtsim.core import BusParams, PointMassParams, SpoolParams, TetherParams


def test_lists_are_coerced_to_float_vectors():
    s = SpoolParams(0.5, 0.01, 0.04, 0.05, [0, 0, 1], (0, 0, 0.1))
    assert s.axis_s_B.dtype == float and s.axis_s_B.shape == (3,)
    assert s.com_B.shape == (3,)
    t = TetherParams(10, 0.1, [0, 0, -0.1])
    assert t.attach_B.shape == (3,)


@pytest.mark.parametrize("bad", [
    lambda: BusParams(0.0, 0.1, 0.1, 0.1),
    lambda: BusParams(1.0, -0.1, 0.1, 0.1),
    lambda: SpoolParams(0.5, 0.05, 0.04, 0.05, [0, 0, 1], [0, 0, 0]),      # r_o <= r_i
    lambda: SpoolParams(0.5, 0.0, 0.04, 0.05, [0, 0, 0], [0, 0, 0]),       # zero axis
    lambda: SpoolParams(0.5, 0.0, 0.04, 0.05, [0, 1], [0, 0, 0]),          # wrong length
    lambda: TetherParams(0.0, 0.1, [0, 0, 0]),
    lambda: TetherParams(10.0, -0.1, [0, 0, 0]),
    lambda: PointMassParams(-1.0),
])
def test_invalid_parameters_raise(bad):
    with pytest.raises(ValueError):
        bad()


def test_zero_end_mass_and_zero_inner_radius_allowed():
    PointMassParams(0.0)
    SpoolParams(0.5, 0.0, 0.04, 0.05, [0, 0, 1], [0, 0, 0])
