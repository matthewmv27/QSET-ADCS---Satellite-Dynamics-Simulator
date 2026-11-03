import numpy as np
import pytest

from edtsim.core.constants import MU_EARTH, R_EARTH_EQ
from edtsim.environment import (
    CircularReferenceOrbit, sso_inclination_deg, j2_nodal_rate_rad_s,
    SampledMagneticField, UniformMagneticField, ned_to_ecef_matrix,
    corotating_flow_I, motional_electric_field_I, segment_lorentz_force_I,
    aerodynamic_drag_force_I,
)
from edtsim.environment.orbit import mean_sun_rate_rad_s


@pytest.mark.parametrize('alt', [400e3, 450e3, 500e3])
def test_first_order_sso_inclination(alt):
    inclination = sso_inclination_deg(alt)
    assert 96 < inclination < 99
    rate = j2_nodal_rate_rad_s(R_EARTH_EQ + alt, np.deg2rad(inclination))
    assert np.isclose(rate, mean_sun_rate_rad_s(), rtol=1e-12)


def test_circular_state_at_epoch_is_analytic():
    model = CircularReferenceOrbit(450e3, 0, 0, 0, precess_raan=False)
    r, v = model.state(0)
    a = R_EARTH_EQ + 450e3
    np.testing.assert_allclose(r, [a,0,0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(v, [0, np.sqrt(MU_EARTH / a), 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.state(model.period_s)[0], r, atol=2e-8)


def test_orbit_radius_constant_and_tangent_velocity():
    model = CircularReferenceOrbit(500e3, 97.4, 20, 45, precess_raan=True)
    for t in [0, 150, model.period_s*3]:
        r, v = model.state(t)
        assert np.isclose(np.linalg.norm(r), model.radius_m)
        assert abs(np.dot(r, v)) < 1e-5


def test_precessing_velocity_is_time_derivative_of_position():
    model = CircularReferenceOrbit(400e3, sso_inclination_deg(400e3), 50)
    t = 123.4
    h = 0.002
    numerical = (model.state(t+h)[0] - model.state(t-h)[0])/(2*h)
    np.testing.assert_allclose(model.state(t)[1], numerical, rtol=1e-7, atol=1e-3)


def test_uniform_field_and_strict_time_domain():
    constant = UniformMagneticField((1e-5, -2e-5, 3e-5))
    np.testing.assert_array_equal(constant.sample(500, [7000e3, 0, 0]), [1e-5,-2e-5,3e-5])
    grid = SampledMagneticField([0,10,20], [[0,0,0],[1e-5,2e-5,0],[2e-5,4e-5,0]])
    np.testing.assert_allclose(grid.sample(5), [5e-6,1e-5,0])
    np.testing.assert_allclose(grid.sample(20), [2e-5,4e-5,0])
    for t in [-0.001,20.001]:
        with pytest.raises(ValueError, match='outside'):
            grid.sample(t)


def test_field_grid_rejects_ill_formed_data():
    for times, b in [([0,0],[[1,2,3],[4,5,6]]), ([0,10], [[1,2,3]]),([0,1],[[np.nan,0,0],[1,0,0]])]:
        with pytest.raises(ValueError):
            SampledMagneticField(times,b)


def test_ned_ecef_known_reference_at_equator_greenwich():
    R = ned_to_ecef_matrix(0,0)
    np.testing.assert_allclose(R @ [1,0,0],[0,0,1], atol=1e-15)
    np.testing.assert_allclose(R @ [0,1,0],[0,1,0], atol=1e-15)
    np.testing.assert_allclose(R @ [0,0,1],[-1,0,0], atol=1e-15)
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-15)
    assert np.linalg.det(R)==pytest.approx(1)


def test_flow_and_motional_emf_with_known_vectors():
    np.testing.assert_allclose(corotating_flow_I([1e6,0,0], omega_rad_s=1e-4), [0,100,0])
    np.testing.assert_allclose(motional_electric_field_I([0,10,0], [0,2,0], [0,0,2e-5]), [1.6e-4,0,0])
    np.testing.assert_allclose(segment_lorentz_force_I(2, [0,1,0], [0,0,3e-5]), [6e-5,0,0])


def test_drag_direction_and_zero_relative_flow():
    f = aerodynamic_drag_force_I(1e-12, 2, 1, [7700,0,0], [0,0,0])
    assert f[0] < 0 and f[1] == f[2] == 0
    np.testing.assert_array_equal(aerodynamic_drag_force_I(1e-12,2,1,[7700,0,0],[7700,0,0]), [0,0,0])


@pytest.mark.parametrize('bad',[-1,0,float('nan')])
def test_invalid_altitude(bad):
    with pytest.raises(ValueError):
        sso_inclination_deg(bad)


def test_cached_field_rejects_arbitrary_positions():
    grid = SampledMagneticField([0,10], [[0,0,0],[1e-5,0,0]])
    with pytest.raises(ValueError, match='reference trajectory'):
        grid.sample(5,[7000e3,0,0])
