"""Independent analytical checks for M2 assembly mass-property snapshots."""
import numpy as np
import pytest

from edtsim.core import BusParams, PointMassParams, TetherParams
from edtsim.inertia import (
    MassComponent, WoundTetherGeometry, MissionMassModel,
    assemble_mass_properties, box_inertia_about_com,
    parallel_axis_shift, point_mass_inertia_about_origin,
)


def model():
    return MissionMassModel(
        bus=BusParams(mass=4., a=.1, b=.1, c=.3),
        spool_hardware=MassComponent(
            "spool_hardware", .10, [0, 0, .12], np.diag([.00010, .00010, .00005])
        ),
        tether=TetherParams(length=10., mass=.5, attach_B=[0, 0, -.15]),
        end_mass=PointMassParams(.15),
        wound_geometry=WoundTetherGeometry(.015, .04, .06, [0, 0, 1], [0, 0, .12]),
        end_stowed_B=[.02, .01, .13],
    )


@pytest.mark.parametrize("fraction", [0., .01, .25, .5, .8, 1.])
def test_total_mass_conserved_for_deployment(fraction):
    mission = model()
    result = mission.straight(fraction)
    assert result.mass_kg == pytest.approx(4.75)
    assert sum(result.masses_by_name().values()) == pytest.approx(4.75)
    assert np.linalg.eigvalsh(result.inertia_com_B).min() >= -1e-10
    np.testing.assert_allclose(
        result.inertia_bus_B,
        result.inertia_com_B + parallel_axis_shift(result.mass_kg, result.com_B),
        rtol=1e-11, atol=1e-10,
    )
    masses = result.masses_by_name()
    assert masses.get("wound_tether", 0.) == pytest.approx(.5 * (1-fraction))
    assert masses.get("deployed_tether", 0.) == pytest.approx(.5 * fraction)


def test_fully_deployed_matches_independent_analytical_rod_and_endmass():
    m = model()
    r = m.straight(1.)
    L, mt, me = 10., .5, .15
    attach = np.array([0., 0., -.15])
    center = attach + np.array([0., 0., -L/2])
    tip = attach + np.array([0., 0., -L])
    rod_com = np.diag([mt*L**2/12, mt*L**2/12, 0.])
    expected = (box_inertia_about_com(4., .1, .1, .3)
                + np.diag([.0001, .0001, .00005])
                + parallel_axis_shift(.1, [0, 0, .12])
                + rod_com + parallel_axis_shift(mt, center)
                + point_mass_inertia_about_origin(me, tip))
    np.testing.assert_allclose(r.inertia_bus_B, expected, atol=1e-12, rtol=1e-12)
    assert "wound_tether" not in r.masses_by_name()


def test_stowed_end_position_and_com_are_accounted_for():
    m = model()
    r = m.straight(0.)
    expected_com = (.10*np.array([0,0,.12]) + .50*np.array([0,0,.12])
                    + .15*np.array([.02,.01,.13])) / 4.75
    np.testing.assert_allclose(r.com_B, expected_com, atol=1e-14)
    assert "deployed_tether" not in r.masses_by_name()


def test_partial_deployment_tip_and_rod_locations():
    r = model().straight(.5, direction_B=[0, 0, -2])
    components = {x.name: x for x in r.components}
    np.testing.assert_allclose(components["deployed_tether"].com_B, [0,0,-2.65])
    np.testing.assert_allclose(components["end_mass"].com_B, [0,0,-5.15])
    assert components["deployed_tether"].mass_kg == pytest.approx(.25)


def test_bent_two_node_case_agrees_with_analytical_straight_case():
    m = model()
    straight = m.straight(1., direction_B=[1,0,0])
    bent = m.bent(1., [[0,0,-.15],[10,0,-.15]])
    np.testing.assert_allclose(straight.com_B, bent.com_B, atol=2e-14)
    np.testing.assert_allclose(straight.inertia_bus_B, bent.inertia_bus_B, atol=1e-12)
    np.testing.assert_allclose(straight.inertia_com_B, bent.inertia_com_B, atol=1e-12)


def test_bent_polyline_mass_uses_reference_lengths_not_stretched_lengths():
    m = model()
    # 2 material segments, initial rest lengths 3 m and 7 m, each stretched
    # to different instantaneous lengths. They retain masses 0.15 and 0.35 kg.
    b = m.bent(1., [[0,0,-.15], [2,0,-.15], [2,0,-10]], [3.,7.])
    masses = b.masses_by_name()
    assert masses["deployed_tether_segment_0000"] == pytest.approx(.15)
    assert masses["deployed_tether_segment_0001"] == pytest.approx(.35)
    assert b.mass_kg == pytest.approx(4.75)


def test_rod_midpoint_discretization_converges_to_continuous_result():
    m = model()
    analytical = m.straight(1.)
    errors = []
    for n in [4, 8, 16, 32]:
        nodes = np.column_stack((np.zeros(n+1), np.zeros(n+1),
                                 -.15 - np.linspace(0., 10., n+1)))
        # Model each rod as point mass at its midpoint; errors should
        # decrease as O(N^-2). The M2 segment integrals are exact at any N.
        midpoints = (nodes[:-1] + nodes[1:]) / 2
        midpoint_components = [MassComponent.point_mass(
            f"mid_{j}", .5 / n, p) for j, p in enumerate(midpoints)]
        for component in analytical.components:
            if component.name != "deployed_tether":
                midpoint_components.append(component)
        lumped = assemble_mass_properties(midpoint_components)
        errors.append(np.linalg.norm(lumped.inertia_bus_B - analytical.inertia_bus_B))
        exact = m.bent(1., nodes)
        np.testing.assert_allclose(exact.inertia_bus_B, analytical.inertia_bus_B, rtol=1e-12, atol=1e-11)
    assert all(x > y for x, y in zip(errors, errors[1:]))
    np.testing.assert_allclose([errors[i] / errors[i+1] for i in range(3)], [4, 4, 4], rtol=1e-10)


def test_invalid_geometries_and_double_counting_rejected():
    m = model()
    for f in [-.01, 1.01, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.straight(f)
    with pytest.raises(ValueError):
        m.straight(.5, direction_B=[0,0,0])
    with pytest.raises(ValueError):
        m.bent(0., [[0,0,-.15],[1,0,-.15]])
    with pytest.raises(ValueError):
        m.bent(1., [[1,0,0],[2,0,0]])
    with pytest.raises(ValueError):
        m.bent(1., [[0,0,-.15],[2,0,-.15],[4,0,-.15]], [1,2])
    with pytest.raises(ValueError):
        m.bent(1., [[0,0,-.15],[2,0,-.15]], [-10])
    with pytest.raises(ValueError):
        assemble_mass_properties([MassComponent.point_mass("dup", 1., [0,0,0]),
                                  MassComponent.point_mass("dup", 2., [0,0,1])])


def test_tensor_and_arrays_validated_and_protected():
    for bad in [-1, 0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            MassComponent.point_mass("bad", bad, [0,0,0])
    with pytest.raises(ValueError):
        MassComponent("bad", 1., [0,0,0], np.diag([1.,1.,4.]))
    with pytest.raises(ValueError):
        WoundTetherGeometry(.04, .015, .06, [0,0,1], [0,0,0])
    p = np.array([.1, .2, .3])
    c = MassComponent.point_mass("p", 1., p)
    p[0] = 9.
    assert c.com_B[0] == pytest.approx(.1)
    with pytest.raises(ValueError):
        c.com_B[0] = 9.


def test_fixed_component_mass_budget_is_separate_and_conserved():
    original = model()
    extra = MassComponent("camera", .25, [.02,.01,.1], np.diag([.001,.002,.003]))
    mission = MissionMassModel(original.bus, original.spool_hardware, original.tether,
                               original.end_mass, original.wound_geometry,
                               original.end_stowed_B, (extra,))
    assert mission.straight(0.).mass_kg == pytest.approx(5.)
    assert mission.straight(.5).mass_kg == pytest.approx(5.)
    assert mission.straight(1.).mass_kg == pytest.approx(5.)
