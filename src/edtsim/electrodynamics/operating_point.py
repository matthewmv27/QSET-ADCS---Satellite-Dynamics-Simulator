"""M4e diagnostic coupling of prescribed-bias OML with empirical CNT capacity.

NOT an electrical operating-point solution: bus-to-plasma bias, electron density,
T_e and motional EMF are prescribed.  CNT gate voltage is an independent
quantity; requiring it does not establish a closed electrical/plasma loop.
"""
from dataclasses import dataclass
import numpy as np
from .cnt_cathode import EmpiricalCNTCathode
from .oml_collection import PrescribedBusPotentialOML
from .profile import SpatialCurrentProfile
from edtsim.tether.geometry import HomogeneousTether


def _nonnegative(x, name):
    value = float(x)
    if not np.isfinite(value) or value < 0:
        raise ValueError(name+' must be finite and nonnegative')
    return value


@dataclass(frozen=True)
class CathodeDemandCheck:
    current_required_A: float
    gate_voltage_required_V: float
    estimated_cathode_power_W: float
    within_gate_voltage_limit: bool
    within_power_limit: bool
    within_assumed_current_limit: bool
    supports_assumed_demand: bool
    warnings: tuple
    status: str = 'diagnostic only; imposed bus-to-plasma bias, no electrical circuit closure'


def check_cnt_demand(current_required_A, cathode=None, *, equivalent_age_h=0.,
                     available_gate_voltage_V, max_cathode_power_W=None,
                     max_emission_current_A=None):
    """Compare an OML *demand* with approximate CNT capability constraints.

    If the limits pass this is NECESSARY but NOT SUFFICIENT for circuit operation.
    """
    emitter = cathode if cathode is not None else EmpiricalCNTCathode()
    if not isinstance(emitter, EmpiricalCNTCathode):
        raise TypeError('cathode must be EmpiricalCNTCathode')
    requested = _nonnegative(current_required_A, 'current_required_A')
    gate_limit = _nonnegative(available_gate_voltage_V, 'available_gate_voltage_V')
    age = _nonnegative(equivalent_age_h, 'equivalent_age_h')
    max_power = None if max_cathode_power_W is None else _nonnegative(max_cathode_power_W, 'max_cathode_power_W')
    max_current = None if max_emission_current_A is None else _nonnegative(max_emission_current_A, 'max_emission_current_A')
    voltage = emitter.required_voltage_V(requested, age)
    power = 0. if requested == 0 else emitter.estimated_cathode_power_W(requested, voltage)
    vg_ok = voltage <= gate_limit
    power_ok = max_power is None or power <= max_power
    current_ok = max_current is None or requested <= max_current
    warnings = (tuple(emitter.support_warnings(voltage, age)) if voltage > 0 else ())
    warnings += ('Passing limits does not close plasma return, bus charging, or circuit power balance.',)
    return CathodeDemandCheck(requested, voltage, power, bool(vg_ok), bool(power_ok),
                              bool(current_ok), bool(vg_ok and power_ok and current_ok), warnings)


@dataclass(frozen=True)
class PrescribedBiasElectricalSnapshot:
    profile: SpatialCurrentProfile
    oml_solution: object
    cathode_check: CathodeDemandCheck
    motional_E_along_V_m: float
    assumptions: tuple


def solve_prescribed_bias_snapshot(tether, *, electron_density_m3, electron_temperature_K,
                                  bus_plasma_bias_V, motional_E_along_V_m,
                                  available_gate_voltage_V, equivalent_age_h=0.,
                                  max_cathode_power_W=None, max_emission_current_A=None,
                                  nodes=101, cathode=None):
    """Calculate OML profile and independent CNT requirement; do not infer closure."""
    if not isinstance(tether, HomogeneousTether):
        raise TypeError('tether must be HomogeneousTether')
    solver = PrescribedBusPotentialOML(tether, electron_density_m3,
                                      electron_temperature_K, bus_plasma_bias_V,
                                      motional_E_along_V_m)
    solution = solver.solve(nodes=nodes)
    check = check_cnt_demand(solution.bus_emission_current_required_A, cathode,
                equivalent_age_h=equivalent_age_h,
                available_gate_voltage_V=available_gate_voltage_V,
                max_cathode_power_W=max_cathode_power_W,
                max_emission_current_A=max_emission_current_A)
    return PrescribedBiasElectricalSnapshot(
        SpatialCurrentProfile.from_oml_solution(solution), solution, check,
        float(motional_E_along_V_m),
        ('Prescribed bus-to-plasma potential and scalar motional E along tether',
         'Plasma density and temperature uniform; simplified OML collection geometry',
         'CNT empirical gate-voltage requirement is not bus-to-plasma potential',
         'No circuit closure, spacecraft charging, sheath, return plasma, or electronics losses'))
