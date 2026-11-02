"""M2: mass-conserving, configuration-dependent *assembly* mass properties.

All positions and tensors use body axes B; the origin is the rigid bus COM.
This module computes static mass-property snapshots, **not** deployment
kinematics, flexible-body angular momentum, or the bus's rigid-body inertia.

The remaining wound tether is approximated by a *uniform annulus of fixed
geometry* and decreasing mass.  Actual winding-layer geometry requires
flight-specific spool dimensions/winding information and is not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

from edtsim.core.params import BusParams, PointMassParams, TetherParams
from edtsim.core.states import MassProperties
from edtsim.inertia.primitives import (
    annular_cylinder_inertia_about_com,
    box_inertia_about_com,
    parallel_axis_shift,
    slender_rod_inertia_about_com,
)

__all__ = [
    "MassComponent",
    "WoundTetherGeometry",
    "MissionMassModel",
    "AssemblyMassResult",
    "assemble_mass_properties",
]


def _vec3(value, name: str) -> np.ndarray:
    a = np.array(value, dtype=float, copy=True)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite vector with shape (3,).")
    a.setflags(write=False)
    return a


def _tensor(value, name: str) -> np.ndarray:
    a = np.array(value, dtype=float, copy=True)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3x3 tensor.")
    scale = max(1.0, float(np.max(np.abs(a))))
    if not np.allclose(a, a.T, rtol=0.0, atol=1e-12 * scale):
        raise ValueError(f"{name} must be symmetric.")
    moments = np.linalg.eigvalsh((a + a.T) / 2.0)
    if moments[0] < -1e-12 * scale or moments[0] + moments[1] < moments[2] - 1e-12 * scale:
        raise ValueError(f"{name} is not a physically admissible inertia tensor.")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class MassComponent:
    """A separate mass-budget item, with COM and COM-inertia in body axes.

    Mass cannot be negative or zero. A zero-mass contribution should be omitted
    from the assembly instead. All inputs are copied and made read-only.
    """

    name: str
    mass_kg: float
    com_B: np.ndarray
    inertia_com_B: np.ndarray

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Component name cannot be empty.")
        m = float(self.mass_kg)
        if not np.isfinite(m) or m <= 0:
            raise ValueError("Component mass_kg must be finite and positive.")
        object.__setattr__(self, "mass_kg", m)
        object.__setattr__(self, "com_B", _vec3(self.com_B, "com_B"))
        object.__setattr__(self, "inertia_com_B", _tensor(self.inertia_com_B, "inertia_com_B"))

    @classmethod
    def point_mass(cls, name: str, mass_kg: float, position_B) -> "MassComponent":
        return cls(name, mass_kg, position_B, np.zeros((3, 3)))


@dataclass(frozen=True)
class WoundTetherGeometry:
    """Approximate distribution of *remaining tether* on a fixed spool.

    Does not include spool-hardware mass. For intermediate configurations, the
    annulus remains fixed in size; this is an explicit geometric approximation.
    """

    r_inner_m: float
    r_outer_m: float
    width_m: float
    axis_B: np.ndarray
    com_B: np.ndarray

    def __post_init__(self):
        ri, ro, w = float(self.r_inner_m), float(self.r_outer_m), float(self.width_m)
        if (not all(np.isfinite(x) for x in (ri, ro, w)) or ri < 0 or ro <= ri or w <= 0):
            raise ValueError("Wound-tether geometry needs finite 0 <= inner radius < outer radius and width > 0.")
        object.__setattr__(self, "r_inner_m", ri)
        object.__setattr__(self, "r_outer_m", ro)
        object.__setattr__(self, "width_m", w)
        axis = _vec3(self.axis_B, "axis_B")
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("axis_B must be nonzero.")
        object.__setattr__(self, "axis_B", axis)
        object.__setattr__(self, "com_B", _vec3(self.com_B, "com_B"))


@dataclass(frozen=True)
class AssemblyMassResult:
    """Combined mass properties and auditable components for one configuration.

    ``properties.inertia_com_B`` is about the combined system COM;
    ``inertia_bus_B`` is about the rigid bus COM. Both are expressed in B axes.
    These are assembly diagnostics, not an attitude-solver bus inertia.
    """

    properties: MassProperties
    inertia_bus_B: np.ndarray
    components: Tuple[MassComponent, ...]

    def __post_init__(self):
        object.__setattr__(self, "inertia_bus_B", _tensor(self.inertia_bus_B, "inertia_bus_B"))
        object.__setattr__(self, "components", tuple(self.components))

    @property
    def mass_kg(self) -> float:
        return self.properties.mass_kg

    @property
    def com_B(self) -> np.ndarray:
        return self.properties.com_B

    @property
    def inertia_com_B(self) -> np.ndarray:
        return self.properties.inertia_com_B

    def masses_by_name(self) -> dict:
        return {component.name: component.mass_kg for component in self.components}


def assemble_mass_properties(components: Sequence[MassComponent]) -> AssemblyMassResult:
    """Aggregate components using the parallel-axis theorem around bus COM.

    This function does not allow duplicated names, which helps prevent double
    counting the same mass-budget entry across spacecraft configurations.
    """
    components = tuple(components)
    if not components or not all(isinstance(x, MassComponent) for x in components):
        raise ValueError("At least one valid MassComponent is required.")
    names = [c.name for c in components]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate component names are not allowed.")
    mass = sum(c.mass_kg for c in components)
    com = sum((c.mass_kg * c.com_B for c in components), np.zeros(3)) / mass
    inertia_bus = sum(
        (c.inertia_com_B + parallel_axis_shift(c.mass_kg, c.com_B) for c in components),
        np.zeros((3, 3)),
    )
    inertia_com = inertia_bus - parallel_axis_shift(mass, com)
    # Roundoff at long tether lengths can leave ~eps asymmetric entries.
    inertia_com = 0.5 * (inertia_com + inertia_com.T)
    inertia_bus = 0.5 * (inertia_bus + inertia_bus.T)
    return AssemblyMassResult(MassProperties(mass, com, inertia_com), inertia_bus, components)


@dataclass(frozen=True)
class MissionMassModel:
    """One consistent mass inventory for stowed, partial and deployed states.

    ``bus`` mass excludes spool hardware, tether and end mass, to prevent
    double counting. ``spool_hardware`` is always present and has constant mass.
    ``end_stowed_B`` only applies at exactly zero deployment; this is a static
    configuration model, not a continuous deployment trajectory.

    For a straight tether, each deployed segment is idealized as a uniform
    zero-radius rod. For a bent tether, it is a chain of uniform straight rods.
    """

    bus: BusParams
    spool_hardware: MassComponent
    tether: TetherParams
    end_mass: PointMassParams
    wound_geometry: WoundTetherGeometry
    end_stowed_B: np.ndarray
    other_fixed: Tuple[MassComponent, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.bus, BusParams) or not isinstance(self.tether, TetherParams):
            raise TypeError("bus and tether must use BusParams and TetherParams.")
        if not isinstance(self.end_mass, PointMassParams):
            raise TypeError("end_mass must use PointMassParams.")
        if not isinstance(self.spool_hardware, MassComponent) or not isinstance(self.wound_geometry, WoundTetherGeometry):
            raise TypeError("spool_hardware and wound_geometry must be their respective types.")
        if self.tether.mass <= 0 or self.end_mass.mass <= 0:
            raise ValueError("An M2 mission model requires positive tether and end masses.")
        object.__setattr__(self, "end_stowed_B", _vec3(self.end_stowed_B, "end_stowed_B"))
        others = tuple(self.other_fixed)
        if not all(isinstance(c, MassComponent) for c in others):
            raise TypeError("other_fixed must contain MassComponent objects.")
        object.__setattr__(self, "other_fixed", others)
        reserved = {"bus", "spool_hardware", "wound_tether", "deployed_tether", "end_mass"}
        if any(c.name in reserved for c in others) or self.spool_hardware.name != "spool_hardware":
            raise ValueError("Use name spool_hardware for the fixed reel and distinct names for extra components.")
        names = [c.name for c in others]
        if len(names) != len(set(names)):
            raise ValueError("Extra component names must be unique.")

    @property
    def total_mass_kg(self) -> float:
        return (self.bus.mass + self.spool_hardware.mass_kg + self.tether.mass
                + self.end_mass.mass + sum(c.mass_kg for c in self.other_fixed))

    def _fixed(self) -> list:
        bus = MassComponent("bus", self.bus.mass, [0., 0., 0.],
                            box_inertia_about_com(self.bus.mass, self.bus.a, self.bus.b, self.bus.c))
        return [bus, self.spool_hardware, *self.other_fixed]

    def _wound(self, fraction: float) -> list:
        mass = self.tether.mass * (1.0 - fraction)
        if mass <= 0:
            return []
        g = self.wound_geometry
        inertia = annular_cylinder_inertia_about_com(mass, g.r_inner_m, g.r_outer_m, g.width_m, g.axis_B)
        return [MassComponent("wound_tether", mass, g.com_B, inertia)]

    def _completed(self, components: list) -> AssemblyMassResult:
        result = assemble_mass_properties(components)
        if not np.isclose(result.mass_kg, self.total_mass_kg, rtol=1e-12, atol=1e-12):
            raise ValueError("Mass accounting failed: configuration mass differs from inventory.")
        return result

    def straight(self, fraction: float, direction_B=(0., 0., -1.)) -> AssemblyMassResult:
        """Configuration snapshot with ``fraction`` in [0,1] paid out in a straight line.

        f=0: end mass is at ``end_stowed_B``; f>0: end mass at the tip,
        which is ``tether.attach_B + f * tether.length * direction_B``.
        A discontinuous stowed-to-first-extension COM is possible because
        deployment travel/storage mechanics are deliberately not modelled.
        """
        f = _fraction(fraction)
        direction = _vec3(direction_B, "direction_B")
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            raise ValueError("direction_B cannot be zero.")
        direction = direction / norm
        components = self._fixed() + self._wound(f)
        if f > 0:
            L, m = self.tether.length * f, self.tether.mass * f
            center = self.tether.attach_B + 0.5 * L * direction
            components.append(MassComponent("deployed_tether", m, center,
                                            slender_rod_inertia_about_com(m, L, direction)))
            end_pos = self.tether.attach_B + L * direction
        else:
            end_pos = self.end_stowed_B
        components.append(MassComponent.point_mass("end_mass", self.end_mass.mass, end_pos))
        return self._completed(components)

    def bent(self, fraction: float, nodes_B, reference_lengths_m=None) -> AssemblyMassResult:
        """Bent deployed tether as exact uniform straight-segment integrals.

        N>=2 node positions are in B, starting at ``tether.attach_B``; the
        endpoint is where the end mass lies. Mass is apportioned by *unstretched*
        segment lengths (uniform if omitted), NOT instantaneous segment lengths.
        This handles nonuniform stretch without silently shifting material mass.
        For f=0 use ``straight(0)``; it has no deployed nodes.
        """
        f = _fraction(fraction)
        if f == 0:
            raise ValueError("For zero deployment, use straight(0); bent nodes do not exist.")
        r = np.array(nodes_B, dtype=float, copy=True)
        if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] != 3 or not np.all(np.isfinite(r)):
            raise ValueError("nodes_B must be a finite (N,3) array with N>=2.")
        if not np.allclose(r[0], self.tether.attach_B, atol=1e-10, rtol=0.0):
            raise ValueError("The first node must match tether.attach_B in body axes.")
        n = len(r) - 1
        if reference_lengths_m is None:
            rest_lengths = np.full(n, f * self.tether.length / n)
        else:
            rest_lengths = np.asarray(reference_lengths_m, dtype=float)
            if rest_lengths.shape != (n,) or not np.all(np.isfinite(rest_lengths)) or np.any(rest_lengths <= 0):
                raise ValueError("reference_lengths_m must be (N-1,) positive finite lengths.")
            if not np.isclose(sum(rest_lengths), f * self.tether.length, atol=1e-9, rtol=1e-9):
                raise ValueError("Rest lengths must sum to paid-out tether length.")
        segment_masses = f * self.tether.mass * rest_lengths / sum(rest_lengths)
        components = self._fixed() + self._wound(f)
        for i, m in enumerate(segment_masses):
            delta = r[i + 1] - r[i]
            center = (r[i] + r[i + 1]) * 0.5
            L = np.linalg.norm(delta)
            inertia_com = (slender_rod_inertia_about_com(m, L, delta / L)
                           if L > 0 else np.zeros((3, 3)))
            components.append(MassComponent(f"deployed_tether_segment_{i:04d}", m, center, inertia_com))
        components.append(MassComponent.point_mass("end_mass", self.end_mass.mass, r[-1]))
        return self._completed(components)


def _fraction(value: float) -> float:
    try:
        f = float(value)
    except (ValueError, TypeError):
        raise ValueError("Deployment fraction must be a number in [0,1].") from None
    if not np.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError("Deployment fraction must be finite and in [0,1].")
    return f
