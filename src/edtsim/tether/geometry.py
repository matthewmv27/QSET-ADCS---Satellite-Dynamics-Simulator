"""M4 design geometry: compute physical properties from cross-section inputs.

Static, uniform homogeneous sections only. No spool winding, thermal gradients,
mechanical instability, or plasma sheath enhancement is inferred.
"""
from dataclasses import dataclass
import numpy as np


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('%s must be finite and positive' % name)
    return value


def _unit(value, name):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-14:
        raise ValueError('%s must be a finite nonzero 3-vector' % name)
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class RoundWire:
    diameter_m: float

    def __post_init__(self):
        object.__setattr__(self, 'diameter_m', _positive(self.diameter_m, 'diameter_m'))

    @property
    def area_m2(self):
        return np.pi * self.diameter_m**2 / 4

    @property
    def perimeter_m(self):
        return np.pi * self.diameter_m

    def segment_projected_area_m2(self, length_m, axis_I, flow_I, width_dir_I=None):
        """Illustrative geometric silhouette, including a circular end cap."""
        length = _positive(length_m, 'length_m')
        axis = _unit(axis_I, 'axis_I')
        flow = np.asarray(flow_I, float)
        if flow.shape != (3,) or not np.all(np.isfinite(flow)):
            raise ValueError('flow_I must be finite 3-vector')
        if np.linalg.norm(flow) == 0:
            return 0.0
        u = flow / np.linalg.norm(flow)
        return float(self.diameter_m*length*np.linalg.norm(np.cross(u, axis)) +
                     self.area_m2*abs(np.dot(u, axis)))


@dataclass(frozen=True)
class RectangularTape:
    width_m: float
    thickness_m: float

    def __post_init__(self):
        object.__setattr__(self, 'width_m', _positive(self.width_m, 'width_m'))
        object.__setattr__(self, 'thickness_m', _positive(self.thickness_m, 'thickness_m'))

    @property
    def area_m2(self):
        return self.width_m*self.thickness_m

    @property
    def perimeter_m(self):
        return 2*(self.width_m+self.thickness_m)

    def segment_projected_area_m2(self, length_m, axis_I, flow_I, width_dir_I):
        """Projected silhouette of rectangular prism including its end face.

        width_dir_I is perpendicular to the local tether axis (otherwise
        projected/normalized before use). Approximate Cd must be supplied separately.
        """
        length = _positive(length_m, 'length_m')
        axis = _unit(axis_I, 'axis_I')
        w = _unit(width_dir_I, 'width_dir_I')
        w = w - np.dot(w, axis)*axis
        if np.linalg.norm(w) < 1e-10:
            raise ValueError('width_dir_I cannot be parallel to tether axis')
        w = w / np.linalg.norm(w)
        thickness_dir = np.cross(axis, w)
        flow = np.asarray(flow_I, float)
        if flow.shape != (3,) or not np.all(np.isfinite(flow)):
            raise ValueError('flow_I must be finite 3-vector')
        if np.linalg.norm(flow) == 0:
            return 0.0
        u = flow / np.linalg.norm(flow)
        return float(length*(self.width_m*abs(u @ thickness_dir)
                              + self.thickness_m*abs(u @ w))
                     + self.area_m2*abs(u @ axis))


@dataclass(frozen=True)
class HomogeneousTether:
    """Physical properties for a conductor, not a complete electrical circuit."""
    cross_section: object
    length_m: float
    density_kg_m3: float
    young_modulus_Pa: float
    conductivity_S_m: float
    bare_fraction: float = 1.0

    def __post_init__(self):
        if not isinstance(self.cross_section, (RoundWire, RectangularTape)):
            raise TypeError('cross_section must be RoundWire or RectangularTape')
        for name in ('length_m', 'density_kg_m3', 'young_modulus_Pa', 'conductivity_S_m'):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        f = float(self.bare_fraction)
        if not np.isfinite(f) or not 0 <= f <= 1:
            raise ValueError('bare_fraction must be in [0,1]')
        object.__setattr__(self, 'bare_fraction', f)

    @property
    def linear_density_kg_m(self):
        return self.density_kg_m3*self.cross_section.area_m2

    @property
    def mass_kg(self):
        return self.length_m*self.linear_density_kg_m

    @property
    def axial_rigidity_N(self):
        return self.young_modulus_Pa*self.cross_section.area_m2

    @property
    def resistance_ohm(self):
        return self.length_m/(self.conductivity_S_m*self.cross_section.area_m2)

    @property
    def bare_length_m(self):
        return self.length_m*self.bare_fraction

    @property
    def collecting_perimeter_m(self):
        """Geometric exposed perimeter ONLY, not sheath-enhanced OML perimeter."""
        return self.cross_section.perimeter_m
