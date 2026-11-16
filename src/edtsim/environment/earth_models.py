"""Optional real-main-field and thermosphere model adapters.

IGRF14: ppigrf (east, north, up), WMM2025: wmm-calculator (north, east, down).
All outputs transformed from WGS84 local NED -> ITRS -> GCRS and nT -> T.
NRLMSIS 2.0: pymsis; forcings MUST be explicit to avoid network downloads.
These are environmental reference models, not field/plasma storm predictions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .epoch_frames import EarthFrame, utc_datetime, epoch_at
from .fields import ned_to_ecef_matrix


def _valid_field_input(t_s, r):
    t = float(t_s)
    x = np.asarray(r, dtype=float)
    if not np.isfinite(t) or x.shape != (3,) or not np.all(np.isfinite(x)):
        raise ValueError('invalid t_s or inertial position')
    return t, x


@dataclass
class GeomagneticMainField:
    """Provider `sample(t_s, position_I_m)` returns Tesla in inertial GCRS.

    model='igrf14' requires ppigrf containing IGRF14 coefficients.
    model='wmm2025' requires wmm-calculator, valid in [2025, 2030).
    WMM's time input in this adapter is day-resolution; supported illustrative
    altitude interval is 0..850 km by conservative project choice.
    """
    epoch_utc: object
    model: str = 'igrf14'

    def __post_init__(self):
        self.epoch_utc = utc_datetime(self.epoch_utc)
        if self.model not in ('igrf14', 'wmm2025'):
            raise ValueError('model must be igrf14 or wmm2025')
        self.frame = EarthFrame(self.epoch_utc)

    def _field_ned_nT(self, date, lat, lon, alt_km):
        if self.model == 'igrf14':
            if not 1900 <= date.year <= 2030:
                raise ValueError('IGRF14 reference coefficient epoch outside 1900..2030')
            try:
                import ppigrf
            except ImportError as exc:
                raise RuntimeError("IGRF14 requires ppigrf: pip install -e '.[earth-models]'") from exc
            # ppigrf expects naive UTC datetime; inputs are lon, lat, WGS84 km.
            be, bn, bu = ppigrf.igrf(lon, lat, alt_km, date.replace(tzinfo=None))
            return np.array([float(np.asarray(bn).item()), float(np.asarray(be).item()),
                             -float(np.asarray(bu).item())])
        if not (2025 <= date.year < 2030):
            raise ValueError('WMM2025 coefficient validity: 2025 <= year < 2030')
        if not 0 <= alt_km <= 850:
            raise ValueError('WMM adapter altitude restricted to 0..850 km')
        try:
            from wmm import wmm_calc
        except ImportError as exc:
            raise RuntimeError("WMM2025 requires wmm-calculator: pip install -e '.[earth-models]'") from exc
        model = wmm_calc()
        model.setup_time([date.year], [date.month], [date.day])
        model.setup_env([lat], [lon], [alt_km])
        field = model.get_all()
        return np.array([float(np.asarray(field[key]).item()) for key in ('x', 'y', 'z')])

    def sample(self, t_s, position_I_m):
        t, position = _valid_field_input(t_s, position_I_m)
        lat, lon, alt_km = self.frame.geodetic(t, position)
        local_ned_nT = self._field_ned_nT(epoch_at(self.epoch_utc,t), lat,lon,alt_km)
        if not np.all(np.isfinite(local_ned_nT)):
            raise ValueError('geomagnetic provider returned non-finite values')
        local_to_ecef = ned_to_ecef_matrix(np.deg2rad(lat),np.deg2rad(lon))
        B_E_T = local_to_ecef @ local_ned_nT * 1e-9
        return self.frame.vector_ecef_to_inertial(t, B_E_T)


@dataclass
class MSIS2AtmosphericDensity:
    """NRLMSIS 2.0 mass density [kg/m^3], explicit activity inputs.

    f107: previous-day daily 10.7 cm solar flux; f107a: 81-day mean;
    ap: representative 3-hour/daily geomagnetic Ap (7-entry constant history).
    This is a scenario approximation to the full magnetic index history.
    """
    epoch_utc: object
    f107: float
    f107a: float
    ap: float

    def __post_init__(self):
        self.epoch_utc = utc_datetime(self.epoch_utc)
        for name in ('f107','f107a','ap'):
            value = float(getattr(self,name))
            if not np.isfinite(value) or (value <= 0 if name != 'ap' else value < 0):
                raise ValueError(f'{name} invalid')
            setattr(self,name,value)
        self.frame = EarthFrame(self.epoch_utc)

    def sample(self, t_s, position_I_m):
        t,position = _valid_field_input(t_s,position_I_m)
        lat, lon, alt_km = self.frame.geodetic(t,position)
        if not 0 <= alt_km <= 1000:
            raise ValueError('MSIS2 project envelope: 0..1000 km')
        try:
            import pymsis
        except ImportError as exc:
            raise RuntimeError("NRLMSIS 2.0 needs pymsis: pip install -e '.[earth-models]'") from exc
        date = epoch_at(self.epoch_utc,t)
        date64 = np.datetime64(date.replace(tzinfo=None), 'ms')
        data = pymsis.calculate(np.array([date64]), np.array([lon]), np.array([lat]),
                np.array([alt_km]), f107s=np.array([self.f107]),
                f107as=np.array([self.f107a]), aps=np.array([[self.ap]*7]), version=2.0)
        density = float(np.asarray(data)[..., pymsis.Variable.MASS_DENSITY].reshape(-1)[0])
        if not np.isfinite(density) or density < 0:
            raise ValueError('MSIS returned invalid mass density')
        return density
