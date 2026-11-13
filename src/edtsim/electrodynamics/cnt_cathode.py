"""Preliminary empirical CNT field-emission cathode *capacity* model.

Adapted from the user-provided 'CNT Current Profile.py' (13 Nov 2026).
The digitized Fig. 1–3 values have no attached source paper or uncertainty
budget. These are approximate, NOT measured flight hardware characteristics.
Fig. 1/2 source current curves are NOT the same cathode as the emitted-current
age anchor (Fig. 3); anchoring is empirical, not transmission efficiency.

Emitted-current capacity [A] does NOT imply that the tether/plasma circuit can
collect or sustain this current. No OML/current-profile model is provided here.
Age is equivalent *hours*, not seconds; supply limits are scenario predictions.
"""
from dataclasses import dataclass
import numpy as np

AGE_H = np.array([0,50,100,150,200,250,300,350,400,450,500], dtype=float)
VOLTAGE_AT_2P5MA_V = np.array([248,265,281,292,303,320,337,356,375,393,411], dtype=float)
FRESH_IV_V_MA = np.array([[125,.10],[150,.40],[175,.90],[200,1.70],[225,3],[240,4],[248,4.70]])
AGED_IV_V_MA = np.array([[300,.20],[350,.75],[375,1.20],[400,2],[420,2.90],[432,3.50]])
I_ANCHOR_A = 2.5e-3
ASSUMED_POWER_FACTOR = 0.70
ASSUMED_WEAR_EXPONENT = 1.0


def _positive(value, name, allow_zero=False):
    x = float(value)
    if not np.isfinite(x) or (x < 0 if allow_zero else x <= 0):
        raise ValueError('%s must be finite and %s' % (name, 'nonnegative' if allow_zero else 'positive'))
    return x


def _shape_fit(data):
    voltage, current_ma = data.T
    x = np.column_stack([np.ones(len(data)), -1.0/voltage])
    y = np.log(current_ma) - 2.0*np.log(voltage)
    return np.linalg.lstsq(x, y, rcond=None)[0][1]


_B_FRESH = _shape_fit(FRESH_IV_V_MA)
_B_AGED = _shape_fit(AGED_IV_V_MA)
_LAST_SLOPE_V_H = (VOLTAGE_AT_2P5MA_V[-1]-VOLTAGE_AT_2P5MA_V[-5])/(AGE_H[-1]-AGE_H[-5])


@dataclass(frozen=True)
class EmpiricalCNTCathode:
    """Emitted I–V envelope, ageing and estimated cathode-only power."""
    power_factor: float = ASSUMED_POWER_FACTOR
    wear_exponent: float = ASSUMED_WEAR_EXPONENT

    def __post_init__(self):
        _positive(self.power_factor, 'power_factor')
        _positive(self.wear_exponent, 'wear_exponent')

    def nominal_voltage_V(self, equivalent_age_h):
        age = _positive(equivalent_age_h, 'equivalent_age_h', allow_zero=True)
        if age <= 500.0:
            from scipy.interpolate import PchipInterpolator
            return float(PchipInterpolator(AGE_H, VOLTAGE_AT_2P5MA_V, extrapolate=False)(age))
        return float(VOLTAGE_AT_2P5MA_V[-1] + _LAST_SLOPE_V_H*(age-500))

    def _fit_B(self, age_h):
        anchor = self.nominal_voltage_V(age_h)
        progress = np.clip((anchor-248.0)/(411.0-248.0), 0, 1)
        return _B_FRESH + progress*(_B_AGED-_B_FRESH)

    def emitted_current_A(self, voltage_V, equivalent_age_h=0.0):
        V = _positive(voltage_V, 'voltage_V')
        anchor = self.nominal_voltage_V(equivalent_age_h)
        exponent = 2*np.log(V/anchor) - self._fit_B(equivalent_age_h)*(1/V-1/anchor)
        return float(I_ANCHOR_A*np.exp(np.clip(exponent, -745, 700)))

    def required_voltage_V(self, target_current_A, equivalent_age_h=0.0):
        I = _positive(target_current_A, 'target_current_A', allow_zero=True)
        if I == 0:
            return 0.0
        from scipy.optimize import brentq
        upper = 1e4
        if self.emitted_current_A(upper, equivalent_age_h) < I:
            raise ValueError('target outside 10 kV numerical search boundary')
        return float(brentq(lambda V:self.emitted_current_A(V,equivalent_age_h)-I, 1, upper))

    def estimated_cathode_power_W(self, current_A, voltage_V):
        I = _positive(current_A, 'current_A', allow_zero=True)
        V = _positive(voltage_V, 'voltage_V')
        return I*V/self.power_factor  # DOES NOT include spacecraft converter losses

    def support_warnings(self, voltage_V, equivalent_age_h=0.0):
        """Approximate source-curve support only, not validated operating limits.

        Original source curves are from figures potentially different from the
        emitted-current anchor. The interpolated envelope is illustrative.
        """
        voltage = _positive(voltage_V, 'voltage_V')
        age = _positive(equivalent_age_h, 'equivalent_age_h', allow_zero=True)
        warnings = []
        if age > 500.0:
            warnings.append('Age beyond 500 h: voltage model is extrapolated.')
        p = np.clip((self.nominal_voltage_V(age)-248)/(411-248), 0, 1)
        lower = FRESH_IV_V_MA[0,0]*(1-p)+AGED_IV_V_MA[0,0]*p
        upper = FRESH_IV_V_MA[-1,0]*(1-p)+AGED_IV_V_MA[-1,0]*p
        if not lower <= voltage <= upper:
            warnings.append('Bias outside approximate digitized source I-V support envelope.')
        if self.emitted_current_A(voltage,age)>0.010:
            warnings.append('Estimated current exceeds 10 mA source-current range.')
        return warnings

    def equivalent_age_increment_h(self, elapsed_s, emitted_current_A):
        dt = _positive(elapsed_s, 'elapsed_s', allow_zero=True)
        current = _positive(emitted_current_A, 'emitted_current_A', allow_zero=True)
        return dt/3600.0*(current/I_ANCHOR_A)**self.wear_exponent
