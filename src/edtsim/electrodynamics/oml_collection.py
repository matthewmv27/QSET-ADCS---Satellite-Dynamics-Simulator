"""Preliminary high-positive-bias OML upper-bound electron collection envelope.

This is NOT a closed electrical/plasma solver. It assumes uniform n_e and
positive tether-to-plasma bias (constant along the bare section), ignores
resistive voltage drop and motional EMF and requires a small collector cross
section relative to the Debye length. The emitted cathode current, circuit
load, plasma return and power balance must be solved separately.

Signed conventional current is positive from bus (s=0) to zenith tip (s=L).
Electrons collected near the tip flow inward; I(L)=0 for distributed bare
collection with no external tip terminal current. Then dI/ds=-j_collection.
"""
from dataclasses import dataclass
import numpy as np
from edtsim.tether.geometry import HomogeneousTether, RoundWire, RectangularTape

E_CHARGE_C = 1.602176634e-19
E_MASS_KG = 9.1093837139e-31
EPSILON0_F_M = 8.8541878128e-12
K_B_J_K = 1.380649e-23


def _positive(x, name):
    n = float(x)
    if not np.isfinite(n) or n <= 0:
        raise ValueError(f'{name} must be positive finite')
    return n


@dataclass(frozen=True)
class ThinCollectorOMLEnvelope:
    """Upper-bound *collection* profile, not a solved circuit current.

    Uses high-bias cylindrical OML j_e = (p/pi) * e*n_e * sqrt(2*e*V/m_e)
    [A/m], appropriate only when eV >> kT_e and geometrical OML regime.
    Geometry equivalent radius condition is a warning/guard, not proof of OML.
    """
    tether: HomogeneousTether
    electron_density_m3: float
    electron_temperature_K: float
    positive_bias_V: float

    def __post_init__(self):
        if not isinstance(self.tether,HomogeneousTether):
            raise TypeError('tether must be HomogeneousTether')
        for name in ('electron_density_m3','electron_temperature_K','positive_bias_V'):
            _positive(getattr(self,name),name)
        if self.bias_over_thermal < 10:
            raise ValueError('high-bias OML approximation requires eV/(kT) >= 10')

    @property
    def bias_over_thermal(self):
        return E_CHARGE_C*self.positive_bias_V/(K_B_J_K*self.electron_temperature_K)

    @property
    def debye_length_m(self):
        return np.sqrt(EPSILON0_F_M*K_B_J_K*self.electron_temperature_K/
                       (self.electron_density_m3*E_CHARGE_C**2))

    @property
    def equivalent_radius_m(self):
        cross = self.tether.cross_section
        return cross.diameter_m/2 if isinstance(cross,RoundWire) else cross.width_m/4

    @property
    def geometry_ratio(self):
        return float(self.equivalent_radius_m/self.debye_length_m)

    @property
    def collected_current_density_A_m(self):
        p=self.tether.collecting_perimeter_m
        return float((p/np.pi)*E_CHARGE_C*self.electron_density_m3*
                     np.sqrt(2*E_CHARGE_C*self.positive_bias_V/E_MASS_KG))

    @property
    def bus_current_upper_bound_A(self):
        return self.collected_current_density_A_m*self.tether.bare_length_m

    def segment_currents_A(self, t_s, midpoints_s_m):
        """Analytical upper-bound I(s) with uniformly biased bare tip section.

        This profile may not be physically reachable by the proposed CNT and
        should never be silently used as an actual electrical circuit solution.
        """
        if not np.isfinite(float(t_s)):
            raise ValueError('invalid time')
        s=np.asarray(midpoints_s_m,dtype=float)
        if s.ndim!=1 or len(s)==0 or not np.all(np.isfinite(s)) or np.any(s<0) or np.any(s>self.tether.length_m):
            raise ValueError('midpoints must lie within [0,L]')
        start=self.tether.length_m-self.tether.bare_length_m
        return self.collected_current_density_A_m*np.maximum(0,self.tether.length_m-np.maximum(s,start))

    def cathode_requirement(self, cathode, *, equivalent_age_h=0.0, max_bias_V=None):
        """Preliminary cathode demand check, NOT coupled circuit closure."""
        current = self.bus_current_upper_bound_A
        required = cathode.required_voltage_V(current, equivalent_age_h)
        feasible = max_bias_V is None or required <= float(max_bias_V)
        return {'collection_upper_bound_A':float(current), 'cathode_voltage_required_V':float(required),
                'meets_user_voltage_limit':bool(feasible), 'ratio_R_equiv_to_Debye':self.geometry_ratio,
                'model':'decoupled high-bias OML collection envelope; not circuit current'}


@dataclass(frozen=True)
class OMLCircuitResult:
    """Current profile for an imposed bus-to-plasma potential, not an EDT loop solution."""
    s_m: np.ndarray
    conventional_current_A: np.ndarray
    tether_plasma_bias_V: np.ndarray
    collected_electron_current_per_m_A_m: np.ndarray
    bus_emission_current_required_A: float
    tip_current_residual_A: float
    ohmic_loss_W: float
    collection_model: str


@dataclass(frozen=True)
class PrescribedBusPotentialOML:
    """1-D electron-collection + Ohmic-drop shooting with fixed plasma properties.

    dI/ds = -j_e(V), dV/ds = E_m - I/(sigma*A).
    Boundary conditions: V(0)=fixed conductor-to-plasma bias;
                         I(L)=0 (no conducting current beyond tip).
    E_m is an imposed, scalar motional-field projection along +bus->tip.
    The CNT cathode I-V is not fed back into V(0), and the plasma return
    potential, ion collection, contact impedance, active power electronics,
    space charge and magnetic effects are omitted. Results require a separate
    cathode feasibility and power closure analysis.
    """
    tether: HomogeneousTether
    electron_density_m3: float
    electron_temperature_K: float
    bus_plasma_bias_V: float
    motional_E_along_V_m: float

    def __post_init__(self):
        if not isinstance(self.tether, HomogeneousTether):
            raise TypeError('tether must be HomogeneousTether')
        for name in ('electron_density_m3','electron_temperature_K'):
            _positive(getattr(self,name),name)
        for name in ('bus_plasma_bias_V','motional_E_along_V_m'):
            v = float(getattr(self,name))
            if not np.isfinite(v):
                raise ValueError(name+' must be finite')

    @property
    def electron_thermal_flux_A_m(self):
        return (E_CHARGE_C*self.electron_density_m3*self.tether.collecting_perimeter_m *
            np.sqrt(K_B_J_K*self.electron_temperature_K/(2*np.pi*E_MASS_KG)))

    def collected_density_A_m(self, s_m, bias_V):
        """Long, thin cylindrical OML approximation; thermal+positive-bias form.

        Positive voltage: thermal current times [erfcx(sqrt(eta)) +
        2/sqrt(pi)*sqrt(eta)]. Negative: Boltzmann electron suppression.
        Tape uses perimeter-based OML upper-bound only; sheath geometry and
        width/Debye limits are NOT validated in this function.
        """
        s = float(s_m)
        if s < self.tether.length_m-self.tether.bare_length_m or s>self.tether.length_m:
            return 0.0
        from scipy.special import erfcx
        eta=E_CHARGE_C*float(bias_V)/(K_B_J_K*self.electron_temperature_K)
        if eta >= 0:
            factor=erfcx(np.sqrt(eta))+2/np.sqrt(np.pi)*np.sqrt(eta)
        else:
            factor=np.exp(max(eta,-700))
        return float(self.electron_thermal_flux_A_m*factor)

    def solve(self, nodes=51, rtol=1e-8, atol=1e-11):
        from scipy.integrate import solve_ivp
        from scipy.optimize import brentq
        L=self.tether.length_m
        if type(nodes) is not int or not 3<=nodes<=10001:
            raise ValueError('nodes must be an integer in [3,10001]')
        Rprime=1/(self.tether.conductivity_S_m*self.tether.cross_section.area_m2)
        def rhs(s,y):
            I,V=y
            return [-self.collected_density_A_m(s,V), self.motional_E_along_V_m - Rprime*I]
        # Break integration at the insulating/bare transition to avoid a
        # discontinuous RHS inside the stiff/adaptive step selection.
        start=L-self.tether.bare_length_m
        def integrate(I0, *, dense=False):
            intervals=[(0.,L)] if start<=0 or start>=L else [(0.,start),(start,L)]
            y=np.array([I0,self.bus_plasma_bias_V])
            solutions=[]
            for a,b in intervals:
                sol=solve_ivp(rhs,(a,b),y,method='DOP853',rtol=rtol,atol=atol,dense_output=True,
                              max_step=max((b-a)/20,1e-4))
                if not sol.success or not np.all(np.isfinite(sol.y)):
                    raise RuntimeError('OML shooting IVP failed: '+sol.message)
                y=sol.y[:,-1]
                solutions.append((a,b,sol.sol))
            if dense:
                return y,solutions
            return float(y[0])
        if self.tether.bare_length_m==0:
            s=np.linspace(0,L,nodes)
            V=self.bus_plasma_bias_V+self.motional_E_along_V_m*s
            return OMLCircuitResult(s,np.zeros_like(s),V,np.zeros_like(s),0.,0.,0.,
                    'prescribed bus bias, zero bare collection')
        lo=0.
        # Upper bound with no Ohmic drop: collection cannot exceed the
        # maximum uniform-bias collection along L if E_m is fixed.
        max_bias=max(self.bus_plasma_bias_V,
                     self.bus_plasma_bias_V+self.motional_E_along_V_m*L)
        upper_density=self.collected_density_A_m(L,max_bias)
        hi=max(upper_density*L*2.,self.electron_thermal_flux_A_m*L*1e-3,1e-12)
        while integrate(hi)<0:
            hi*=2
            if hi>1e6:
                raise RuntimeError('cannot bracket OML bus current')
        I0=brentq(integrate,lo,hi,xtol=1e-14,rtol=1e-10)
        end,solutions=integrate(I0,dense=True)
        s=np.linspace(0,L,nodes)
        vals=np.empty((2,nodes))
        for k,sk in enumerate(s):
            for a,b,interp in solutions:
                if a<=sk<=b:
                    vals[:,k]=interp(sk)
                    break
        current=np.maximum(0,vals[0,:])
        bias=vals[1,:]
        collection=np.array([self.collected_density_A_m(sk,v) for sk,v in zip(s,bias)])
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        loss = float(integrate(current**2 * Rprime, s))
        return OMLCircuitResult(s,current,bias,collection,float(I0),float(end[0]),loss,
                'cylindrical OML approximation, prescribed bus bias & EMF; not plasma circuit closure')
