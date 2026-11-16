"""Real third-party model smoke tests. Auto-skip unless optional extras installed.

Run with: pip install -e '.[dev,dynamics,earth-models]'
          python -m pytest tests/test_m4_optional_model_integration.py -rs
"""
from importlib.util import find_spec
import numpy as np
import pytest
from edtsim.environment.earth_models import GeomagneticMainField, MSIS2AtmosphericDensity
from edtsim.core.constants import R_EARTH_EQ


@pytest.mark.skipif(not (find_spec('astropy') and find_spec('ppigrf')),
                    reason='requires astropy and ppigrf optional earth models')
def test_igrf14_real_library_gcrs_field_at_500km():
    provider=GeomagneticMainField('2026-11-15T00:00:00Z','igrf14')
    B=provider.sample(0,[R_EARTH_EQ+500000,0.,0.])
    assert B.shape==(3,)
    assert 1e-6 < np.linalg.norm(B) < 1e-4


@pytest.mark.skipif(not (find_spec('astropy') and find_spec('wmm')),
                    reason='requires astropy and wmm-calculator optional earth models')
def test_wmm2025_real_library_gcrs_field_at_500km():
    provider=GeomagneticMainField('2026-11-15T00:00:00Z','wmm2025')
    B=provider.sample(0,[R_EARTH_EQ+500000,0.,0.])
    assert B.shape==(3,)
    assert 1e-6 < np.linalg.norm(B) < 1e-4


@pytest.mark.skipif(not (find_spec('astropy') and find_spec('pymsis')),
                    reason='requires astropy and pymsis optional earth models')
def test_msis2_real_library_at_500km():
    provider=MSIS2AtmosphericDensity('2026-11-15T00:00:00Z',120,120,7)
    density=provider.sample(0,[R_EARTH_EQ+500000,0.,0.])
    assert np.isfinite(density) and 1e-18 < density < 1e-7
