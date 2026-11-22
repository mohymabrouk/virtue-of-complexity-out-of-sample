import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel():
    """Persistent and fast predictors plus a return series, 220 consecutive months."""
    rng = np.random.default_rng(11)
    N, K = 220, 6
    months = pd.period_range("1980-01", periods=N, freq="M")
    slow = np.cumsum(rng.standard_normal((N, 3)) * 0.05, axis=0) + 1.0
    fast = rng.standard_normal((N, 3))
    pred = pd.DataFrame(np.hstack([slow, fast]), index=months, columns=[f"x{i}" for i in range(K)])
    r = pd.Series(rng.standard_normal(N) * 0.045 + 0.005, index=months)
    return pred, r
