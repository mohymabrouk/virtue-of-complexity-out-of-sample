"""Test 5: the copied Sharpe and deflated Sharpe equal Alpha Audit's functions on fixed inputs."""

import numpy as np
import pandas as pd
import pytest

from voc import metrics

import alpha_audit_ref

# Reference values computed on 2026-11-21 with Alpha Audit's own functions on the series below.
REF = {"sharpe_12": 0.10582360429431814, "sharpe_252": 0.4849446769917668, "dsr_1": 0.6790571657292245,
       "dsr_153": 0.017334116195734883, "dsr_20": 0.007580288002012714}


def _series():
    rng = np.random.default_rng(123)
    return pd.Series(rng.standard_t(5, 240) * 0.04 + 0.006)


def test_against_reference_values():
    x = _series()
    assert metrics.sharpe(x, 12) == pytest.approx(REF["sharpe_12"], rel=1e-12)
    assert metrics.sharpe(x, 252) == pytest.approx(REF["sharpe_252"], rel=1e-12)
    assert metrics.deflated_sharpe(x, 1, 0.0) == pytest.approx(REF["dsr_1"], rel=1e-12)
    assert metrics.deflated_sharpe(x, 153, 0.004) == pytest.approx(REF["dsr_153"], rel=1e-10)
    assert metrics.deflated_sharpe(x, 20, 0.01) == pytest.approx(REF["dsr_20"], rel=1e-10)


@pytest.mark.skipif(alpha_audit_ref.load() is None, reason="Alpha Audit source not on this machine")
def test_against_alpha_audit_source():
    sharpe_aa, dsr_aa = alpha_audit_ref.load()
    rng = np.random.default_rng(5)
    for _ in range(5):
        x = pd.Series(rng.standard_normal(rng.integers(30, 400)) * 0.05 + rng.uniform(-0.01, 0.02))
        assert metrics.sharpe(x, 12) == sharpe_aa(x, 12)
        for trials, var in ((1, 0.0), (10, 0.002), (153, 0.01)):
            assert metrics.deflated_sharpe(x, trials, var) == dsr_aa(x, trials, var)
    assert metrics.sharpe(pd.Series([0.01]), 12) == sharpe_aa(pd.Series([0.01]), 12)


def test_sharpe_se_and_bootstrap_sane():
    x = _series().to_numpy()
    se = metrics.sharpe_se(x)
    assert 0.0 < se < 1.0
    rng = np.random.default_rng(0)
    b = rng.standard_normal(240) * 0.04 + 0.004
    d, lo, hi = metrics.block_bootstrap_dsharpe(x, b, reps=500)
    assert lo <= d <= hi
