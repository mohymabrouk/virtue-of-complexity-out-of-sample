"""Test 4: the timing-strategy return (position at t times the return of t+1) on toy series."""

import math

import numpy as np
import pandas as pd
import pytest

from voc import data, metrics, model, pipeline


def test_strategy_return_toy():
    pos = np.array([1.0, -1.0, 0.5, 2.0])
    y = np.array([0.1, 0.2, -0.3, 0.05])
    ret = np.array([0.1, -0.2, -0.15, 0.1])
    st = metrics.batch_stats(pos[None, :], y, y)
    assert st["mean"][0] == pytest.approx(ret.mean() * 12)
    assert st["sharpe"][0] == pytest.approx(ret.mean() / ret.std(ddof=1) * math.sqrt(12))
    assert st["r2"][0] == pytest.approx(1 - np.sum((y - pos) ** 2) / np.sum(y**2))
    full = metrics.full_stats(pos, y, y, trials=1, trial_var=0.0, bootstrap=False)
    assert full["mean_ann"] == pytest.approx(ret.mean() * 12)
    assert full["sharpe"] == pytest.approx(st["sharpe"][0])


def test_alignment_position_at_m_earns_return_of_m_plus_1(synthetic_panel):
    pred, r = synthetic_panel
    ds = data.make_dataset(pred, r)
    rows = ds.forecast_rows()
    assert (ds.target_months(rows) == ds.months[rows + 1]).all()
    res = pipeline.run_dataset(ds, rows, 1, {"all": np.ones(len(rows), bool)})
    np.testing.assert_array_equal(res["conv"]["std"]["y"], ds.R[rows + 1])
    np.testing.assert_array_equal(res["conv"]["raw"]["y"], ds.r[rows + 1])
    mom = np.array([ds.R[m - 11:m + 1].mean() for m in rows])
    np.testing.assert_allclose(res["conv"]["std"]["simple"]["mom12"], mom)
    tab = pipeline.evaluate(res["conv"]["std"], np.ones(len(rows), bool), names=["Mom12 (equal weights)"], bootstrap=False)
    assert tab["mean_ann"].iloc[0] == pytest.approx(np.mean(mom * ds.R[rows + 1]) * 12)


def test_tsmom_sign_and_hist_mean_use_past_only(synthetic_panel):
    pred, r = synthetic_panel
    ds = data.make_dataset(pred, r)
    rows = ds.forecast_rows()
    sim = model.simple_forecasts(ds, rows)
    m = rows[10]
    assert sim["tsmom_sign"][10] == np.sign(ds.R[m - 11:m + 1].sum())
    valid = ds.R[: m + 1][np.isfinite(ds.R[: m + 1])]
    assert sim["hist_mean"][10] == pytest.approx(valid.mean())
    assert pd.Series(ds.R).notna().sum() > 0
