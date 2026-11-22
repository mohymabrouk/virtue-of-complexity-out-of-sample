"""Test 3: replacing every data point after month t leaves every forecast made up to t unchanged
(predictor scaling, return scaling, VoC, linear ridge and simple rules)."""

import numpy as np
import pandas as pd
import pytest

from voc import config, data, model

P_SMALL = (2, 12, 60)


def _garble(df, t, seed=0):
    out = df.copy()
    after = out.index > t
    rng = np.random.default_rng(seed)
    if isinstance(out, pd.Series):
        out[after] = rng.standard_normal(after.sum()) * 0.3
    else:
        out.loc[after] = rng.standard_normal((after.sum(), out.shape[1])) * 5 + 3
    return out


def _all_forecasts(ds, rows, target=None):
    f = [model.voc_forecasts(ds, rows, 0, P_grid=P_SMALL).ravel(),
         model.linear_forecasts(ds, rows, target=target).ravel()]
    f += [v for v in model.simple_forecasts(ds, rows, target=target).values()]
    return np.concatenate(f)


def _check(ds_a, ds_b, t_idx):
    np.testing.assert_array_equal(ds_a.X[: t_idx + 1], ds_b.X[: t_idx + 1])
    np.testing.assert_array_equal(ds_a.R[: t_idx + 1], ds_b.R[: t_idx + 1])
    rows = ds_a.forecast_rows()
    rows = rows[rows <= t_idx]
    assert len(rows) > 5
    for target_a, target_b in ((None, None), (ds_a.r, ds_b.r)):
        np.testing.assert_array_equal(_all_forecasts(ds_a, rows, target_a), _all_forecasts(ds_b, rows, target_b))


def test_leak_synthetic(synthetic_panel):
    pred, r = synthetic_panel
    t = pred.index[150]
    a = data.make_dataset(pred, r)
    b = data.make_dataset(_garble(pred, t), _garble(r, t, 1))
    assert not np.allclose(a.X[151:], b.X[151:])  # the garbling did change the future
    _check(a, b, 150)


@pytest.mark.skipif(not (config.BUILT / "us_panel.parquet").exists(), reason="run `build` first")
def test_leak_us_panel():
    panel = data.load_us().loc[: pd.Period("1960-12", "M")]
    t = pd.Period("1958-06", "M")
    t_idx = panel.index.get_loc(t)
    a = data.us_dataset(panel)
    b = data.us_dataset(_garble(panel, t))
    _check(a, b, t_idx)
