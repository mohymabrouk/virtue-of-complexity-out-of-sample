"""Replication diagnostics on the paper sample (1931-2020 only), saved to results/replication_diagnostics.json.

Same checks as the exploratory diag_*.py scripts (see PLAN.md, changes log item 3), run once more so that the
README can quote them from a results file. 3 RFF draws per variant; P = 12,000 and z in {1e-3, 1e3}.
"""
import dataclasses
import json

import numpy as np
import pandas as pd

from voc import config, data, metrics, model


def voc_stats(ds, rows, seeds=3):
    y = ds.R[rows + 1]
    acc = []
    for s in range(seeds):
        f = model.voc_forecasts(ds, rows, s, P_grid=(12000,), z_grid=(1e-3, 1000.0))[0]
        st = metrics.batch_stats(f, y, y)
        acc.append([st["sharpe"], st["alpha_t"], st["ir"]])
    a = np.mean(acc, axis=0)
    lin = metrics.batch_stats(model.linear_forecasts(ds, rows, z_grid=(0.0, 1000.0)), y, y)
    return {"bh_sharpe": metrics.sharpe(pd.Series(y), 12),
            "voc_ridgeless": {"sharpe": a[0, 0], "alpha_t": a[1, 0], "ir": a[2, 0]},
            "voc_z1e3": {"sharpe": a[0, 1], "alpha_t": a[1, 1], "ir": a[2, 1]},
            "linear_ridgeless_sharpe": float(lin["sharpe"][0]),
            "linear_z1e3": {"sharpe": float(lin["sharpe"][1]), "alpha_t": float(lin["alpha_t"][1])}}


ds = data.us_dataset(last_month=config.PAPER_LAST)
rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
r = pd.Series(ds.r)
variants = {
    "std_trailing_t-11..t (planned)": r / r.rolling(12).std().shift(1),
    "std_trailing_t-12..t-1": r / r.rolling(12).std().shift(2),
    "rms_trailing_t-11..t": r / np.sqrt((r**2).rolling(12).mean()).shift(1),
    "std_window_includes_forecast_month (look-ahead)": r / r.rolling(12).std(),
    "raw (no standardisation)": r,
}
out = {}
for name, R in variants.items():
    out[name] = voc_stats(dataclasses.replace(ds, R=R.to_numpy()), rows)
    print(name, json.dumps(out[name], default=float))
# Nagel eq. 14 on raw returns with the raw market as factor (his Table I: alpha t 3.684, IR 0.388)
lag = np.arange(12)
rw = ds.r[rows[:, None] - lag[None, :]]
sig2 = ds.X[rows[:, None] + np.arange(-12, 0)[None, :]].var(axis=1).mean(axis=1)
pos = (rw * (12 - lag) / 78).sum(1) / sig2
yy = ds.r[rows + 1]
out["nagel_volmom_raw"] = metrics.alpha_regression(pos * yy, yy)
out["published"] = {"kmz_table1_nonlinear_z1e3": config.PAPER_TABLE1["nonlinear_z1e3"],
                    "kmz_table1_linear_z1e3": config.PAPER_TABLE1["linear_z1e3"],
                    "kmz_table1_linear_ridgeless": config.PAPER_TABLE1["linear_ridgeless"],
                    "nagel_table1_rff_raw_ridgeless": {"alpha_t": 2.417, "ir": 0.255},
                    "nagel_table1_rff_std_ridgeless": {"alpha_t": 2.811, "ir": 0.296},
                    "nagel_table1_volmom_raw": {"alpha_t": 3.684, "ir": 0.388}}
(config.RESULTS / "replication_diagnostics.json").write_text(json.dumps(out, indent=2, default=float))
