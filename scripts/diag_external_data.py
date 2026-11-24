"""Paper sample, diagnostic only (PLAN.md, change 14): why does a public replication reach KMZ's Sharpe with a causal
return scaling when ours does not?

Reads the data file committed in rrikrv/voc-replication (commit 9c77321; data only, none of its code is executed,
the file is not copied into this repo) and runs our unchanged engine on its data, on ours, and on hybrids that swap
one block at a time. Writes results/external_data_diagnostic.json.

Usage: python scripts/diag_external_data.py path/to/voc_data.csv
"""
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from voc import config, data, metrics, model

SOURCE = {"repo": "https://github.com/rrikrv/voc-replication", "commit": "9c77321", "file": "data/voc_data.csv"}
N_SEEDS = 10
Z = (1e-3, 1000.0)

path = sys.argv[1]
raw_bytes = open(path, "rb").read()
ext = pd.read_csv(path)
ext.index = pd.PeriodIndex(ext["date"], freq="M")
ext = ext.rename(columns={"mkt_lag": "lag_mkt"})

ours = data.us_dataset(last_month=config.PAPER_LAST)
own = pd.DataFrame(ours.X, index=ours.months, columns=list(config.US_PREDICTORS))
own["R"], own["r"] = ours.R, ours.r

months = ext.index.intersection(own.index)
E, O = ext.loc[months], own.loc[months]
preds = list(config.US_PREDICTORS)

# 1. the data, column by column, on the common months
compare = {}
for c in preds:
    a, b = E[c].to_numpy(float), O[c].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    compare[c] = {"corr": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                  "mean_abs_ext": float(np.abs(a[ok]).mean()), "mean_abs_ours": float(np.abs(b[ok]).mean()),
                  "sd_ext": float(a[ok].std()), "sd_ours": float(b[ok].std())}
ok = E["R"].notna() & O["R"].notna()
compare["R (scaled return)"] = {"corr": float(np.corrcoef(E.loc[ok, "R"], O.loc[ok, "R"])[0, 1])}
compare["r (raw excess return)"] = {"corr": float(np.corrcoef(E["R_raw"], O["r"])[0, 1]),
                                    "ratio_sd_ours_to_ext": float(O["r"].std() / E["R_raw"].std())}
# the predictors' input scale is what sets the RFF bandwidth: mean squared norm of the 15-vector
compare["mean_sq_norm_of_predictor_vector"] = {"ext": float((E[preds] ** 2).sum(axis=1).mean()),
                                              "ours": float((O[preds] ** 2).sum(axis=1).mean())}


def build(X: pd.DataFrame, R: pd.Series, r: pd.Series, name: str) -> data.Dataset:
    return data.Dataset(name=name, months=months, X=X[preds].to_numpy(float), R=R.to_numpy(float),
                        r=r.to_numpy(float))


variants = {
    "ext data (its X, its R)": build(E, E["R"], E["R_raw"], "ext"),
    "our data (our X, our R)": build(O, O["R"], O["r"], "ours"),
    "our X, its R": build(O, E["R"], E["R_raw"], "oX_eR"),
    "its X, our R": build(E, O["R"], O["r"], "eX_oR"),
}


def run(ds: data.Dataset) -> dict:
    rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
    y = ds.R[rows + 1]
    per_seed = []
    for s in range(N_SEEDS):
        f = model.voc_forecasts(ds, rows, s, P_grid=(12000,), z_grid=Z)[0]
        st = metrics.batch_stats(f, y, y)
        per_seed.append([st["sharpe"], st["alpha_t"], st["ir"]])
    a = np.asarray(per_seed)  # (seeds, 3, len(Z))
    lin = metrics.batch_stats(model.linear_forecasts(ds, rows, z_grid=(0.0, 1000.0)), y, y)
    return {"n_months": int(len(rows)), "first_target": str(ds.months[rows[0] + 1]),
            "last_target": str(ds.months[rows[-1] + 1]), "bh_sharpe": metrics.sharpe(pd.Series(y), 12),
            "voc_z1e3": {"sharpe": a[:, 0, 1].mean(), "sharpe_seed_sd": a[:, 0, 1].std(ddof=1),
                         "alpha_t": a[:, 1, 1].mean(), "ir": a[:, 2, 1].mean()},
            "voc_ridgeless": {"sharpe": a[:, 0, 0].mean(), "alpha_t": a[:, 1, 0].mean(), "ir": a[:, 2, 0].mean()},
            "linear_ridgeless_sharpe": float(lin["sharpe"][0]),
            "linear_z1e3": {"sharpe": float(lin["sharpe"][1]), "alpha_t": float(lin["alpha_t"][1]),
                            "ir": float(lin["ir"][1])}}


out = {"source": {**SOURCE, "sha256": hashlib.sha256(raw_bytes).hexdigest(), "rows": int(len(ext)),
                  "first_month": str(ext.index[0]), "last_month": str(ext.index[-1])},
       "common_months": [str(months[0]), str(months[-1]), int(len(months))],
       "n_seeds": N_SEEDS, "compare": compare, "runs": {}}
for name, ds in variants.items():
    t0 = time.time()
    out["runs"][name] = run(ds)
    print(f"{name:28s} {time.time() - t0:5.0f}s", json.dumps(out["runs"][name], default=float), flush=True)
out["published"] = {"kmz_table1_nonlinear_z1e3": config.PAPER_TABLE1["nonlinear_z1e3"],
                    "kmz_table1_linear_z1e3": config.PAPER_TABLE1["linear_z1e3"],
                    "kmz_table1_linear_ridgeless": config.PAPER_TABLE1["linear_ridgeless"],
                    "rrikrv_readme_rff_T12": {"sharpe": 0.495, "ir": 0.33},
                    "rrikrv_readme_linear_z1e3_T12": {"sharpe": 0.46, "ir": 0.33}}
(config.RESULTS / "external_data_diagnostic.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(compare, indent=1, default=float))
