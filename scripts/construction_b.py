"""Construction B (PLAN.md, change 16): the data choices of a public replication that reproduces KMZ's Table 1,
rebuilt from our own raw files, then scored once after the paper's sample.

Construction A (ours, frozen) vs B, which differs in four data steps only:
  (i)   predictors divided by their expanding std from their first month in the Goyal file (A: from 1926-07);
  (ii)  inflation not lagged an extra month;
  (iii) market return and lagged-market predictor = Goyal `ret - Rfree` (A: Ken French Mkt-RF);
  (iv)  returns divided by the uncentred RMS of the previous 12 months (A: their std).
The engine (RFF, ridge, windows) is the frozen one. Writes results/construction_b.json and
results/construction_b_post_positions.csv.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd

from voc import config, data, metrics, model

P, Z = (12000,), (1e-3, 1000.0)
POST_FIRST, POST_LAST = "2021-01", "2025-12"
N_DECOMP, N_MAIN = 10, 300


def goyal_raw() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = pd.read_excel(config.GOYAL_FILE, sheet_name="Monthly")
    g.index = pd.PeriodIndex([pd.Period(f"{int(v) // 100}-{int(v) % 100:02d}", "M") for v in g["yyyymm"]])
    return g


G_RAW = goyal_raw()
G_PRED = data.read_goyal()  # our 14 Goyal predictors (inflation lagged one extra month)
FRENCH = data.read_french()


def build(full_history_std: bool, infl_unlagged: bool, goyal_ret: bool, rms: bool, name: str) -> data.Dataset:
    gp = G_PRED.copy()
    if infl_unlagged:
        gp["infl"] = G_RAW["infl"].astype(float)
    r = (G_RAW["ret"] - G_RAW["Rfree"]).astype(float).dropna() if goyal_ret else FRENCH["mkt_rf"]
    months = pd.period_range(r.index.min(), r.index.max(), freq="M")
    preds = list(config.US_PREDICTORS)
    if full_history_std:
        allm = pd.period_range(min(gp.index.min(), months.min()), months.max(), freq="M")
        raw = gp.reindex(allm)
        raw["lag_mkt"] = r.reindex(allm)
        X = data.scale_predictors(raw[preds]).reindex(months)
    else:
        raw = gp.reindex(months)
        raw["lag_mkt"] = r.reindex(months)
        X = data.scale_predictors(raw[preds])
    last_goyal = G_PRED.dropna(how="all").index.max()
    X.loc[X.index > last_goyal, :] = np.nan
    rr = r.reindex(months)
    if rms:
        R = rr / np.sqrt((rr ** 2).rolling(config.RET_VOL_WINDOW).mean()).shift(1)
    else:
        R = data.scale_returns(rr)
    return data.Dataset(name=name, months=months, X=X.to_numpy(float), R=R.to_numpy(float), r=rr.to_numpy(float))


def stats(F: np.ndarray, y: np.ndarray) -> dict:
    st = metrics.batch_stats(F, y, y)
    return {k: st[k] for k in ("sharpe", "alpha_t", "ir")}


# ---- 0. sanity: the builder with every switch off is our dataset --------------------------------------------------
A = build(False, False, False, False, "A")
ref = data.us_dataset()
common = A.months.intersection(ref.months)
ia, ib = A.months.get_indexer(common), ref.months.get_indexer(common)
same_X = np.allclose(A.X[ia], ref.X[ib], equal_nan=True)
same_R = np.allclose(A.R[ia], ref.R[ib], equal_nan=True)
print("builder(A) == data.us_dataset():", same_X, same_R, flush=True)
assert same_X and same_R

# ---- 1. cumulative decomposition on the paper sample, 10 draws ---------------------------------------------------
steps = [("A (ours, frozen)", (False, False, False, False)),
         ("+ (i) predictor std from first Goyal month", (True, False, False, False)),
         ("+ (ii) inflation not lagged", (True, True, False, False)),
         ("+ (iii) Goyal S&P 500 return", (True, True, True, False)),
         ("+ (iv) RMS return scaling = B", (True, True, True, True))]
decomp = {}
for label, sw in steps:
    t0 = time.time()
    ds = build(*sw, name=label)
    rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
    y = ds.R[rows + 1]
    per = np.array([[v for v in stats(model.voc_forecasts(ds, rows, s, P_grid=P, z_grid=Z)[0], y).values()]
                    for s in range(N_DECOMP)])  # (seeds, 3, len(Z))
    lin = stats(model.linear_forecasts(ds, rows, z_grid=(0.0, 1000.0)), y)
    decomp[label] = {"n_months": int(len(rows)), "bh_sharpe": metrics.sharpe(pd.Series(y), 12),
                     "voc_z1e3_sharpe": per[:, 0, 1].mean(), "voc_z1e3_sharpe_seed_sd": per[:, 0, 1].std(ddof=1),
                     "voc_z1e3_alpha_t": per[:, 1, 1].mean(), "voc_z1e3_ir": per[:, 2, 1].mean(),
                     "voc_ridgeless_sharpe": per[:, 0, 0].mean(),
                     "linear_ridgeless_sharpe": float(lin["sharpe"][0]),
                     "linear_z1e3_sharpe": float(lin["sharpe"][1]), "linear_z1e3_alpha_t": float(lin["alpha_t"][1]),
                     "linear_z1e3_ir": float(lin["ir"][1])}
    print(f"{label:45s} {time.time() - t0:4.0f}s", json.dumps(decomp[label], default=float), flush=True)


# ---- 2. A and B with draws 0-299: paper sample + one post-sample score (2021-01..2025-12) ------------------------
def main_run(ds: data.Dataset) -> tuple[dict, pd.DataFrame]:
    rows = ds.forecast_rows(config.PAPER_FIRST, POST_LAST)
    tgt = ds.months[rows + 1]
    periods = {"paper": (tgt >= pd.Period(config.PAPER_FIRST, "M")) & (tgt <= pd.Period(config.PAPER_LAST, "M")),
               "post": (tgt >= pd.Period(POST_FIRST, "M")) & (tgt <= pd.Period(POST_LAST, "M"))}
    y_all = ds.R[rows + 1]
    fsum = np.zeros((len(Z), len(rows)))
    per = {k: [] for k in periods}
    t0 = time.time()
    for s in range(N_MAIN):
        f = model.voc_forecasts(ds, rows, s, P_grid=P, z_grid=Z)[0]
        fsum += f
        for k, m in periods.items():
            per[k].append([v for v in stats(f[:, m], y_all[m]).values()])
        if s % 50 == 49:
            print(f"  {ds.name}: {s + 1} draws, {time.time() - t0:.0f}s", flush=True)
    ens = fsum / N_MAIN
    lin = model.linear_forecasts(ds, rows, z_grid=(0.0, 1000.0))
    out = {"n_draws": N_MAIN}
    for k, m in periods.items():
        y = y_all[m]
        a = np.array(per[k])
        e = stats(ens[:, m], y)
        ls = stats(lin[:, m], y)
        ens_ret = ens[1, m] * y
        out[k] = {"first_target": str(tgt[m][0]), "last_target": str(tgt[m][-1]), "n_months": int(m.sum()),
                  "bh_sharpe": metrics.sharpe(pd.Series(y), 12), "bh_sharpe_se": metrics.sharpe_se(y, 12),
                  "voc_z1e3_per_draw": {"sharpe": a[:, 0, 1].mean(), "sharpe_seed_sd": a[:, 0, 1].std(ddof=1),
                                        "alpha_t": a[:, 1, 1].mean(), "ir": a[:, 2, 1].mean()},
                  "voc_z1e3_ensemble": {"sharpe": float(e["sharpe"][1]), "sharpe_se": metrics.sharpe_se(ens_ret, 12),
                                        "alpha_t": float(e["alpha_t"][1]), "ir": float(e["ir"][1]),
                                        "months_long": int((ens[1, m] > 0).sum())},
                  "voc_ridgeless_ensemble_sharpe": float(e["sharpe"][0]),
                  "linear_z1e3": {"sharpe": float(ls["sharpe"][1]), "alpha_t": float(ls["alpha_t"][1])},
                  "linear_ridgeless_sharpe": float(ls["sharpe"][0])}
    pos = pd.DataFrame({"target_month": tgt.astype(str), f"{ds.name}_position": ens[1], f"{ds.name}_R": y_all})
    return out, pos


B = build(True, True, True, True, "B")
res, positions = {}, []
for ds in (A, B):
    o, p = main_run(ds)
    res[ds.name] = o
    positions.append(p.set_index("target_month"))
    print(ds.name, json.dumps(o, default=float), flush=True)

out = {"protocol": "PLAN.md changes 15-16; post-sample scored once", "engine": {"P": P[0], "z": list(Z), "T": config.T,
       "gamma": config.GAMMA}, "builder_reproduces_A": bool(same_X and same_R), "decomposition_paper_sample": decomp,
       "A": res["A"], "B": res["B"],
       "published": {"kmz_table1_nonlinear_z1e3": config.PAPER_TABLE1["nonlinear_z1e3"],
                     "kmz_table1_linear_z1e3": config.PAPER_TABLE1["linear_z1e3"],
                     "kmz_table1_linear_ridgeless": config.PAPER_TABLE1["linear_ridgeless"]},
       "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
(config.RESULTS / "construction_b.json").write_text(json.dumps(out, indent=2, default=float))
pd.concat(positions, axis=1).loc[lambda d: d.index >= POST_FIRST].to_csv(
    config.RESULTS / "construction_b_post_positions.csv", float_format="%.6f")
print("done")
