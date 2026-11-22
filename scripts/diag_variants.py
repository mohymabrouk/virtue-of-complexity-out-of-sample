"""Diagnostics on the paper sample only: how much do the documented ambiguities move the target cell?"""
import numpy as np
from voc import config, data, model, metrics

def grams_uncentered(F, rows, P_grid, T=config.T, chunk=config.FEATURE_CHUNK):
    n = len(rows); out = np.empty((len(P_grid), n, T + 1, T + 1)); G = np.zeros((n, T + 1, T + 1))
    idx = np.asarray(rows)[:, None] + np.arange(-T, 1)[None, :]; start = 0
    for k, P in enumerate(P_grid):
        while start < P:
            stop = min(P, start + chunk); W = F[:, start:stop][idx]
            sd = np.sqrt((W[:, :T, :] ** 2).mean(axis=1, keepdims=True)); W /= sd
            G += W @ W.transpose(0, 2, 1); start = stop
        out[k] = G
    return out

PG = (12, 120, 1200, 12000); ZG = (1e-3, 1.0, 1000.0)
def run(ds, label, seeds=3):
    rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST); y = ds.R[rows + 1]
    acc = []
    for s in range(seeds):
        f = model.voc_forecasts(ds, rows, s, P_grid=PG, z_grid=ZG)
        st = metrics.batch_stats(f.reshape(-1, len(rows)), y, y)
        acc.append(np.stack([st["sharpe"], st["alpha_t"], st["ir"], st["r2"]]))
    a = np.mean(acc, axis=0).reshape(4, len(PG), len(ZG))
    lin = metrics.batch_stats(model.linear_forecasts(ds, rows, z_grid=(0.0, 1000.0)), y, y)
    print(f"{label:28s} SR  P12k: z1e-3 {a[0,3,0]:.3f} z1 {a[0,3,1]:.3f} z1e3 {a[0,3,2]:.3f} | P1200 z1e3 {a[0,2,2]:.3f} | P12 z1e3 {a[0,0,2]:.3f}"
          f" | t(z1e-3,1e3) {a[1,3,0]:.2f},{a[1,3,2]:.2f} IR {a[2,3,0]:.2f},{a[2,3,2]:.2f} R2 {a[3,3,2]:.4f}"
          f" | lin SR 0+ {lin['sharpe'][0]:.3f} 1e3 {lin['sharpe'][1]:.3f} t {lin['alpha_t'][1]:.2f} R2 {lin['r2'][1]:.3f}")

panel = data.load_us()
run(data.us_dataset(panel, config.PAPER_LAST), "A as planned")
orig = model.window_grams
model.window_grams = grams_uncentered
run(data.us_dataset(panel, config.PAPER_LAST), "B uncentered RFF scaling")
model.window_grams = orig
p2 = panel.copy(); g = data.read_goyal(); p2["infl"] = g["infl"].shift(-1).reindex(p2.index)  # undo the extra lag
run(data.us_dataset(p2, config.PAPER_LAST), "C infl not lagged")
