"""Thin slice: US paper sample, one seed, full grid; timing and the target cell's Sharpe."""
import time
import numpy as np
from voc import config, data, model, metrics

ds = data.us_dataset(last_month=config.PAPER_LAST)
rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
print("rows", len(rows), ds.target_months(rows)[0], ds.target_months(rows)[-1])
y = ds.R[rows + 1]
for seed in range(3):
    t0 = time.time()
    f = model.voc_forecasts(ds, rows, seed)
    t1 = time.time()
    st = metrics.batch_stats(f.reshape(-1, len(rows)), y, y)
    k = config.P_GRID.index(12000); j = config.Z_GRID.index(1000.0)
    print(f"seed {seed}: {t1-t0:.2f}s  SR(P=12000,z=1e3)={st['sharpe'].reshape(len(config.P_GRID), -1)[k, j]:.3f}  "
          f"R2={st['r2'].reshape(len(config.P_GRID), -1)[k, j]:.4f} alpha_t={st['alpha_t'].reshape(len(config.P_GRID), -1)[k, j]:.2f}")
lin = model.linear_forecasts(ds, rows)
st = metrics.batch_stats(lin, y, y)
print("linear z=0+,1e-3..1e3 SR", np.round(st["sharpe"], 3), "R2", np.round(st["r2"], 3))
sim = model.simple_forecasts(ds, rows)
for k, v in sim.items():
    print(k, round(float(metrics.batch_stats(v, y, y)["sharpe"][0]), 3))
print("BH", round(float(metrics.batch_stats(np.ones_like(y), y, y)["sharpe"][0]), 3))
