"""Paper sample: Nagel's vol-timed momentum on raw returns (his Table I: alpha t 3.684, IR 0.388) and on R."""
import numpy as np
from voc import config, data, model, metrics

ds = data.us_dataset(last_month=config.PAPER_LAST)
rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
y = ds.R[rows + 1]; rr = ds.r[rows + 1]
lag = np.arange(12); w = (12 - lag) / 78
rw = ds.r[rows[:, None] - lag[None, :]]
Xw = ds.X[rows[:, None] + np.arange(-12, 0)[None, :]]
sig2 = Xw.var(axis=1).mean(axis=1)
pos_raw = (rw * w).sum(1) / sig2
print("Nagel volmom raw r, raw mkt:", metrics.alpha_regression(pos_raw * rr, rr))
sim = model.simple_forecasts(ds, rows)
for k, v in sim.items():
    a = metrics.alpha_regression(v * y, y)
    print(f"{k:14s} on R: SR {metrics.sharpe(__import__('pandas').Series(v*y), 12):.3f} alpha_t {a['alpha_t']:.2f} IR {a['ir']:.2f}")
print("BH raw SR", round(metrics.sharpe(__import__('pandas').Series(rr), 12), 3), " BH R SR", round(metrics.sharpe(__import__('pandas').Series(y), 12), 3))
