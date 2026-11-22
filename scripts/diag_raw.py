"""Nagel Table I setup: RFF ridgeless P=12000 on raw (unstandardised) returns; his alpha t 2.417, IR 0.255."""
import numpy as np, dataclasses
from voc import config, data, model, metrics

ds = data.us_dataset(last_month=config.PAPER_LAST)
rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
raw = dataclasses.replace(ds, R=ds.r.copy())
y = ds.r[rows + 1]
acc = []
for s in range(3):
    f = model.voc_forecasts(raw, rows, s, P_grid=(12000,), z_grid=(1e-3, 1000.0))[0]
    st = metrics.batch_stats(f, y, y)
    acc.append([st["sharpe"], st["alpha_t"], st["ir"]])
print("raw-return RFF P=12000 [z=1e-3, z=1e3]: SR, alpha_t, IR =", np.round(np.mean(acc, 0), 3).tolist())
