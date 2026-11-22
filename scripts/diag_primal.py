"""Check the batched dual engine against a direct per-month primal computation (a few months, P=600)."""
import numpy as np
from voc import config, data, model

ds = data.us_dataset(last_month=config.PAPER_LAST)
rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)[[0, 100, 500, 1079]]
P = 600
f = model.voc_forecasts(ds, rows, seed=0, P_grid=(P,), z_grid=(1e-3, 1.0, 1000.0))
omega = model.draw_omega(0, max(config.P_GRID) // 2, ds.X.shape[1])  # same stream as P_GRID max
F = model.rff(ds.X, omega)[:, :P]
for i, m in enumerate(rows):
    S = F[m - 12:m]; s = F[m]; y = ds.R[m - 11:m + 1]
    sd = S.std(axis=0); S, s = S / sd, s / sd
    direct = [float(s @ model.ridge_primal(S, y, z)) for z in (1e-3, 1.0, 1000.0)]
    print(m, np.round(f[0, :, i], 6), np.round(direct, 6))
