"""Weights on the 12 training returns implied by the P=12000 ridgeless RFF forecast (compare Nagel 2025, Fig. I)."""
import numpy as np
from voc import config, data, model

ds = data.us_dataset(last_month=config.PAPER_LAST)
rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
omega = model.draw_omega(0, 6000, ds.X.shape[1])
F = np.full((len(ds.months), 12000), np.nan); lo = rows.min() - 12
F[lo:] = model.rff(ds.X[lo:], omega)
G = model.window_grams(F, rows, (12000,))[0]
K = G[:, :12, :12]; k = G[:, 12, :12]
w = np.linalg.solve(K + 1e-3 * 12 * np.eye(12)[None], k[..., None])[..., 0]  # (n, 12), column j = training row m-12+j
w_lag = w[:, ::-1]  # lag 1 = most recent return R[m]
print("mean weight by lag 1..12:", np.round(w_lag.mean(0), 3))
print("p10:", np.round(np.percentile(w_lag, 10, 0), 3)); print("p90:", np.round(np.percentile(w_lag, 90, 0), 3))
print("mean of weights over time, by decade:")
dec = ds.target_months(rows).year // 10 * 10
for d in np.unique(dec):
    print(d, round(float(w[dec == d].mean()), 4))
# month-to-month change of the scaled predictors
X = ds.X[rows]; dX = np.diff(ds.X, axis=0)[rows - 1]
print("mean squared monthly change per predictor:", dict(zip(config.US_PREDICTORS, np.round((dX ** 2).mean(0), 3))))
print("mean ||dx||^2:", round(float((dX ** 2).sum(1).mean()), 2), " median:", round(float(np.median((dX ** 2).sum(1))), 2))
