"""Paper sample, diagnostic only: which return standardisation reproduces KMZ/Nagel's t = 2.81, IR = 0.296?"""
import numpy as np, pandas as pd, dataclasses
from voc import config, data, model, metrics

ds = data.us_dataset(last_month=config.PAPER_LAST)
rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
r = pd.Series(ds.r)
variants = {
    "sd(t-11..t) [planned]": r / r.rolling(12).std().shift(1),
    "sd(t-12..t-1)": r / r.rolling(12).std().shift(2),
    "sd(t-10..t+1) look-ahead": r / r.rolling(12).std(),
    "rms(t-11..t)": r / np.sqrt((r ** 2).rolling(12).mean()).shift(1),
}
for name, R in variants.items():
    d = dataclasses.replace(ds, R=R.to_numpy())
    y = d.R[rows + 1]
    acc = []
    for s in range(2):
        f = model.voc_forecasts(d, rows, s, P_grid=(12000,), z_grid=(1e-3, 1000.0))[0]
        st = metrics.batch_stats(f, y, y)
        acc.append([st["sharpe"], st["alpha_t"], st["ir"], st["r2"]])
    a = np.mean(acc, 0)
    bh = metrics.sharpe(pd.Series(y), 12)
    lin = metrics.batch_stats(model.linear_forecasts(d, rows, z_grid=(0.0, 1000.0)), y, y)
    print(f"{name:26s} BH {bh:.3f} | VoC z1e-3 SR {a[0,0]:.3f} t {a[1,0]:.2f} IR {a[2,0]:.3f} | z1e3 SR {a[0,1]:.3f} t {a[1,1]:.2f} IR {a[2,1]:.3f} R2 {a[3,1]:.4f}"
          f" | lin 0+ SR {lin['sharpe'][0]:.3f}, 1e3 SR {lin['sharpe'][1]:.3f} t {lin['alpha_t'][1]:.2f} R2 {lin['r2'][1]:.3f}")
