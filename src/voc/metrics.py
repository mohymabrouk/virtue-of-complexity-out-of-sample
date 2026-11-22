"""Performance statistics.

`sharpe` and `deflated_sharpe` are copied verbatim (not imported) from Alpha Audit,
C:/dev/quant-agent-lab/backend/src/qal/lab/metrics.py (functions `sharpe` and `deflated_sharpe`), by the same
author's project, so that both projects compute these two numbers the same way. Only the imports they need are
repeated here. Everything else in this file is new.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


# ---- copied from Alpha Audit (qal/lab/metrics.py) -------------------------------------------------------------
def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def deflated_sharpe(returns: pd.Series, trials: int, trial_sharpe_variance: float) -> float:
    """Probability that the strategy's true Sharpe is above what the best of `trials` random strategies would show
    (Bailey & López de Prado, 2014). Sharpe ratios here are per period (not annualised), as in the paper.

    `trial_sharpe_variance` is the variance of the per-period Sharpe ratios of all trials in the family.
    With fewer than two trials the benchmark is zero, which is the probabilistic Sharpe ratio.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return 0.0
    sr = float(x.mean() / x.std(ddof=1))
    # Sample skewness and (non-excess) kurtosis, as in the paper.
    z_scores = (x - x.mean()) / x.std()
    skew = float(np.mean(z_scores**3))
    kurtosis = float(np.mean(z_scores**4))
    if trials >= 2 and trial_sharpe_variance > 0:
        expected_max = math.sqrt(trial_sharpe_variance) * (
            (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / trials)
            + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (trials * math.e))
        )
    else:
        expected_max = 0.0
    denominator = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if denominator <= 0:
        return 0.0
    z = (sr - expected_max) * math.sqrt(n - 1) / math.sqrt(denominator)
    return float(_NORMAL.cdf(z))


# ---- end of copied code -----------------------------------------------------------------------------------------


def sharpe_se(returns, periods_per_year: int = 12) -> float:
    """Annualised standard error of the Sharpe ratio: Lo (2002) iid formula with the Mertens (2002) correction
    for skewness and kurtosis, Var(SR) = (1 + SR^2/2 - g3 SR + (g4 - 3)/4 SR^2) / n on the per-period SR."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std()
    g3, g4 = float(np.mean(z**3)), float(np.mean(z**4))
    v = (1 + 0.5 * sr**2 - g3 * sr + (g4 - 3) / 4 * sr**2) / n
    return float(math.sqrt(max(v, 0.0)) * math.sqrt(periods_per_year))


def oos_r2(forecast, realised) -> float:
    """1 - sum (R - f)^2 / sum R^2 (zero benchmark, KMZ section 2.3)."""
    f, y = np.asarray(forecast, float), np.asarray(realised, float)
    return float(1 - np.sum((y - f) ** 2) / np.sum(y**2))


def newey_west_se_mean(e: np.ndarray, lags: int) -> float:
    """HAC (Bartlett kernel) variance of the mean of a series e, returned as a standard error."""
    e = e - e.mean()
    n = len(e)
    s = np.dot(e, e) / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * np.dot(e[L:], e[:-L]) / n
    return math.sqrt(max(s, 0.0) / n)


def alpha_regression(strategy, market, periods_per_year: int = 12, nw_lags: int = 12) -> dict:
    """OLS of strategy on market: annualised alpha, iid and Newey-West t-stats, beta, information ratio."""
    y = np.asarray(strategy, float)
    x = np.asarray(market, float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (n - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_iid = math.sqrt(s2 * XtX_inv[0, 0])
    # Newey-West sandwich for the intercept
    u = X * resid[:, None]
    S = u.T @ u / n
    for L in range(1, min(nw_lags, n - 1) + 1):
        w = 1 - L / (nw_lags + 1)
        g = u[L:].T @ u[:-L] / n
        S += w * (g + g.T)
    V = n * XtX_inv @ S @ XtX_inv
    se_nw = math.sqrt(max(V[0, 0], 0.0))
    sd_resid = math.sqrt(s2)
    return {"alpha_ann": float(coef[0] * periods_per_year), "beta": float(coef[1]),
            "alpha_t": float(coef[0] / se_iid) if se_iid > 0 else float("nan"),
            "alpha_t_nw": float(coef[0] / se_nw) if se_nw > 0 else float("nan"),
            "ir": float(coef[0] / sd_resid * math.sqrt(periods_per_year)) if sd_resid > 0 else float("nan")}


def batch_stats(F: np.ndarray, y: np.ndarray, mkt: np.ndarray, periods_per_year: int = 12) -> dict[str, np.ndarray]:
    """Vectorised statistics for many forecast series at once (used per seed, per grid cell).

    F: (S, n) forecasts = positions, y: (n,) realised standardised return, mkt: (n,) benchmark return.
    Strategy return = F * y. Returns arrays of length S: sharpe, r2, alpha_t (iid), ir, mean, vol, mean_pos.
    """
    F = np.atleast_2d(F)
    n = F.shape[1]
    ret = F * y[None, :]
    mean = ret.mean(axis=1)
    sd = ret.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = mean / sd * math.sqrt(periods_per_year)
        r2 = 1 - ((y[None, :] - F) ** 2).sum(axis=1) / np.sum(y**2)
        xm = mkt.mean()
        xc = mkt - xm
        sxx = np.sum(xc**2)
        beta = (ret - mean[:, None]) @ xc / sxx
        alpha = mean - beta * xm
        resid = ret - alpha[:, None] - beta[:, None] * mkt[None, :]
        s2 = (resid**2).sum(axis=1) / (n - 2)
        alpha_t = alpha / np.sqrt(s2 * (1 / n + xm**2 / sxx))
        ir = alpha / np.sqrt(s2) * math.sqrt(periods_per_year)
    return {"sharpe": sr, "r2": r2, "alpha_t": alpha_t, "ir": ir, "mean": mean * periods_per_year,
            "vol": sd * math.sqrt(periods_per_year), "mean_pos": F.mean(axis=1)}


def block_bootstrap_dsharpe(strategy, benchmark, reps: int = 2000, block: int | None = None, seed: int = 7,
                            periods_per_year: int = 12) -> tuple[float, float, float]:
    """Sharpe(strategy) - Sharpe(benchmark) with a 95% circular block-bootstrap percentile CI (paired months)."""
    a = np.asarray(strategy, float)
    b = np.asarray(benchmark, float)
    n = len(a)
    L = block or max(3, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / L))
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(reps, -1)[:, :n] % n
    A, B = a[idx], b[idx]

    def sr(M):
        return M.mean(axis=1) / M.std(axis=1, ddof=1) * math.sqrt(periods_per_year)

    d = sr(A) - sr(B)
    point = float(a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * math.sqrt(periods_per_year)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return point, float(lo), float(hi)


def full_stats(positions, y, mkt, trials: int, trial_var: float, periods_per_year: int = 12,
               bootstrap: bool = True) -> dict:
    """All statistics for one strategy: Sharpe, SE/CI, deflated Sharpe, alpha, IR, R2, Delta Sharpe vs market."""
    f = np.asarray(positions, float)
    ret = pd.Series(f * y)
    sr = sharpe(ret, periods_per_year)
    se = sharpe_se(ret, periods_per_year)
    reg = alpha_regression(ret.to_numpy(), mkt, periods_per_year)
    out = {"n": len(ret), "sharpe": sr, "sharpe_se": se, "sharpe_lo": sr - 1.96 * se, "sharpe_hi": sr + 1.96 * se,
           "dsr": deflated_sharpe(ret, trials, trial_var), "r2": oos_r2(f, y),
           "mean_ann": float(ret.mean() * periods_per_year), "vol_ann": float(ret.std(ddof=1) * math.sqrt(periods_per_year)),
           "skew": float(ret.skew()), "max_loss_sd": float(-(ret.min() - ret.mean()) / ret.std(ddof=1)),
           "mean_pos": float(f.mean()), "share_long": float((f > 0).mean()), **reg}
    if bootstrap:
        d, lo, hi = block_bootstrap_dsharpe(ret.to_numpy(), mkt, periods_per_year=periods_per_year)
        out.update({"dsharpe": d, "dsharpe_lo": lo, "dsharpe_hi": hi})
    return out


def power_one_sided(ir: float, years: float, alpha: float = 0.05) -> float:
    """Probability that an alpha t-stat exceeds the one-sided critical value when the true IR is `ir`."""
    return float(1 - _NORMAL.cdf(_NORMAL.inv_cdf(1 - alpha) - ir * math.sqrt(years)))
