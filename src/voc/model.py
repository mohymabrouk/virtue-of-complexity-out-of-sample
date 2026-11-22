"""Random Fourier features + ridge in dual form, the linear kitchen sink, and the simple timing rules.

Notation: a forecast made at the end of month m (row m) is for R[m+1]. Its training pairs are
(X[m-T+j], R[m-T+j+1]) for j = 0..T-1, i.e. predictor rows m-T..m-1 and return rows m-T+1..m (KMZ section 6.3:
training observations {(R_t, S_t-1), ..., (R_t-T+1, S_t-T)}), and the test row is X[m].
"""

import numpy as np

from voc import config


# ---------------------------------------------------------------------------------------------------------------
# random Fourier features
# ---------------------------------------------------------------------------------------------------------------
def draw_omega(seed: int, n_pairs: int, k: int) -> np.ndarray:
    """omega_i ~ iid N(0, I_k); one independent stream per seed index."""
    rng = np.random.default_rng([config.BASE_SEED, int(seed)])
    return rng.standard_normal((n_pairs, k))


def rff(X: np.ndarray, omega: np.ndarray, gamma: float = config.GAMMA) -> np.ndarray:
    """KMZ eq. (21) without the P^-1/2 factor (it cancels in the within-window scaling).

    Columns are [sin(g w_1'x), cos(g w_1'x), sin(g w_2'x), cos(g w_2'x), ...], so the first P columns are the
    first P/2 pairs: models with fewer features are nested in models with more.
    """
    proj = gamma * (X @ omega.T)
    F = np.empty((X.shape[0], 2 * omega.shape[0]))
    F[:, 0::2] = np.sin(proj)
    F[:, 1::2] = np.cos(proj)
    return F


# ---------------------------------------------------------------------------------------------------------------
# ridge
# ---------------------------------------------------------------------------------------------------------------
def ridge_primal(S: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
    """beta(z) = (z I + S'S/T)^-1 S'y/T (KMZ section 3.1), S is T x P."""
    T, P = S.shape
    return np.linalg.solve(z * np.eye(P) + S.T @ S / T, S.T @ y / T)


def ridge_dual_forecast(S: np.ndarray, y: np.ndarray, s_test: np.ndarray, z: float) -> float:
    """s' beta(z) = s' S' (S S' + z T I)^-1 y, a T x T system."""
    T = S.shape[0]
    return float((S @ s_test) @ np.linalg.solve(S @ S.T + z * T * np.eye(T), y))


def window_grams(F: np.ndarray, rows: np.ndarray, P_grid, T: int = config.T, chunk: int = config.FEATURE_CHUNK) -> np.ndarray:
    """Gram matrices of the windows rows m-T..m (the last row is the test row) for every P in P_grid.

    Each feature is divided by its std (ddof = 0) over the T training rows m-T..m-1 of that window (KMZ footnote
    36). Returns an array (len(P_grid), len(rows), T+1, T+1); entry [k, i] uses the first P_grid[k] features.
    Only rows rows.min()-T .. rows.max() of F are read.
    """
    rows = np.asarray(rows)
    lo, hi = rows.min() - T, rows.max() + 1
    rel = rows - lo
    idx = rel[:, None] + np.arange(-T, 1)[None, :]
    n = len(rows)
    out = np.empty((len(P_grid), n, T + 1, T + 1))
    G = np.zeros((n, T + 1, T + 1))
    start = 0
    for k, P in enumerate(P_grid):
        if P > F.shape[1]:
            raise ValueError(f"P={P} > number of features {F.shape[1]}")
        while start < P:
            stop = min(P, start + chunk)
            Fc = np.ascontiguousarray(F[lo:hi, start:stop])
            # training-window variance from cumulative sums of centred values (variance is shift-invariant)
            D = Fc - Fc.mean(axis=0)
            C1 = np.zeros((len(D) + 1, D.shape[1]))
            C2 = np.zeros_like(C1)
            np.cumsum(D, axis=0, out=C1[1:])
            np.cumsum(D * D, axis=0, out=C2[1:])
            s1 = C1[rel] - C1[rel - T]  # rows m-T..m-1
            s2 = C2[rel] - C2[rel - T]
            var = s2 / T - (s1 / T) ** 2
            inv_sd = 1.0 / np.sqrt(np.maximum(var, 1e-24))
            W = Fc[idx]  # (n, T+1, b)
            W *= inv_sd[:, None, :]
            G += W @ W.transpose(0, 2, 1)
            start = stop
        out[k] = G
    return out


def dual_forecasts_multi(G: np.ndarray, Ys, z_grid, T: int = config.T) -> list[np.ndarray]:
    """Forecasts for every z from window Gram matrices G (n, T+1, T+1), for several target sets Ys (each (n, T)).

    z = 0 gives the ridgeless (minimum-norm) limit via the pseudo-inverse. Returns a list of (len(z_grid), n).
    """
    K = G[:, :T, :T]
    k = G[:, T, :T]
    lam, V = np.linalg.eigh(K)
    lam = np.maximum(lam, 0.0)
    kV = np.einsum("ni,nij->nj", k, V)
    invs = []
    for z in z_grid:
        if z == 0.0:
            tol = lam.max(axis=1, keepdims=True) * 1e-10
            invs.append(np.where(lam > tol, 1.0 / np.where(lam > tol, lam, 1.0), 0.0))
        else:
            invs.append(1.0 / (lam + z * T))
    res = []
    for Y in Ys:
        prod = kV * np.einsum("nij,ni->nj", V, Y)
        res.append(np.stack([(prod * inv).sum(axis=1) for inv in invs]))
    return res


def dual_forecasts(G: np.ndarray, Y: np.ndarray, z_grid, T: int = config.T) -> np.ndarray:
    """Single-target version of `dual_forecasts_multi`: returns (len(z_grid), n)."""
    return dual_forecasts_multi(G, [Y], z_grid, T)[0]


def training_targets(R: np.ndarray, rows: np.ndarray, T: int = config.T) -> np.ndarray:
    """Y[i] = R[m-T+1 .. m] for m = rows[i]."""
    return R[np.asarray(rows)[:, None] + np.arange(-T + 1, 1)[None, :]]


# ---------------------------------------------------------------------------------------------------------------
# forecasts
# ---------------------------------------------------------------------------------------------------------------
def voc_forecasts(ds, rows: np.ndarray, seed: int, P_grid=config.P_GRID, z_grid=config.Z_GRID,
                  T: int = config.T, gamma: float = config.GAMMA, targets=None) -> np.ndarray:
    """RFF ridge forecasts for one seed: array (len(P_grid), len(z_grid), len(rows)).

    `targets`: optional list of alternative training-target series (same length as ds.R); then the result has a
    leading axis over [ds.R] + targets. The features, Gram matrices and eigendecompositions are shared.
    """
    rows = np.asarray(rows)
    lo, hi = rows.min() - T, rows.max() + 1
    omega = draw_omega(seed, max(P_grid) // 2, ds.X.shape[1])
    F = np.full((len(ds.months), max(P_grid)), np.nan)
    F[lo:hi] = rff(ds.X[lo:hi], omega, gamma)
    G = window_grams(F, rows, P_grid, T)
    series = [ds.R] + (list(targets) if targets else [])
    Ys = [training_targets(np.asarray(R), rows, T) for R in series]
    out = np.empty((len(series), len(P_grid), len(z_grid), len(rows)))
    for k in range(len(P_grid)):
        res = dual_forecasts_multi(G[k], Ys, z_grid, T)
        for v in range(len(series)):
            out[v, k] = res[v]
    return out if targets else out[0]


def linear_forecasts(ds, rows: np.ndarray, z_grid=config.LINEAR_Z, T: int = config.T, target=None) -> np.ndarray:
    """Kitchen-sink ridge on the K scaled predictors (same window and within-window scaling): (len(z), n).

    `target`: training-target series (default ds.R, the standardised return)."""
    K = ds.X.shape[1]
    G = window_grams(ds.X, rows, (K,), T, chunk=K)[0]
    R = ds.R if target is None else np.asarray(target)
    return dual_forecasts(G, training_targets(R, rows, T), z_grid, T)


def simple_forecasts(ds, rows: np.ndarray, T: int = config.T, target=None) -> dict[str, np.ndarray]:
    """Positions of the simple rules (PLAN.md section 6), each using data up to month m only.

    `target`: the return series the rules look back on (default ds.R; the raw-return variant passes ds.r).
    `kmz_mom` always uses raw returns, as in KMZ appendix G."""
    rows = np.asarray(rows)
    R = ds.R if target is None else np.asarray(target)
    lag = np.arange(0, 12)  # k = 0..11
    Rw = R[rows[:, None] - lag[None, :]]  # R[m-k]
    rw = ds.r[rows[:, None] - lag[None, :]]
    out = {}
    ok = np.isfinite(R)
    assert ok[rows].all()
    csum = np.cumsum(np.where(ok, R, 0.0))
    cnt = np.cumsum(ok)
    out["hist_mean"] = csum[rows] / np.maximum(cnt[rows], 1)
    out["tsmom_sign"] = np.sign(Rw.sum(axis=1))
    out["mom12"] = Rw.mean(axis=1)
    mean_r = rw.mean(axis=1)
    var_r = (rw ** 2).mean(axis=1) - mean_r ** 2
    out["kmz_mom"] = mean_r / var_r
    Xw = ds.X[rows[:, None] + np.arange(-T, 0)[None, :]]  # training predictor rows m-T..m-1
    sig2 = Xw.var(axis=1).mean(axis=1)
    w = (12 - lag) / 78.0
    out["nagel_volmom"] = (Rw * w[None, :]).sum(axis=1) / sig2
    return out
