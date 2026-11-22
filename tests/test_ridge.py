"""Test 1: the dual (kernel) ridge equals the primal ridge, also through the batched window engine."""

import numpy as np
import pytest

from voc import model


@pytest.mark.parametrize("T,P", [(12, 4), (12, 12), (12, 40), (12, 600), (30, 7)])
@pytest.mark.parametrize("z", [1e-3, 1e-1, 1.0, 10.0, 1e3])
def test_dual_equals_primal(T, P, z):
    rng = np.random.default_rng(T * 1000 + P)
    S = rng.standard_normal((T, P))
    y = rng.standard_normal(T)
    s = rng.standard_normal(P)
    primal = s @ model.ridge_primal(S, y, z)
    dual = model.ridge_dual_forecast(S, y, s, z)
    assert dual == pytest.approx(primal, rel=1e-8, abs=1e-10)


def test_ridgeless_limit_is_min_norm():
    rng = np.random.default_rng(1)
    T, P = 12, 50
    S, y, s = rng.standard_normal((T, P)), rng.standard_normal(T), rng.standard_normal(P)
    beta = np.linalg.pinv(S) @ y  # minimum-norm least squares
    G = np.zeros((1, T + 1, T + 1))
    W = np.vstack([S, s])
    G[0] = W @ W.T
    f = model.dual_forecasts(G, y[None, :], (0.0, 1e-9), T)
    assert f[0, 0] == pytest.approx(s @ beta, rel=1e-6)
    assert f[1, 0] == pytest.approx(s @ beta, rel=1e-4)


def test_window_engine_matches_direct_primal():
    """window_grams + dual_forecasts (batched, nested P, within-window std) == per-month primal ridge."""
    rng = np.random.default_rng(7)
    N, K, T = 60, 5, 12
    X = np.cumsum(rng.standard_normal((N, K)), axis=0) * 0.1 + rng.standard_normal((N, K)) * 0.3
    R = rng.standard_normal(N)
    omega = model.draw_omega(0, 50, K)
    F = model.rff(X, omega)
    rows = np.arange(T + 3, N - 1, 7)
    P_grid, z_grid = (4, 20, 100), (1e-3, 1.0, 100.0)
    G = model.window_grams(F, rows, P_grid, T, chunk=16)
    Y = model.training_targets(R, rows, T)
    for k, P in enumerate(P_grid):
        fc = model.dual_forecasts(G[k], Y, z_grid, T)
        for i, m in enumerate(rows):
            S, s = F[m - T:m, :P], F[m, :P]
            sd = S.std(axis=0)
            for j, z in enumerate(z_grid):
                direct = (s / sd) @ model.ridge_primal(S / sd, R[m - T + 1:m + 1], z)
                assert fc[j, i] == pytest.approx(direct, rel=1e-7, abs=1e-10)
