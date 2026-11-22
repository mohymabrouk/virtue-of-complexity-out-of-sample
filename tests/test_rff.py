"""Test 2: the random Fourier feature map has the right shape, sin/cos pairing and nesting."""

import numpy as np

from voc import config, model


def test_shape_and_pairing():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 15))
    omega = model.draw_omega(3, 6, 15)
    F = model.rff(X, omega, gamma=2.0)
    assert F.shape == (30, 12)
    proj = 2.0 * X @ omega.T
    np.testing.assert_allclose(F[:, 0::2], np.sin(proj))
    np.testing.assert_allclose(F[:, 1::2], np.cos(proj))
    np.testing.assert_allclose(F[:, 0::2] ** 2 + F[:, 1::2] ** 2, 1.0)


def test_nested_features_and_seeds():
    """The first P features of a 12,000-feature draw are the P-feature model (nested complexity)."""
    X = np.random.default_rng(1).standard_normal((10, 15))
    big = model.rff(X, model.draw_omega(5, config.TARGET_P // 2, 15))
    small = model.rff(X, model.draw_omega(5, 10, 15))
    np.testing.assert_allclose(big[:, :20], small)
    assert big.shape[1] == max(config.P_GRID)
    # different seeds give different draws, the same seed gives the same draw
    assert not np.allclose(model.draw_omega(1, 4, 15), model.draw_omega(2, 4, 15))
    np.testing.assert_array_equal(model.draw_omega(9, 4, 15), model.draw_omega(9, 4, 15))


def test_kernel_approximation():
    """Unscaled RFF dot products approximate the Gaussian kernel exp(-gamma^2 |x-x'|^2 / 2)."""
    rng = np.random.default_rng(2)
    x, xp = rng.standard_normal(3) * 0.3, rng.standard_normal(3) * 0.3
    omega = model.draw_omega(0, 200_000, 3)
    F = model.rff(np.vstack([x, xp]), omega, gamma=2.0)
    approx = F[0] @ F[1] / omega.shape[0]
    exact = np.exp(-(2.0**2) * np.sum((x - xp) ** 2) / 2)
    assert abs(approx - exact) < 0.01
