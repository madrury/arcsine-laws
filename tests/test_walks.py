from itertools import pairwise

import numpy as np
import pytest

from arcsine_laws import (
    Paths,
    brownian_dyadic,
    brownian_refinements,
    donsker,
    gaussian,
    rademacher,
    random_walk,
    refine,
)

N_PATHS = 20_000


def test_simple_walk_steps_are_plus_or_minus_one():
    walk = random_walk(100, rademacher(), n_paths=10, rng=0)
    assert walk.x.shape == (10, 101)
    assert np.all(walk.x[:, 0] == 0)
    assert set(np.unique(walk.dx)) == {-1.0, 1.0}


def test_simple_walk_endpoint_moments():
    walk = random_walk(100, rademacher(), n_paths=N_PATHS, rng=1)
    end = walk.x[:, -1]
    assert end.mean() == pytest.approx(0, abs=0.2)
    assert end.var() == pytest.approx(100, rel=0.05)


def test_biased_walk_drifts():
    walk = random_walk(100, rademacher(p=0.6), n_paths=N_PATHS, rng=2)
    assert walk.x[:, -1].mean() == pytest.approx(100 * 0.2, rel=0.05)


def test_donsker_scaling_gives_unit_variance_at_T():
    walk = donsker(random_walk(400, rademacher(), n_paths=N_PATHS, rng=3), T=2.0)
    assert walk.t[-1] == pytest.approx(2.0)
    assert walk.x[:, -1].var() == pytest.approx(2.0, rel=0.05)


def test_refine_keeps_existing_points():
    coarse = random_walk(8, gaussian(1 / 8), n_paths=3, dt=1 / 8, rng=4)
    fine = refine(coarse, rng=5)
    assert fine.n_steps == 16
    np.testing.assert_array_equal(fine.t[0::2], coarse.t)
    np.testing.assert_array_equal(fine.x[:, 0::2], coarse.x)


def test_refinements_are_nested():
    levels = brownian_refinements(6, n_paths=2, rng=6)
    assert [p.n_steps for p in levels] == [2**k for k in range(7)]
    for coarse, fine in pairwise(levels):
        np.testing.assert_array_equal(fine.x[:, 0::2], coarse.x)


@pytest.mark.parametrize("variance", [1.0, 0.25])
def test_dyadic_brownian_motion_covariance(variance):
    # Cov(B_s, B_t) = variance * min(s, t), and increments are uncorrelated.
    bm = brownian_dyadic(4, n_paths=N_PATHS, rng=7, variance=variance)
    expected = variance * np.minimum.outer(bm.t, bm.t)
    np.testing.assert_allclose(np.cov(bm.x.T), expected, atol=0.03 * variance)
    increments = np.cov(bm.dx.T)
    np.testing.assert_allclose(
        increments, np.diag(np.diag(increments)), atol=0.01 * variance
    )


def test_comparisons_are_elementwise():
    walk = random_walk(50, rademacher(), n_paths=4, rng=8)
    np.testing.assert_array_equal(walk > 0, walk.x > 0)
    np.testing.assert_array_equal(walk <= 1, walk.x <= 1)
    assert (walk >= 0).shape == walk.x.shape


def test_reductions_are_per_path():
    walk = random_walk(50, rademacher(), n_paths=4, rng=9)
    np.testing.assert_array_equal(walk.max(), walk.x.max(axis=1))
    np.testing.assert_array_equal(walk.min(), walk.x.min(axis=1))
    np.testing.assert_array_equal(walk.x[np.arange(4), walk.argmax()], walk.max())
    np.testing.assert_array_equal(walk.x[np.arange(4), walk.argmin()], walk.min())


# A hand checked simple walk: up, down, down, up, up, up.
HAND = Paths(np.arange(7.0), np.array([[0.0, 1, 0, -1, 0, 1, 2]]))


def test_time_positive_simple_walk():
    # Steps (0, 1), (1, 0), (0, 1), (1, 2) are positive.
    assert HAND.time_positive()[0] == 4


def test_last_zero_simple_walk():
    assert HAND.last_zero()[0] == 4


def test_last_zero_interpolates_crossings():
    paths = Paths(
        np.array([0.0, 1.0, 2.0]),
        np.array([
            [0.0, 1.0, -1.0],  # Crosses halfway through the last step.
            [0.0, 1.0, 0.0],   # Ends at zero.
            [0.0, 1.0, 2.0],   # Never returns.
        ]),
    )
    np.testing.assert_allclose(paths.last_zero(), [1.5, 2.0, 0.0])


@pytest.mark.parametrize("statistic", ["time_positive", "last_zero", "argmax"])
def test_arcsine_laws(statistic):
    # In the Brownian limit each statistic, on [0, 1], has the arcsine
    # distribution, with cdf 2/pi * arcsin(sqrt(x)).
    walks = donsker(random_walk(1000, rademacher(), n_paths=N_PATHS, rng=10))
    if statistic == "argmax":
        values = walks.t[walks.argmax()]
    else:
        values = getattr(walks, statistic)()
    for x in [0.05, 0.25, 0.5, 0.75, 0.95]:
        expected = 2 / np.pi * np.arcsin(np.sqrt(x))
        assert np.mean(values <= x) == pytest.approx(expected, abs=0.02)
