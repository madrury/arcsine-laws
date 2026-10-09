"""Samplers for random walks and Brownian motion paths.

All samplers return a `Paths`: a shared time grid `t` of shape (n_points,) and
an array of path values `x` of shape (n_paths, n_points). Every path starts at
zero at time zero.

Three ways to get (approximately) Brownian motion:

* `random_walk` with `gaussian(variance=dt)` steps and `dt=dt` is *exactly*
  Brownian motion observed on the grid 0, dt, 2 dt, ...
* `donsker` rescales any mean zero walk (e.g. a simple +-1 walk) in time and
  space; as n_steps grows the result converges in distribution to Brownian
  motion on [0, T].
* `refine` (Levy's midpoint construction) takes exact Brownian values on a grid
  and samples the values at every midpoint from the Brownian bridge. Starting
  from B(0) = 0, B(T) ~ N(0, T) and refining repeatedly gives Brownian motion
  at the dyadic rationals; `brownian_refinements` returns every level.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

type Seed = np.random.Generator | int | None
type StepSampler = Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]


@dataclass(frozen=True)
class Paths:
    t: np.ndarray
    x: np.ndarray

    @property
    def n_paths(self) -> int:
        return self.x.shape[0]

    @property
    def n_steps(self) -> int:
        return len(self.t) - 1

    @property
    def dx(self) -> np.ndarray:
        """The increments of each path, shape (n_paths, n_steps)."""
        return np.diff(self.x, axis=1)


# Step distributions.


def rademacher(p: float = 0.5) -> StepSampler:
    """Steps of +1 with probability p, -1 otherwise (a simple random walk)."""

    def sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
        return np.where(rng.random(size) < p, 1.0, -1.0)

    return sample


def gaussian(variance: float = 1.0, mean: float = 0.0) -> StepSampler:
    """Normally distributed steps."""
    scale = np.sqrt(variance)

    def sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
        return rng.normal(mean, scale, size)

    return sample


# Walks.


def random_walk(
    n_steps: int,
    step: StepSampler | None = None,
    n_paths: int = 1,
    dt: float = 1.0,
    rng: Seed = None,
) -> Paths:
    """Sample n_paths walks of n_steps i.i.d. steps, with steps spaced dt apart.

    >>> walk = random_walk(1000, rademacher(), n_paths=5)
    >>> bm = random_walk(1024, gaussian(variance=1 / 1024), dt=1 / 1024)
    """
    rng = np.random.default_rng(rng)
    step = step or rademacher()
    steps = step(rng, (n_paths, n_steps))
    x = np.zeros((n_paths, n_steps + 1))
    np.cumsum(steps, axis=1, out=x[:, 1:])
    t = dt * np.arange(n_steps + 1)
    return Paths(t, x)


def donsker(paths: Paths, step_variance: float = 1.0, T: float = 1.0) -> Paths:
    """Rescale a walk onto [0, T] so it approximates Brownian motion.

    The k'th of n steps lands at time k T / n, and space is scaled by
    sqrt(T / (n step_variance)) so the endpoint has variance T.
    """
    n = paths.n_steps
    t = T * np.arange(n + 1) / n
    x = paths.x * np.sqrt(T / (n * step_variance))
    return Paths(t, x)


# Brownian motion by midpoint refinement.


def refine(paths: Paths, rng: Seed = None, variance: float = 1.0) -> Paths:
    """Insert a Brownian bridge sample at the midpoint of every interval.

    Given B(a) and B(b), the midpoint B((a + b) / 2) is normal with mean
    (B(a) + B(b)) / 2 and variance (b - a) / 4. `variance` is the variance of
    the Brownian motion per unit time. Existing points are left unchanged.
    """
    rng = np.random.default_rng(rng)
    h = np.diff(paths.t)
    t_mid = paths.t[:-1] + h / 2
    x_mid = (paths.x[:, :-1] + paths.x[:, 1:]) / 2 + rng.normal(
        0.0, np.sqrt(variance * h / 4), (paths.n_paths, len(h))
    )

    t = np.empty(2 * len(paths.t) - 1)
    t[0::2], t[1::2] = paths.t, t_mid
    x = np.empty((paths.n_paths, len(t)))
    x[:, 0::2], x[:, 1::2] = paths.x, x_mid
    return Paths(t, x)


def brownian_refinements(
    levels: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: Seed = None,
    variance: float = 1.0,
) -> list[Paths]:
    """Brownian motion on [0, T] at successively finer dyadic grids.

    Element k of the result has 2**k steps, and each level contains every
    point of the one before it, so they are the same paths at finer and finer
    resolution.
    """
    rng = np.random.default_rng(rng)
    coarse = random_walk(1, gaussian(variance * T), n_paths=n_paths, dt=T, rng=rng)
    refinements = [coarse]
    for _ in range(levels):
        refinements.append(refine(refinements[-1], rng=rng, variance=variance))
    return refinements


def brownian_dyadic(
    levels: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: Seed = None,
    variance: float = 1.0,
) -> Paths:
    """Brownian motion on [0, T] sampled at the 2**levels + 1 dyadic points."""
    return brownian_refinements(levels, n_paths, T, rng, variance)[-1]
