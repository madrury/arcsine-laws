"""Samplers for random walks and Brownian motion paths.

All samplers return a `Paths`: a shared time grid `t` of shape (n_points,) and
an array of path values `x` of shape (n_paths, n_points). Every path starts at
zero at time zero.

Three ways to get (approximately) Brownian motion:

* `random_walk` with `gaussian(variance=dt)` steps and `dt=dt` is *exactly*
  Brownian motion observed on the grid 0, dt, 2*dt, ...
* `donsker` rescales any mean zero walk (e.g. a simple +-1 walk) in time and
  space; as n_steps grows the result converges in distribution to Brownian
  motion on [0, T].
* `refine` (Levy's midpoint construction) takes exact Brownian values on a grid
  and samples the values at every midpoint from the Brownian bridge connecting the
  adjacent points. Starting from B(0) = 0, B(T) ~ N(0, T) and refining repeatedly
  gives Brownian motion at the dyadic rationals; `brownian_refinements` returns
  every level in a list[Paths].
"""

from dataclasses import dataclass

import numpy as np

from arcsine_laws.types import Seed, StepSampler


@dataclass(frozen=True)
class Paths:
    """A collection of sample paths observed on a shared time grid.

    Parameters
    ----------
    t : np.ndarray, shape (n_points,)
        Increasing observation times, starting at 0.
    x : np.ndarray, shape (n_paths, n_points)
        Path values; x[i, j] is the value of path i at time t[j].

    Attributes
    ----------
    n_paths : int
        Number of sample paths.
    n_steps : int
        Number of steps in each path, one fewer than the number of points.
    dx : np.ndarray, shape (n_paths, n_steps)
        The increments of each path.
    """

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
        return np.diff(self.x, axis=1)


#-----------------------------------------------------------------------------
# Step distributions.
#-----------------------------------------------------------------------------
def rademacher(p: float = 0.5) -> StepSampler:
    """Steps of +1 or -1, as in a simple random walk.

    Parameters
    ----------
    p : float, default 0.5
        Probability of a +1 step. Values other than 0.5 give a walk with drift
        2*p - 1 per step.

    Returns
    -------
    StepSampler
        A function `sample(rng, size)` returning an array of steps.
    """
    def sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
        return np.where(rng.random(size) < p, 1.0, -1.0)
    return sample


def gaussian(variance: float = 1.0, mean: float = 0.0) -> StepSampler:
    """Normally distributed steps.

    Parameters
    ----------
    variance : float, default 1.0
        Variance of each step. To sample Brownian motion exactly on a grid
        with spacing `dt`, use `variance=dt`.
    mean : float, default 0.0
        Mean of each step, i.e. the drift per step.

    Returns
    -------
    StepSampler
        A function `sample(rng, size)` returning an array of steps.
    """
    scale = np.sqrt(variance)
    def sample(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
        return rng.normal(mean, scale, size)
    return sample


#-----------------------------------------------------------------------------
# Walks.
#-----------------------------------------------------------------------------
def random_walk(
    n_steps: int,
    step: StepSampler | None = None,
    n_paths: int = 1,
    dt: float = 1.0,
    rng: Seed = None,
) -> Paths:
    """Sample random walks with i.i.d. steps.

    Parameters
    ----------
    n_steps : int
        Number of steps in each walk.
    step : StepSampler, optional
        Distribution of the steps, e.g. `rademacher()` or `gaussian(...)`.
        Defaults to `rademacher()`, a simple random walk.
    n_paths : int, default 1
        Number of independent walks to sample.
    dt : float, default 1.0
        Time between consecutive steps.
    rng : np.random.Generator, int or None, optional
        Random generator, or a seed for one.

    Returns
    -------
    Paths
        Walks with n_steps + 1 points at times 0, dt, 2*dt, ..., n_steps*dt.

    Examples
    --------
    >>> walk = random_walk(1000, rademacher(), n_paths=5)
    >>> bm = random_walk(1024, gaussian(variance=1/1024), dt=1/1024)
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

    The k'th of n steps lands at time k*T / n, and space is scaled by

        sqrt(T / (n*step_variance))

    so the endpoint has variance `T`. By Donsker's theorem the result converges
    in distribution to Brownian motion as n grows.

    https://en.wikipedia.org/wiki/Donsker%27s_theorem

    Parameters
    ----------
    paths : Paths
        Mean zero walks to rescale. Only the number of steps is used from the
        time grid.
    step_variance : float, default 1.0
        Variance of a single step of the walk.
    T : float, default 1.0
        Length of the time interval to rescale onto.

    Returns
    -------
    Paths
        The rescaled walks.
    """
    n = paths.n_steps
    t = T * np.arange(n + 1) / n
    x = paths.x * np.sqrt(T / (n * step_variance))
    return Paths(t, x)


#-----------------------------------------------------------------------------
# Brownian motion by midpoint refinement.
#-----------------------------------------------------------------------------
def refine(paths: Paths, rng: Seed = None, variance: float = 1.0) -> Paths:
    """Insert a Brownian bridge sample at the midpoint of every interval.

    If B is Brownian motion with variance v per unit time (v = `variance`),
    then given B(a) and B(b), the midpoint B((a + b) / 2) is normal with mean
    (B(a) + B(b)) / 2 and variance:

        v * (b - a) / 4

    So, if the input is exact Brownian motion on its grid, so is the output on
    the finer grid.

    Parameters
    ----------
    paths : Paths
        Paths to refine. Existing points are left unchanged.
    rng : np.random.Generator, int or None, optional
        Random generator, or a seed for one.
    variance : float, default 1.0
        Variance per unit time of the Brownian motion.

    Returns
    -------
    Paths
        Paths with 2*n_steps steps; the original points sit at the even indices
        and the new midpoints at the odd ones.
    """
    rng = np.random.default_rng(rng)
    dt = np.diff(paths.t)
    t_mid = paths.t[:-1] + dt / 2
    x_mid = (
        (paths.x[:, :-1] + paths.x[:, 1:]) / 2
        + rng.normal(
            0.0,
            np.sqrt(variance * dt / 4),
            size=(paths.n_paths, len(dt))
        )
    )
    t = np.empty(2 * len(paths.t) - 1)
    t[0::2] = paths.t
    t[1::2] = t_mid
    x = np.empty((paths.n_paths, len(t)))
    x[:, 0::2] = paths.x
    x[:, 1::2] = x_mid
    return Paths(t, x)


def brownian_refinements(
    n_refinements: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: Seed = None,
    variance: float = 1.0,
) -> list[Paths]:
    """Brownian motion on [0, T] at successively finer dyadic grids.

    Each level contains every point of the one before it, so these are the
    same paths at finer and finer resolution.

    Parameters
    ----------
    n_refinements : int
        Number of midpoint refinements to apply.
    n_paths : int, default 1
        Number of independent paths to sample.
    T : float, default 1.0
        Length of the time interval.
    rng : np.random.Generator, int or None, optional
        Random generator, or a seed for one.
    variance : float, default 1.0
        Variance per unit time of the Brownian motion.

    Returns
    -------
    list of Paths
        The n_refinements + 1 levels; element k has 2**k steps.
    """
    rng = np.random.default_rng(rng)
    coarse = random_walk(1, gaussian(variance * T), n_paths=n_paths, dt=T, rng=rng)
    refinements = [coarse]
    for _ in range(n_refinements):
        refinements.append(refine(refinements[-1], rng=rng, variance=variance))
    return refinements


def brownian_dyadic(
    n_refinements: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: Seed = None,
    variance: float = 1.0,
) -> Paths:
    """Brownian motion on [0, T] sampled at 2**n_refinements + 1 dyadic points.

    Parameters
    ----------
    n_refinements : int
        Number of midpoint refinements; the result has 2**n_refinements steps.
    n_paths : int, default 1
        Number of independent paths to sample.
    T : float, default 1.0
        Length of the time interval.
    rng : np.random.Generator, int or None, optional
        Random generator, or a seed for one.
    variance : float, default 1.0
        Variance per unit time of the Brownian motion.

    Returns
    -------
    Paths
        The finest level of `brownian_refinements`.
    """
    return brownian_refinements(n_refinements, n_paths, T, rng, variance)[-1]
