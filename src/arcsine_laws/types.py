"""Domain types shared across the package."""

from collections.abc import Callable
from enum import Flag, auto
from typing import Protocol

import numpy as np

type Seed = np.random.Generator | int | None
"""Source of randomness for a sampler.

Anything accepted by `np.random.default_rng`: an existing generator (used as
is, so successive calls continue its stream), an integer seed (reproducible
results), or None (fresh entropy from the operating system).
"""

type StepSampler = Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]
"""The distribution of the steps of a random walk.

A function `sample(rng, size)` returning an array of the given shape filled
with i.i.d. steps drawn using `rng`. Built by `rademacher` and `gaussian` in
`arcsine_laws.walks`.
"""


class Distribution(Protocol):
    """A continuous probability distribution with a density.

    Any frozen scipy distribution fits, e.g. `scipy.stats.norm(0, 1)` or
    `scipy.stats.arcsine()`. Used as a reference curve drawn over histograms
    of simulated statistics.
    """

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Density at each point of x."""
        ...

    def ppf(self, q: float) -> float:
        """Quantile function: the point below which a fraction q of mass lies."""
        ...


class ArcsineStatistic(Flag):
    """The statistics of a path covered by the arcsine laws.

    Members combine with `|`, e.g. `TIME_POSITIVE | LAST_ZERO`, to select
    several at once.

    Attributes
    ----------
    TIME_POSITIVE
        Total time the path spends above zero.
    LAST_ZERO
        Time of the path's last zero.
    MAXIMUM
        Time at which the path first reaches its maximum.
    ALL
        All three.
    """

    TIME_POSITIVE = auto()
    LAST_ZERO = auto()
    MAXIMUM = auto()
    ALL = TIME_POSITIVE | LAST_ZERO | MAXIMUM
