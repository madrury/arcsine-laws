"""Domain types shared across the package."""

from collections.abc import Callable

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
