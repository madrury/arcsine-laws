# arcsine-laws

Simulations and plots for a ~30 minute Python users group talk on the arcsine
laws, inspired by Feller:

> We shall encounter theoretical conclusions which not only are unexpected but
> actually come as a shock to intuition and common sense.

Route: random walks → convergence to a distribution / CLT → Brownian motion →
arcsine laws → what a long winning (or losing) streak says about skill.

## Setup

    uv sync
    uv run pytest
    uv run jupyter lab notebooks/

## Library

`arcsine_laws.walks` (re-exported at top level). Every sampler returns a
`Paths(t, x)` with `t` of shape `(n_points,)` and `x` of shape
`(n_paths, n_points)`, starting at 0.

| function | what it gives |
|---|---|
| `random_walk(n, step, n_paths, dt)` | i.i.d. step walk; `step` is `rademacher(p)` or `gaussian(variance)` |
| `donsker(paths, step_variance, T)` | rescale a walk onto `[0, T]` → approximately Brownian |
| `refine(paths)` | Lévy midpoint refinement: Brownian bridge sample at each midpoint |
| `brownian_refinements(levels)` | Brownian motion at 2, 3, 5, 9, … dyadic points, nested |
| `brownian_dyadic(levels)` | the finest of those |
