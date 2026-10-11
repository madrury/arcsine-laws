"""Plots of random walks, Brownian motion, and their distributions.

Every function takes a matplotlib `Axes` as its first argument, draws on it,
and returns it. None of them create figures, so the caller controls layout;
wide axes, e.g. `plt.subplots(figsize=(14, 4))`, suit sample paths best.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from arcsine_laws.walks import Paths

#-----------------------------------------------------------------------------
# Colors.
#-----------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"

# Light to dark blue, for ordered sequences like refinement levels.
BLUES = LinearSegmentedColormap.from_list("blues", ["#86b6ef", "#0d366b"])


#-----------------------------------------------------------------------------
# Sample paths.
#-----------------------------------------------------------------------------
def plot_paths(
    ax: plt.Axes,
    paths: Paths,
    *,
    color: str = BLUE,
    alpha: float | None = None,
    linewidth: float = 1.0,
    highlight: int | None = None,
) -> plt.Axes:
    """Draw every path in a collection, with a reference line at zero.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    paths : Paths
        The paths to draw.
    color : str, default BLUE
        Line color for the paths.
    alpha : float, optional
        Line opacity. Defaults to a value that shrinks with the number of
        paths, so that dense regions of a large fan read as darker:

            clip(3 / sqrt(n_paths), 0.05, 1)

    linewidth : float, default 1.0
        Line width in points.
    highlight : int, optional
        Index of one path to draw on top of the others, opaque and in orange.

    Returns
    -------
    plt.Axes
        The axes drawn on.
    """
    if alpha is None:
        alpha = np.clip(3 / np.sqrt(paths.n_paths), 0.05, 1.0)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.plot(
        paths.t,
        paths.x.T,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
    )
    if highlight is not None:
        ax.plot(
            paths.t,
            paths.x[highlight],
            color=ORANGE,
            linewidth=1.5 * linewidth,
        )
    ax.set_xlim(paths.t[0], paths.t[-1])
    ax.set_xlabel("t")
    _tidy(ax)
    return ax


def plot_refinements(
    ax: plt.Axes,
    refinements: list[Paths],
    *,
    path: int = 0,
    max_markers: int = 64,
) -> plt.Axes:
    """Draw one path at each level of a midpoint refinement, coarse to fine.

    Coarser levels are lighter and drawn underneath finer ones. Levels with
    few enough points also mark them, to show where each refinement adds its
    midpoints.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    refinements : list of Paths
        Successive refinements of the same paths, coarse to fine, as returned
        by `brownian_refinements`. Pass a slice to draw only some levels.
    path : int, default 0
        Index of the path to draw.
    max_markers : int, default 64
        Mark the points of levels with at most this many steps.

    Returns
    -------
    plt.Axes
        The axes drawn on.
    """
    colors = BLUES(np.linspace(0, 1, len(refinements)))
    ax.axhline(0, color=INK, linewidth=0.8)
    for level, color in zip(refinements, colors):
        marker = "o" if level.n_steps <= max_markers else None
        label = "1 step" if level.n_steps == 1 else f"{level.n_steps} steps"
        ax.plot(
            level.t,
            level.x[path],
            color=color,
            linewidth=1.0,
            marker=marker,
            markersize=4,
            label=label,
        )
    ax.set_xlim(refinements[0].t[0], refinements[0].t[-1])
    ax.set_xlabel("t")
    # Outside the axes, to the right; wide figures have room to spare there.
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    _tidy(ax)
    return ax


#-----------------------------------------------------------------------------
# Distributions.
#-----------------------------------------------------------------------------
def plot_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    bins: int | np.ndarray | None = None,
    normal: tuple[float, float] | None = None,
    color: str = BLUE,
) -> plt.Axes:
    """Density histogram of samples, optionally against a normal density.

    Samples from a lattice, like the endpoints of a simple random walk, are
    binned with one bin per lattice point. Automatic bins would alias against
    the lattice and show a spurious comb pattern.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    values : np.ndarray
        One dimensional array of samples, e.g. `paths.x[:, -1]` for the
        endpoints of a collection of walks.
    bins : int or np.ndarray, optional
        Number of bins, or bin edges. Defaults to one bin per distinct value
        if there are at most 200 of them, and to 50 bins otherwise.
    normal : (float, float), optional
        Mean and variance of a normal density to draw over the histogram.
    color : str, default BLUE
        Fill color of the bars.

    Returns
    -------
    plt.Axes
        The axes drawn on.
    """
    if bins is None:
        bins = _default_bins(values)
    ax.hist(
        values,
        bins=bins,
        density=True,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        label="simulated",
    )
    if normal is not None:
        mean = normal[0]
        variance = normal[1]
        sd = np.sqrt(variance)
        grid = np.linspace(mean - 4*sd, mean + 4*sd, 400)
        ax.plot(
            grid,
            stats.norm.pdf(grid, mean, sd),
            color=ORANGE,
            linewidth=2.0,
            label=f"N({mean:g}, {variance:g})",
        )
        ax.legend(frameon=False, loc="upper right")
    _tidy(ax)
    return ax


#-----------------------------------------------------------------------------
# Helpers.
#-----------------------------------------------------------------------------
def _default_bins(values: np.ndarray) -> int | np.ndarray:
    lattice = np.unique(values)
    if len(lattice) > 200:
        return 50
    if len(lattice) == 1:
        return np.array([lattice[0] - 0.5, lattice[0] + 0.5])
    # Edges halfway between neighbouring lattice points, plus half a spacing
    # beyond each end.
    midpoints = (lattice[:-1] + lattice[1:]) / 2
    first = lattice[0] - (midpoints[0] - lattice[0])
    last = lattice[-1] + (lattice[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def _tidy(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
