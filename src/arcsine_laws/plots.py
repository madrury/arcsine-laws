"""Plots of random walks, Brownian motion, and their distributions.

Every function takes a matplotlib `Axes` as its first argument, draws on it,
and returns it. None of them create figures, so the caller controls layout;
wide axes, e.g. `plt.subplots(figsize=(14, 4))`, suit sample paths best.
"""


from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from arcsine_laws.types import ArcsineStatistic, Distribution
from arcsine_laws.walks import Paths


#-----------------------------------------------------------------------------
# Colors.
#-----------------------------------------------------------------------------
class Colors:
    """Named colors used across the plots, as hex strings.

    Attributes
    ----------
    BLUE : str
        Default color for paths and histogram bars.
    LIGHTBLUE : str
        Light end of the blue ramp.
    DARKBLUE : str
        Dark end of the blue ramp.
    WARMGREEN : str
        A yellowish green, about as light as BLUE.
    DARKGREEN : str
        A dark green, about as dark as DARKBLUE.
    ORANGE : str
        Highlights and reference curves drawn over the blues.
    INK : str
        Axes, ticks, labels, and reference lines.
    GRID : str
        Grid lines.
    """

    BLUE = "#2a78d6"
    LIGHTBLUE = "#86b6ef"
    DARKBLUE = "#0d366b"
    WARMGREEN = "#5e9e2f"
    DARKGREEN = "#1d4f22"
    ORANGE = "#eb6834"
    INK = "#52514e"
    GRID = "#e4e3df"


class ColorMaps:
    """Color maps for ordered sequences, like refinement levels.

    Attributes
    ----------
    BLUES : LinearSegmentedColormap
        Light to dark blue.
    BLUE_GREEN : LinearSegmentedColormap
        Blue to warm green, at roughly constant lightness, so neither end
        fades into the background.
    """

    BLUES = LinearSegmentedColormap.from_list(
        "blues",
        [Colors.LIGHTBLUE, Colors.DARKBLUE],
    )
    BLUE_GREEN = LinearSegmentedColormap.from_list(
        "blue_green",
        [Colors.BLUE, Colors.WARMGREEN],
    )


#-----------------------------------------------------------------------------
# Sample paths.
#-----------------------------------------------------------------------------
def plot_paths(
    ax: Axes,
    paths: Paths,
    *,
    color: str = Colors.BLUE,
    alpha: float | None = None,
    linewidth: float = 1.0,
    highlight: int | None = None,
) -> Axes:
    """Draw every path in a collection, with a reference line at zero.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    paths : Paths
        The paths to draw.
    color : str, default Colors.BLUE
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
    Axes
        The axes drawn on.
    """
    if alpha is None:
        alpha = np.clip(3 / np.sqrt(paths.n_paths), 0.05, 1.0)
    ax.axhline(0, color=Colors.INK, linewidth=0.8, linestyle='--')
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
            color=Colors.ORANGE,
            linewidth=1.5 * linewidth,
        )
    ax.set_xlim(paths.t[0], paths.t[-1])
    ax.set_xlabel("t")
    _tidy(ax)
    return ax


def plot_refinements(
    ax: Axes,
    refinements: list[Paths],
    *,
    path: int = 0,
    max_markers: int = 64,
) -> Axes:
    """Draw one path at each level of a midpoint refinement, coarse to fine.

    Coarser levels are lighter and drawn underneath finer ones. Levels with
    few enough points also mark them, to show where each refinement adds its
    midpoints.

    Parameters
    ----------
    ax : Axes
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
    Axes
        The axes drawn on.
    """
    colors = ColorMaps.BLUES(np.linspace(0, 1, len(refinements)))
    ax.axhline(0, color=Colors.INK, linewidth=0.8, linestyle='--')
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


def plot_path_statistics(
    ax: Axes,
    paths: Paths,
    *,
    path: int = 0,
    show: ArcsineStatistic = ArcsineStatistic.ALL,
) -> Axes:
    """Draw one path, marking the statistics of the arcsine laws.

    Each statistic has the arcsine distribution in the Brownian limit, and
    each is drawn only if selected in `show`:

    * TIME_POSITIVE shades the area between the path and zero where the path
      is above zero, and marks the positive steps with a thick bar along the
      t-axis. The bar's total length is the time positive.
    * LAST_ZERO draws a vertical line at the last zero.
    * MAXIMUM marks the first maximum with a dot.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    paths : Paths
        The paths to draw from.
    path : int, default 0
        Index of the path to draw.
    show : ArcsineStatistic, default ArcsineStatistic.ALL
        Which statistics to mark, e.g. `ArcsineStatistic.LAST_ZERO`, or several
        combined with `|`.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    one = Paths(paths.t, paths.x[path:path + 1])
    t = one.t
    x = one.x[0]
    ax.axhline(0, color=Colors.INK, linewidth=0.8, linestyle='--')
    if ArcsineStatistic.TIME_POSITIVE in show:
        shade_under_walk(ax, one)
        plot_positive_intervals(ax, one, 0.0)
    ax.plot(t, x, color=Colors.BLUE, linewidth=1.0)
    if ArcsineStatistic.LAST_ZERO in show:
        t_last_zero = one.last_zero()[0]
        ax.axvline(
            t_last_zero,
            color=Colors.ORANGE,
            linewidth=1.5,
            label=f"last zero: {t_last_zero:.3g}",
        )
    if ArcsineStatistic.MAXIMUM in show:
        i_max = one.argmax()[0]
        ax.plot(
            t[i_max],
            x[i_max],
            marker="o",
            markersize=7,
            color=Colors.DARKGREEN,
            linestyle="none",
            label=f"maximum at: {t[i_max]:.3g}",
        )
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("t")
    if show:
        ax.legend(frameon=False)#, loc="upper left", bbox_to_anchor=(1.01, 1))
    _tidy(ax)
    return ax


def shade_under_walk(ax: Axes, one: Paths) -> Axes:
    """Shade the area bounded *under* a random walk path and above the
    horizontal axis.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    one : Paths
        The path (singular) to draw from.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    t = one.t
    x = one.x[0]
    ax.fill_between(
        t,
        0,
        x,
        where=x > 0,
        interpolate=True,
        color=Colors.LIGHTBLUE,
        alpha=0.5,
        linewidth=0
    )
    return ax

def plot_positive_intervals(ax: Axes, one: Paths, y: float) -> Axes:
    """Highlight (in the same sense of a highlighter pen) intervals over which
    the random walk is positive.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    one : Paths
        The path (singular) to draw from.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    # Runs of consecutive positive steps, as (start, end) step indices. Pad
    # with a non-positive step at each end so every run has both a rise and a
    # fall in the differences.
    t = one.t
    time_positive = one.time_positive()[0]
    positive = one.positive_steps()[0].astype(int)
    changes = np.diff(np.concatenate([[0], positive, [0]]))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    # A bar along the t-axis: x in data coordinates, y in axes coordinates.
    ax.hlines(
        np.full(len(starts), y),
        t[starts],
        t[ends],
        # transform=ax.get_xaxis_transform(),
        color=Colors.LIGHTBLUE,
        linewidth=6,
        capstyle="butt",
        clip_on=False,
        zorder=3,
        label=f"time positive: {time_positive:2.0%}",
    )
    return ax

#-----------------------------------------------------------------------------
# Distributions.
#-----------------------------------------------------------------------------
def plot_histogram(
    ax: Axes,
    values: np.ndarray,
    *,
    bins: int | Sequence[float] | np.ndarray | None = None,
    density: bool = True,
    reference: Distribution | None = None,
    reference_label: str = "reference",
    color: str = Colors.BLUE,
) -> Axes:
    """Histogram of samples, optionally against a reference density.

    Samples from a lattice, like the endpoints of a simple random walk, are
    binned with one bin per lattice point. Automatic bins would alias against
    the lattice and show a spurious comb pattern.

    When plotting counts, the reference density is scaled to expected counts
    per bin, assuming equal width bins:

        n_samples * bin_width * pdf(x)

    A reference density may be unbounded, like the arcsine density at 0 and 1.
    The y-axis is then capped a little above the tallest bar, or the density
    over the central 90% of its mass, whichever is higher.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    values : np.ndarray
        One dimensional array of samples, e.g. `paths.x[:, -1]` for the
        endpoints of a collection of walks.
    bins : int, sequence of float, or np.ndarray, optional
        Number of bins, or bin edges. Defaults to one bin per distinct value
        if there are at most 200 of them, and to 50 bins otherwise.
    density : bool, default True
        If true, plot probability density. Otherwise plot bin counts.
    reference : Distribution, optional
        A distribution whose density is drawn over the histogram, e.g.
        `scipy.stats.norm(0, 1)` or `scipy.stats.arcsine()`.
    reference_label : str, default "reference"
        Legend label for the reference density.
    color : str, default Colors.BLUE
        Fill color of the bars.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    # Matplotlib's type stubs accept a sequence of edges, but not an array.
    hist_bins: int | Sequence[float]
    if bins is None:
        hist_bins = _default_bins(values)
    elif isinstance(bins, np.ndarray):
        hist_bins = bins.tolist()
    else:
        hist_bins = bins
    heights, edges, _ = ax.hist(
        values,
        bins=hist_bins,
        density=density,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        label="simulated",
    )
    if reference is not None:
        scale = 1.0 if density else len(values) * np.diff(edges).mean()
        grid = np.linspace(reference.ppf(1e-4), reference.ppf(1 - 1e-4), 1000)
        curve = scale * reference.pdf(grid)
        ax.plot(
            grid,
            curve,
            color=Colors.ORANGE,
            linewidth=2.0,
            label=reference_label,
        )
        central = (grid >= reference.ppf(0.05)) & (grid <= reference.ppf(0.95))
        top = max(np.max(heights), np.max(curve[central]))
        ax.set_ylim(0, 1.15 * top)
        ax.legend(frameon=False, loc="upper right")
    _tidy(ax)
    return ax


#-----------------------------------------------------------------------------
# Helpers.
#-----------------------------------------------------------------------------
def _default_bins(values: np.ndarray) -> int | list[float]:
    lattice = np.unique(values)
    if len(lattice) > 200:
        return 50
    if len(lattice) == 1:
        return [lattice[0] - 0.5, lattice[0] + 0.5]
    # Edges halfway between neighbouring lattice points, plus half a spacing
    # beyond each end.
    midpoints = (lattice[:-1] + lattice[1:]) / 2
    first = lattice[0] - (midpoints[0] - lattice[0])
    last = lattice[-1] + (lattice[-1] - midpoints[-1])
    return [first, *midpoints.tolist(), last]


def _tidy(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(Colors.INK)
    ax.tick_params(colors=Colors.INK)
    ax.xaxis.label.set_color(Colors.INK)
    ax.yaxis.label.set_color(Colors.INK)
    ax.grid(color=Colors.GRID, linewidth=0.6, linestyle='--')
    ax.set_axisbelow(True)
