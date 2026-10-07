"""Plot the multinomial vs. Gaussian-approximation comparison from the
``results/noise_comparison_d=..._ntrain=..._M=....pkl`` files produced by
``noise_comparison.py`` in this same folder.

The figure compares single simulated *experiments*: one experiment is one
noise realization of the training data (one draw of N shots per training
state), whose MSE is averaged over the test states and observables. For
each n_tr, with the true multinomial shot noise, the mean MSE over all
experiments (all noise realizations of all instances) is plotted against
the training shot budget N on a log-log scale, with ring-and-dot markers
from ``markers.py``, standard-error bars, and a shaded band spanning the
[p10, p90] quantiles of the single-experiment MSE. With the Gaussian
approximation the same three quantities are drawn on top as dotted lines
of the same color (thick: mean; thin: p10 and p90). The two noise models
give statistically the same experiments wherever the dotted lines follow
the markers and the band edges. The asymptotes E[b^2]/(N + d(d+2))^2 and
E[V]/(N n_tr) are shown for reference in the panels of the n_tr in
`ASYMPTOTE_NTRAIN` (they require n_tr >> n_out), and only for the MUB POVM,
for which they were derived.

Set the parameters below to match a previous ``noise_comparison.py`` run
(DIM, M_TEST and the n_tr values determine the results filenames). Run
with ``python3 plot_noise_comparison.py`` from anywhere; it locates its
input/output directories next to itself.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noise_comparison.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
import noise_comparison as nc  # noqa: E402
from markers import circle, diamond, hexagon, marker_inner_style, marker_outer_style, pentagon, square, styled, triangle  # noqa: E402

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb, mathpazo,bm}"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 17,
        "axes.labelsize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.0,
    }
)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "noise_comparison_plot.pdf"

# ----------------------------------------------------------------------
# Parameters identifying which saved results files to load -- must match
# a previous noise_comparison.py run.
# ----------------------------------------------------------------------

DIM = 2
M_TEST = None  # None = exact test statistics
POVM_TYPE = nc.POVM_TYPE  # "mub" or "random", as in the noise_comparison.py run
N_OUT = nc.N_OUT  # number of POVM outcomes (enters the results filename for "random")
NTRAIN_ORDER = tuple(nc.N_TRAIN_VALUES)  # one panel each, row by row
NCOLS = 3

# (color, shape, markersize), cycled over the panels; palette and shapes as
# in `M_inf/plot_N_scaling.py`. `square` (from markers.py) renders as a
# diamond and `diamond` renders as a square -- see the note in markers.py.
PANEL_STYLES = [
    ("#8FB032", square, 11),    # green, diamond
    ("#5E81B5", diamond, 11),   # blue, square
    ("#E19C24", triangle, 13),  # orange, triangle
    ("#EB6235", circle, 11),    # red, circle
    ("#A64CB8", pentagon, 11),  # purple, pentagon
    ("#3AA39B", hexagon, 11),   # teal, hexagon
]
EDGEWIDTH = 1.3

# Line style of each curve, shared by the panels and the legend so the two
# always match (the multinomial mean is drawn as markers only).
LINE_STYLE = {
    "mult_median":  dict(linestyle="-", linewidth=1.6),
    "gauss_mean":   dict(linestyle="-.", linewidth=2.2),
    "gauss_median": dict(linestyle="--", linewidth=1.6),
    "gauss_p10_p90": dict(linestyle=":", linewidth=2.2),
}

ASYMPTOTE_NTRAIN = (100, 1000)  # n_tr values for which E[V]/(N n_tr) is drawn

X_LIMITS = (0, 1.2e4)
Y_LIMITS = (3e-8, 2)
Y_MARGIN = 0.08  # y-range margin on each side, as a fraction of the panel's span (in decades)
MIN_DECADES_FOR_DECADE_TICKS = 1.5  # narrower panels get plain-decimal tick labels instead
PANEL_HPAD = 0.2  # vertical gap between panels (in units of the font size)
PANEL_WPAD = 0.4  # horizontal gap between panels
LEGEND_GAP = 0.005  # gap between the legend and the top row of panels (figure fraction)


def load_curves():
    curves = {}
    for n_train in NTRAIN_ORDER:
        path = nc.RESULTS_DIR / nc.results_filename(DIM, n_train, M_TEST, POVM_TYPE, N_OUT)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run noise_comparison.py with these same parameters first.")
        curves[n_train] = nc.load_results(path)
    return curves


def experiment_stats(mse_real):
    """`mse_real` has shape (n_reps, len(N_values), R): the MSE of every
    single experiment. Pools the experiments of all instances and returns
    their mean, the standard error of that mean, and the p10 / p50 (median)
    / p90 quantiles, each of shape (len(N_values),). Near the interpolation
    threshold (n_tr ~ n_out) the single-experiment MSE is heavy-tailed, so the
    mean is dominated by a few experiments while the median stays stable.
    """
    n_reps, n_N, R = mse_real.shape
    pooled = mse_real.transpose(0, 2, 1).reshape(n_reps * R, n_N)  # (experiments, len(N_values))
    mean = pooled.mean(axis=0)
    sem = pooled.std(axis=0, ddof=1) / np.sqrt(pooled.shape[0])
    p10, p50, p90 = np.percentile(pooled, [10, 50, 90], axis=0)
    return mean, sem, p10, p50, p90


def ntrain_tex(n_train):
    """n_tr for the panel label: powers of ten >= 100 as "10^k", else plain."""
    k = round(np.log10(n_train))
    return rf"10^{k}" if n_train >= 100 and 10**k == n_train else str(n_train)


def style_panel_y(ax, lo, hi):
    """Log y axis on [lo, hi]. Spans of at least `MIN_DECADES_FOR_DECADE_TICKS`
    decades get "$10^n$" labels at (at most about 6) decades; narrower spans
    get about 4 evenly spaced plain-decimal labels, so every panel has labeled
    ticks however small its range.
    """
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    if np.log10(hi / lo) >= MIN_DECADES_FOR_DECADE_TICKS:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=6))
        ax.yaxis.set_major_formatter(lambda val, pos: rf"$10^{{{int(round(np.log10(val)))}}}$")
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(lambda val, pos: rf"${val:g}$")
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10), numticks=15))
    ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    curves = load_curves()
    Eb2, EV = nc.theoretical_bias2_variance(DIM)

    ncols = min(NCOLS, len(NTRAIN_ORDER))
    nrows = -(-len(NTRAIN_ORDER) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    x_ticks = [10, 100, 1000, 1e4, 1e5]
    x_labels = [rf"$10^{{{i}}}$" for i in range(1, 6)]

    for k, (ax, n_train) in enumerate(zip(axes, NTRAIN_ORDER)):
        data = curves[n_train]
        N = data["N_values"].astype(float)
        stats = {model: experiment_stats(data["results"][model]["mse_real"]) for model in ("mult", "gauss")}
        mean, sem, p10, p50, p90 = stats["mult"]
        g_mean, _, g_p10, g_p50, g_p90 = stats["gauss"]
        color, shape, size = PANEL_STYLES[k % len(PANEL_STYLES)]
        m = styled(shape)

        ax.fill_between(N, p10, p90, color=color, alpha=0.25, linewidth=1.2, zorder=2)
        ax.errorbar(N, mean, yerr=sem, color=color, linestyle="None", linewidth=2.2, capsize=2, zorder=3)
        ax.plot(N, mean, marker=m, linestyle="None", zorder=4, **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH))
        ax.plot(N, mean, marker=m, linestyle="None", zorder=5, **marker_outer_style(color, size=size * 15 / 35))
        ax.plot(N, p50, color=color, zorder=3, **LINE_STYLE["mult_median"])
        ax.plot(N, g_mean, color=color, zorder=6, **LINE_STYLE["gauss_mean"])
        ax.plot(N, g_p50, color=color, zorder=6, **LINE_STYLE["gauss_median"])
        for q in (g_p10, g_p90):
            ax.plot(N, q, color=color, zorder=6, **LINE_STYLE["gauss_p10_p90"])

        if POVM_TYPE == "mub" and n_train in ASYMPTOTE_NTRAIN:  # the asymptotes hold for the MUB POVM only
            # Bias asymptote up to, and variance asymptote from, the N at which
            # the variance term overtakes the bias term.
            N_cross = Eb2 * n_train / EV
            interval = np.logspace(np.log10(X_LIMITS[0]), np.log10(N_cross), 100)
            ax.plot(interval, Eb2 / (interval + DIM * (DIM + 2)) ** 2, color="r", linestyle="-.", linewidth=2, zorder=1)
            interval = np.logspace(np.log10(N_cross), np.log10(X_LIMITS[1]), 100)
            ax.plot(interval, EV / (interval * n_train), color="k", linestyle="--", linewidth=2, zorder=1)

        # Each panel's own y range: the bands of both models, with a margin.
        lows = np.concatenate([p10, g_p10, mean - sem])
        lo = lows[lows > 0].min()  # mean - sem can be <= 0 when a few experiments dominate the mean
        hi = min(.2,max(p90.max(), g_p90.max(), (mean + sem).max()))
        margin = (hi / lo) ** Y_MARGIN
        style_panel_y(ax, lo / margin, hi * margin)

        ax.set_xscale("log")
        ax.set_xlim(*X_LIMITS)
        #ax.set_yscale("log")
        #ax.set_ylim(*Y_LIMITS)
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)] if val in x_ticks else "")
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.text(
            0.97, 0.94, rf"$n_{{\mathrm{{tr}}}} = {ntrain_tex(n_train)}$", transform=ax.transAxes,
            ha="right", va="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.9),
        )
        if k // ncols == nrows - 1:
            ax.set_xlabel(r"$N$")
        if k % ncols == 0:
            ax.set_ylabel(r"MSE")

    for ax in axes[len(NTRAIN_ORDER):]:
        ax.set_visible(False)

    m = styled(circle)
    mult_proxy = (
        Line2D([], [], linestyle="None", marker=m, **marker_inner_style("0.35", size=11, edgewidth=EDGEWIDTH)),
        Line2D([], [], linestyle="None", marker=m, **marker_outer_style("0.35", size=11 * 15 / 35)),
    )
    median_proxy = Line2D([], [], color="0.35", **LINE_STYLE["mult_median"])
    gauss_proxy = Line2D([], [], color="0.35", **LINE_STYLE["gauss_mean"])
    gauss_median_proxy = Line2D([], [], color="0.35", **LINE_STYLE["gauss_median"])
    # One row; the shaded band (multinomial p10-p90) and the dotted lines
    # (Gaussian p10, p90) are explained in the caption.
    legend = fig.legend(
        [mult_proxy, median_proxy, gauss_proxy, gauss_median_proxy],
        [r"multinomial: mean", r"multinomial: median", r"Gaussian: mean", r"Gaussian: median"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.2,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    # Panels start just below the legend (its measured height plus LEGEND_GAP);
    # small gaps between panels, while tight_layout still leaves room for each
    # panel's own y tick labels.
    fig.canvas.draw()
    legend_height = legend.get_window_extent().transformed(fig.transFigure.inverted()).height
    fig.tight_layout(rect=(0, 0, 1, 1 - legend_height - LEGEND_GAP), h_pad=PANEL_HPAD, w_pad=PANEL_WPAD)
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
