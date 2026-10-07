"""Plot Fig. SM4 from the files in ``data/`` produced by ``check_fit.py``:
the empirical MSE(N) of the fixed regression instance for every (gamma,
gamma_test, M) combination, on the same axes. For each curve the figure
shows the mean over realizations (markers, with standard-error bars), the
10th-90th percentile band of the squared errors, and the fit f(N) (black
dashed line).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
import check_fit as cf  # noqa: E402
from markers import circle, diamond, hexagon, marker_inner_style, marker_outer_style, pentagon, square, styled, triangle  # noqa: E402

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb, mathpazo, bm}"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 17,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.0,
    }
)

OUTPUT_PATH = Path(__file__).resolve().parent / f"check_fit_d={cf.DIM}_nout={cf.N_OUT}_ntrain={cf.N_TRAIN}.pdf"

COLORS = ["#5E81B5", "#E19C24", "#8FB032", "#EB6235", "#A64CB8", "#3AA39B"]  # cycled over the curves
SHAPES = [circle, square, triangle, diamond, pentagon, hexagon]  # `square` looks like a diamond, see markers.py
MARKERSIZE = [9.5, 11, 13, 11, 11, 11]
EDGEWIDTH = 1.3

LEGEND_COLUMN_WIDTHS = ("2.4em", "2.8em", "3.6em")  # gamma_tr, gamma_test, M


def plot_one(ax, results, color, shape, size):
    """Draw the empirical MSE(N) and the fit f(N) of one results file."""
    N = results["N_values"].astype(float)
    sq = results["sq_errors"]
    mse = sq.mean(axis=0)
    sem = sq.std(axis=0, ddof=1) / np.sqrt(sq.shape[0])
    p10, p90 = np.percentile(sq, [10, 90], axis=0)
    N_dense = np.logspace(np.log10(N[0]), np.log10(N[-1]), 400)

    ax.plot(N_dense, cf.mse_fit(N_dense, results["coeffs"], results["dim"]), color="k", linestyle="--", linewidth=2.0, zorder=4)
    ax.fill_between(N, p10, p90, color=color, alpha=0.2, linewidth=0, zorder=1)

    m = styled(shape)
    ax.errorbar(N, mse, yerr=sem, color=color, linestyle="None", elinewidth=1.2, capsize=2, zorder=3)
    ax.plot(N, mse, color=color, linewidth=1.4, zorder=3)
    ax.plot(N, mse, marker=m, linestyle="None", zorder=5, **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH))
    ax.plot(N, mse, marker=m, linestyle="None", zorder=6, **marker_outer_style(color, size=size * 15 / 35))


def legend_row(*cells):
    """One row of the table-style legend: each cell centred in a fixed-width box."""
    return "".join(rf"\makebox[{w}]{{{c}}}" for w, c in zip(LEGEND_COLUMN_WIDTHS, cells))


def style_axes(ax, x_ticks):
    """Log-log axes with decade ticks, grid and inward ticks."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10), numticks=15))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("MSE")

    x_labels = [rf"$10^{{{int(np.log10(t))}}}$" for t in x_ticks]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)] if val in x_ticks else "")
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def main():
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    # The legend is a table: a header row with no marker, then one row of values per curve.
    handles = [Line2D([], [], linestyle="None")]
    labels = [legend_row(r"$\gamma_{\rm tr}$", r"$\gamma_{\rm test}$", r"$M$")]
    i = 0
    for gamma, gamma_test in cf.GAMMA_PAIRS:
        for M in cf.M_VALUES:
            path = cf.RESULTS_DIR / cf.results_filename(cf.DIM, cf.N_OUT, cf.N_TRAIN, M, gamma, gamma_test)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- run check_fit.py first.")
            results = cf.load_results(path)

            color, shape, size = COLORS[i % len(COLORS)], SHAPES[i % len(SHAPES)], MARKERSIZE[i % len(MARKERSIZE)]
            plot_one(ax, results, color, shape, size)
            i += 1

            m = styled(shape)
            handles.append((
                Line2D([], [], color=color, linewidth=1.4, marker=m, **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH)),
                Line2D([], [], linestyle="None", marker=m, **marker_outer_style(color, size=size * 15 / 35)),
            ))
            M_tex = r"\infty" if M is None else rf"10^{{{int(np.log10(M))}}}"
            labels.append(legend_row(f"${gamma}$", f"${gamma_test}$", f"${M_tex}$"))

    style_axes(ax, x_ticks=(10, 100, 1000, 10**4, 10**5, 10**6))
    ax.set_xlim(2, 1.2e6)
    ax.set_ylim(1e-8, 0.3)

    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=True, framealpha=0.9, edgecolor="0.8")

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
