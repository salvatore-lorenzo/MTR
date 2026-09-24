"""Reproduce the Figure 5 plot (MSE vs N, for M = 10^2, 10^3, 10^4, 10^5)
from the ``MSE_vs_M_MUB_d=..._.txt`` files produced by
``fig5_M_saturation.py`` in this same folder.

Same visual style as ``ntrain_saturation/plot_ntrain_saturation.py`` and
``M_inf/plot_N_scaling.py``: for each (fixed) test shot budget M, the
median MSE (p50) is plotted against the training shot budget N on a
log-log scale, with a shaded band spanning the [p10, p90] quantiles, and
the same ring-and-dot markers from ``markers.py``.

Note: in the original notebook/paper this figure's x-axis is labeled M,
but the quantity actually swept along the x-axis in the exported data is
the training shot budget N, with M held fixed per curve/file (its value
is recorded, mislabeled "N=", in each filename) — so it is labeled $N$
here to match what is actually plotted.

Run this script from anywhere; it locates its data files next to itself.
"""

import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FixedLocator, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_mse_data  # noqa: E402
from markers import circle, diamond, marker_inner_style, marker_outer_style, square, styled, triangle  # noqa: E402
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
OUTPUT_PATH = DATA_DIR / "M_saturation_plot.pdf"

# M -> (color, shape, legend label), matching the reference figure.
# `square` (from markers.py) renders as a diamond and `diamond` renders
# as a square — see the note in markers.py.
STYLE = {
    100: ("#8FB032", circle, r"$M = 10^2$"),  # green, circle
    1000: ("#5E81B5", triangle, r"$M = 10^3$"),  # blue, triangle
    10000: ("#E19C24", diamond, r"$M = 10^4$"),  # orange, square
    100000: ("#EB6235", square, r"$M = 10^5$"),  # red, diamond
}

MARKERSIZE = [9.5, 13, 11, 11]
EDGEWIDTH = 1.3

# Legend / z-order, top to bottom as in the reference figure.
M_ORDER = (100, 1000, 10000, 100000)

Y_LIMITS = (3e-6, 0.08)


def load_curves():
    curves = {}
    for path in DATA_DIR.glob("MSE_vs_N_MUB_d=*_ntrain=100_*_.txt"):
        match = re.search(r"_M=(\d+)", path.name)
        m_value = int(match.group(1))
        stat_list, res_mse = load_mse_data(path)
        curves[m_value] = (stat_list, res_mse)
    return curves


def main():
    curves = load_curves()
    missing = [m for m in M_ORDER if m not in curves]
    if missing:
        raise FileNotFoundError(f"no MSE_vs_N_MUB_d=..._.txt file found for M={missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []

    for n, m_value in enumerate(M_ORDER):
        stat_list, res_mse = curves[m_value]
        p10, p50, p90 = res_mse[:, 0], res_mse[:, 1], res_mse[:, 2]
        color, shape, label = STYLE[m_value]
        m = styled(shape)

        ax.fill_between(stat_list, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(stat_list, p50, color=color,linestyle='-', linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            stat_list, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=MARKERSIZE[n], edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            stat_list, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=MARKERSIZE[n] * 15 / 35),
        )
        d=2
        Varwo=((d-1)*(d+2)/(d*(d+1)))
        dashed_artist = ax.hlines(Varwo/m_value, 2*1e2, 5*1e6, color='k', linestyle='--', linewidth=2, zorder=3)
        
        if n == 0:
            dashed_handle = dashed_artist

        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*Y_LIMITS)

    y_ticks = [1e-2, 1e-3, 1e-4, 1e-5]
    y_labels = [r"$0.010$", r"$0.001$", r"$10^{-4}$", r"$10^{-5}$"]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(lambda val, pos: y_labels[y_ticks.index(val)])
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [10, 100, 1000, 10000,100000]
    x_labels = [rf"$10^{{{i}}}$" for i in range(1, 6)]
    ax.set_xlim(1.5, 5.e5)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    M_legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.03, 0.07],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )
    ax.add_artist(M_legend)

    ax.legend(
        [dashed_handle],
        [r"Var$(\boldsymbol{w}_{\mathcal{O}})/M$"],
        loc=[0.62,0.83],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
