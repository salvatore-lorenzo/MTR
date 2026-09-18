"""Reproduce the Figure 3 plot (MSE vs N, for n_tr = 10^2, 10^3, 10^4) from
the ``MSE_vs_N_MUB_d=..._ntrain=..._.txt`` files produced by
``fig3_N_scaling.py`` in this same folder.

Same visual style as ``ntrain_saturation/plot_ntrain_saturation.py``: for
each n_tr, the median MSE (p50) is plotted against the training shot
budget N on a log-log scale, with a shaded band spanning the [p10, p90]
quantiles, and the same ring-and-dot markers from ``markers.py``.
Run this script from anywhere; it locates its data files next to itself.
"""

import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FixedLocator, LogLocator
from sympy import gamma

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_mse_data  # noqa: E402
from markers import diamond, square, triangle, circle, marker_inner_style, marker_outer_style, styled  # noqa: E402
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb}"
plt.rcParams.update(
    {
        "text.usetex": True,
        #"text.latex.preamble": r"\usepackage{amsmath}",
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
OUTPUT_PATH = DATA_DIR / "N_scaling_plot.pdf"

# n_tr -> (color, shape, legend label), matching the reference figure.
# `square` (from markers.py) renders as a diamond and `diamond` renders
# as a square — see the note in markers.py.
STYLE = {
    100:   ("#8FB032", square, r"$n_{\mathrm{tr}} = 10^2$"),  # green, diamond
    1000:  ("#5E81B5", diamond, r"$n_{\mathrm{tr}} = 10^3$"),  # blue, square
    10000: ("#E19C24", triangle, r"$n_{\mathrm{tr}} = 10^4$"),  # orange, triangle
    100000: ("#EB6235", circle, r"$n_{\mathrm{tr}} = 10^5$"),  # orange, triangle
}

MARKERSIZE = [11, 11, 13]
EDGEWIDTH = 1.3

# Legend / z-order, top to bottom as in the reference figure.
NTRAIN_ORDER = (100, 1000, 10000)

Y_LIMITS = (1e-10, 0.1)


def load_curves():
    curves = {}
    for path in DATA_DIR.glob("MSE_vs_N_MUB_d=*_.txt"):
        match = re.search(r"ntrain=(\d+)", path.name)
        n_train = int(match.group(1))
        stat_list, res_mse = load_mse_data(path)
        curves[n_train] = (stat_list, res_mse)
    return curves


def main():
    curves = load_curves()
    missing = [n for n in NTRAIN_ORDER if n not in curves]
    if missing:
        raise FileNotFoundError(f"no MSE_vs_N_MUB_d=..._ntrain=..._.txt file found for ntrain={missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []

    for n,n_train in enumerate(NTRAIN_ORDER):
        stat_list, res_mse = curves[n_train]
        p10, p50, p90 = res_mse[:, 0], res_mse[:, 1], res_mse[:, 2]
        color, shape, label = STYLE[n_train]
        m = styled(shape)

        ax.fill_between(stat_list, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(stat_list, p50, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            stat_list, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=MARKERSIZE[n], edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            stat_list, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=MARKERSIZE[n] * 15 / 35),
        )
        d=2
        Eb2=((d-1)*(d+2)**2/(d+1))
        EV=(d*(d-1)*(d+2)/(d+1))

        interval=np.linspace(2*1e3, 5*1e6, 100)
        (dashed_artist,) = ax.plot(interval,EV/(interval*n_train), color='k', linestyle='--', linewidth=2, zorder=3)
        interval=np.linspace(2*1e0, 5*1e3, 100)
        (dashdot_line,) = ax.plot(interval, Eb2/interval**2, 
                                  color='r', linestyle='-.',linewidth=2, zorder=3)

        if n == 0:
            dashed_handle, dashdot_handle = dashed_artist, dashdot_line

        legend_handles.append((line, inner, outer))
        legend_labels.append(label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*Y_LIMITS)

    y_ticks = [1e-1, 1e-4, 1e-7, 1e-10]
    y_labels = [r"$0.1$", r"$10^{-4}$", r"$10^{-7}$", r"$10^{-10}$"]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(lambda val, pos: y_labels[y_ticks.index(val)])
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [10, 100, 1000, 1e4, 1e5, 1e6]
    x_labels = [rf"$10^{{{i}}}$" for i in range(1, 7)]
    ax.set_xlim(2, 2e6)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ntrain_legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    ax.add_artist(ntrain_legend)

    ax.legend(
        [dashdot_handle,dashed_handle],
        #[r"$b^2/N^2$",r"$V/(Nn_{tr})$"],
        [r"$\mathbb{E}_{\sigma,\mathcal{O}}[b^2]/N^2$",r"$\mathbb{E}_{\sigma,\mathcal{O}}[V]/(Nn_{tr})$"],
        loc=[0.06,0.13],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
