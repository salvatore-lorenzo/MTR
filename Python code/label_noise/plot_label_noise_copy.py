"""Reproduce the Figure 6 plot (MSE vs N, for gamma = 0.0, 0.1, 0.2) from
the ``MSE_vs_N_MUB_d=..._gamma_train=..._copy.txt`` files produced by
``fig6_label_noise_copy.py`` in this same folder.

Same visual style as the other figures' plotting scripts: for each label
(target) noise level gamma, the median MSE (p50) is plotted against the
training shot budget N on a log-log scale, with a shaded band spanning
the [p10, p90] quantiles, and the same ring-and-dot markers from
``markers.py``. For gamma > 0, the curves dip and then rise again at
large N — the target-noise amplification effect discussed in the paper.

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
from markers import circle, diamond, square, triangle, marker_inner_style, marker_outer_style, styled  # noqa: E402
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
OUTPUT_PATH = DATA_DIR / "label_noise_plot_copy.pdf"

# gamma -> (color, shape, legend label), matching the reference figure.
# `diamond` (from markers.py) renders as a square — see the note in markers.py.
STYLE = {
    0.0: ("#8FB032", square,  r"$\gamma = 0.0$"),  # green, circle
    0.1: ("#5E81B5", diamond, r"$\gamma = 0.1$"),  # blue, triangle
    0.2: ("#E19C24", triangle,r"$\gamma = 0.2$"),  # orange, square
}

MARKERSIZE = [11, 11, 13]
EDGEWIDTH = 1.3

# Legend / z-order, top to bottom as in the reference figure.
GAMMA_ORDER = (0.0, 0.1, 0.2)

Y_LIMITS = (0.004, 6.0)


def load_curves():
    curves = {}
    for path in DATA_DIR.glob("MSE_vs_N_MUB_d=2_gamma_train*_M=100_*_copy.txt"):
        print(path.name)
        match = re.search(r"gamma_train=([0-9.]+)_", path.name)
        
        gamma = float(match.group(1))
        stat_list, res_mse = load_mse_data(path)
        curves[gamma] = (stat_list, res_mse)
    return curves


def main():
    curves = load_curves()
    missing = [g for g in GAMMA_ORDER if g not in curves]
    if missing:
        raise FileNotFoundError(f"no MSE_vs_N_MUB_d=..._gamma_train=..._M=100_..._copy.txt file found for gamma={missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []
    dashed_handle, dashdot_handle = None, None

    for n, gamma in enumerate(GAMMA_ORDER):
        stat_list, res_mse = curves[gamma]
        p10, p50, p90 = res_mse[:, 0], res_mse[:, 1], res_mse[:, 2]
        color, shape, label = STYLE[gamma]
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

        dim=2
        ntrain=100
        nout=6
        N=np.linspace(8*1e3, 5*1e6, 100)
        M=100
        Varwo=((dim-1)*(dim+2)/(dim*(dim+1)))

        dashed_artist = ax.hlines(Varwo/M, 100, 2000000, 
                                  color='k', linestyle='--', linewidth=2, zorder=3)


        R=(nout-dim**2)/(ntrain-nout-1)
        (dashdot_line,) = ax.plot(N,gamma**2*N/M*R, 
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

    y_ticks = [1e-3, 1e-2, 1e-1, 1]
    y_labels = [r"$0.001$", r"$0.010$", r"$0.100$", r"$1$"]
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
    ax.tick_params(which="both", direction="in", top=True, right=True,pad=7)

    gamma_legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.09,0.64],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )
    ax.add_artist(gamma_legend)

    ax.legend(
        [dashed_handle, dashdot_handle],
        [r"Var$(\boldsymbol{w}_{\mathcal{O}})/M$", r"$\gamma^2 N C/ M$"],
        loc=[0.09,0.36],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
