"""Reproduce the Figure 4 plot (MSE vs n_tr, for d = 2, 3, 5, 7) from the
``MSE_vs_ntrain_MUB_d=*_.txt`` files produced by
``fig4_ntrain_saturation.py`` in this same folder.

For each dimension, the median MSE (p50) is plotted against n_tr on a
log-log scale, with a shaded band spanning the [p10, p90] quantiles.
Run this script from anywhere; it locates its data files next to itself.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FixedLocator, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_mse_data  # noqa: E402
from markers import circle, diamond, marker_inner_style, marker_outer_style, square, styled, triangle  # noqa: E402
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb}"

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
OUTPUT_PATH = DATA_DIR / "ntrain_saturation_plot.pdf"

# Dimension -> (color, shape), matching the reference figure. Shapes come
# from markers.py: `square` renders as a diamond and `diamond` renders as
# a square (see the note in that file), so they're deliberately swapped
# here to get the intended look.
STYLE = {
    2: ("#8FB032", square,  r"$d = 2$"),  # green, diamond
    3: ("#5E81B5", diamond, r"$d = 3$"),  # blue, square
    5: ("#E19C24", triangle,r"$d = 5$"),  # orange, triangle
    7: ("#EB6235", circle,  r"$d = 7$"  ),  # red, circle
}

MARKERSIZE = [9.5, 13, 11, 11]
EDGEWIDTH = 1.3

# Legend / z-order, top to bottom as in the reference figure.
DIM_ORDER = (7, 5, 3, 2)
N = 100
Y_LIMITS = (4e-4, 0.025)


def load_curves():
    curves = {}
    for path in DATA_DIR.glob("MSE_vs_ntrain_MUB_d=*_.txt"):
        match = re.search(r"d=(\d+)", path.name)
        dim = int(match.group(1))
        n_tr, res_mse = load_mse_data(path)
        curves[dim] = (n_tr, res_mse)
    return curves


def main():
    curves = load_curves()
    missing = [d for d in DIM_ORDER if d not in curves]
    if missing:
        raise FileNotFoundError(f"no MSE_vs_ntrain_MUB_d=..._.txt file found for d={missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []
    dashed_handle, dashdot_handle = None, None

    for d,dim in enumerate(DIM_ORDER):
        if dim==2:
            kk = 3  # skip the first entry, which is n_tr=0
        else:
            kk = 2
        n_tr, res_mse = curves[dim]
        n_tr = n_tr[kk:]  # skip the first entry, which is n_tr=0
        p10, p50, p90 = res_mse[kk:, 0], res_mse[kk:, 1], res_mse[kk:, 2]
        color, shape, label = STYLE[dim]
        m = styled(shape)

        ax.fill_between(n_tr, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(n_tr, p50, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=MARKERSIZE[d], edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=MARKERSIZE[d]  * 15 / 35),
        )

        
        Eb2=((dim-1)*(dim+2)**2/(dim+1))
        EV=(dim*(dim-1)*(dim+2)/(dim+1))

        dashed_artist = ax.hlines(Eb2/(N+dim*(dim+2))**2, 4*1e2, 5*1e6, color='k', linestyle='--', linewidth=2, zorder=3)
        
        if d == 0:
            dashed_handle = dashed_artist

        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{tr}}$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*Y_LIMITS)

    y_ticks = [5e-4, 1e-3, 5e-3, 1e-2]
    y_labels = [r"$0.0005$", r"$0.001$", r"$0.005$", r"$0.010$"]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(lambda val, pos: y_labels[y_ticks.index(val)])
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [10, 100, 1000, 10000]
    x_labels = [r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"]
    ax.set_xlim(6, 2.2e4)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(int(val))])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    dim_legend =ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.7,0.55],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    ax.add_artist(dim_legend)
    
    ax.legend(
        [dashed_handle],
        [r"$\mathbb{E}_{\sigma,\mathcal{O}}[b^2]/(N{+}d(d{+}2))^2$"],
        loc=[0.13,0.815],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
