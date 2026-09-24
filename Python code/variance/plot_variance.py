"""Reproduce the Figure 2 bottom panel (N n_tr Var vs n_tr, for d = 2, 3, 5,
7) from the ``NntrVar_vs_ntrain_MUB_d=*_.txt`` and ``V_vs_ntrain_MUB_d=*_.txt``
files produced by ``fig2_variance.py`` in this same folder.

For each dimension:
- solid line + markers + shaded band: median [p10, p90] of the empirical
  N n_tr Var_tr;
- dashed line + error bars: median [p10, p90] of the asymptotic
  coefficient V of eq. (47);
- dash-dotted horizontal line: the tight-design average
  E_{sigma,O}[V] = d(d-1)(d+2)/(d+1) of eq. (48).
Run this script from anywhere; it locates its data files next to itself.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
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
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.0,
    }
)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "variance_plot.pdf"

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

MARKERSIZE = {2: 11, 3: 13, 5: 13, 7: 9.5}
EDGEWIDTH = 1.3

# Legend order, left to right as in the reference figure.
DIM_ORDER = (2, 3, 5, 7)
X_LIMITS = (6, 4000)
Y_LIMITS = (1, 800)


def load_curves(stem):
    curves = {}
    for path in DATA_DIR.glob(f"{stem}_vs_ntrain_MUB_d=*_.txt"):
        dim = int(re.search(r"d=(\d+)", path.name).group(1))
        curves[dim] = load_mse_data(path)
    return curves


def main():
    var_curves = load_curves("NntrVar")
    V_curves = load_curves("V")
    missing = [d for d in DIM_ORDER if d not in var_curves or d not in V_curves]
    if missing:
        raise FileNotFoundError(f"no NntrVar/V _vs_ntrain_MUB_d=..._.txt files found for d={missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    legend_handles, legend_labels = [], []

    for dim in DIM_ORDER:
        color, shape, label = STYLE[dim]
        m = styled(shape)
        ms = MARKERSIZE[dim]

        # Empirical N n_tr Var: median line, [p10, p90] band, ring+dot markers.
        n_tr, res = var_curves[dim]
        p10, p50, p90 = res[:, 0], res[:, 1], res[:, 2]
        ax.fill_between(n_tr, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(n_tr, p50, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=ms, edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=ms * 15 / 35),
        )

        # Asymptotic coefficient V: dashed median with [p10, p90] error bars.
        n_tr_b, res_b = V_curves[dim]
        q10, q50, q90 = res_b[:, 0], res_b[:, 1], res_b[:, 2]
        ax.errorbar(
            n_tr_b, q50, yerr=[q50 - q10, q90 - q50], color=color, linestyle="--",
            linewidth=1.2, elinewidth=1.0, capsize=0, alpha=0.8, zorder=3,
        )
        ax.plot(
            n_tr_b, q50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=ms * 0.6, edgewidth=1.0),
        )

        # Tight-design average E_{sigma,O}[V], eq. (48).
        EV = dim * (dim - 1) * (dim + 2) / (dim + 1)
        ax.hlines(EV, n_tr[0], X_LIMITS[1], color=color, linestyle="-.", linewidth=1.4, zorder=3)

        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{tr}}$")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)

    y_ticks = [5, 10, 50, 100, 500]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(lambda val, pos: rf"${val:g}$")
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [10, 50, 100, 500, 1000]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: rf"${val:g}$")
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    dim_legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(DIM_ORDER),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    ax.add_artist(dim_legend)

    style_handles = [
        Line2D([], [], color="0.4", linestyle="-", linewidth=1.6),
        Line2D([], [], color="0.4", linestyle="--", linewidth=1.2),
        Line2D([], [], color="0.4", linestyle="-.", linewidth=1.4),
    ]
    style_labels = [
        r"$N n_{\mathrm{tr}}\,\mathrm{Var}_{\mathrm{tr}}$",
        r"$V$",
        r"$\mathbb{E}_{\sigma,\mathcal{O}}(V)$",
    ]
    ax.legend(style_handles, style_labels, 
              loc=[0.66,0.68], 
              frameon=True, 
              framealpha=0.9, 
              edgecolor="0.8",
              labelspacing=0.2,
)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", bbox_extra_artists=[dim_legend])
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
