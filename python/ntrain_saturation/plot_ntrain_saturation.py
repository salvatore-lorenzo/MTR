"""Plot Fig. 3 (MSE vs n_tr, for d = 2, 3, 5, 7) from the files in
``data/`` produced by ``ntrain_saturation.py``.

For each dimension, the median MSE (p50) is plotted against n_tr on a
log-log scale, with a shaded band spanning the [p10, p90] quantiles,
together with the large-n_tr bias floor E[b^2]/(N + d(d+2))^2.
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

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb, mathpazo, bm}"
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "ntrain_saturation.pdf"

N = 100  # training shots per state, as in ntrain_saturation.py

# Dimension -> (color, shape, markersize, legend label), in legend order
# (top to bottom). `square` (from markers.py) renders as a diamond and
# `diamond` renders as a square.
STYLE = {
    7: ("#EB6235", circle, 9.5, r"$d = 7$"),  # red, circle
    5: ("#E19C24", triangle, 13, r"$d = 5$"),  # orange, triangle
    3: ("#5E81B5", diamond, 11, r"$d = 3$"),  # blue, square
    2: ("#8FB032", square, 11, r"$d = 2$"),  # green, diamond
}
EDGEWIDTH = 1.3

# Leading grid points not shown (n_tr too small for the regression to be
# well posed).
SKIP = {2: 3, 3: 2, 5: 2, 7: 2}

Y_LIMITS = (3e-4, 0.025)


def load_curves():
    curves = {}
    for path in DATA_DIR.glob("MSE_vs_ntrain_MUB_d=*.txt"):
        dim = int(re.search(r"d=(\d+)", path.name).group(1))
        curves[dim] = load_mse_data(path)
    return curves


def main():
    curves = load_curves()
    missing = [d for d in STYLE if d not in curves]
    if missing:
        raise FileNotFoundError(f"no data file in {DATA_DIR} for d={missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []
    for dim, (color, shape, size, label) in STYLE.items():
        n_tr, res_mse = curves[dim]
        k = SKIP[dim]
        n_tr = n_tr[k:]
        p10, p50, p90 = res_mse[k:, 0], res_mse[k:, 1], res_mse[k:, 2]
        m = styled(shape)

        ax.fill_between(n_tr, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(n_tr, p50, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=size * 15 / 35),
        )
        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

        Eb2 = (dim - 1) * (dim + 2) ** 2 / (dim + 1)
        floor = ax.hlines(Eb2 / (N + dim * (dim + 2)) ** 2, 4e2, 5e6, color="k", linestyle="--", linewidth=2, zorder=3)

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

    dim_legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.7, 0.54],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )
    ax.add_artist(dim_legend)
    ax.legend(
        [floor],
        [r"$\mathbb{E}_{\sigma,\mathcal{O}}[b^2]/(N{+}d(d{+}2))^2$"],
        loc=[0.13, 0.83],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
