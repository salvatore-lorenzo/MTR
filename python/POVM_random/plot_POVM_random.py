"""Plot Figs. SM2 (M = inf) and SM3 (M = 1000): MSE vs N for random POVMs
with n_out = 8, 16, 32, 64, from the files in ``data/`` produced by
``POVM_random.py``.

For each n_out, the median MSE (p50) is plotted against the training shot
budget N on a log-log scale, with a shaded band spanning the [p10, p90]
quantiles. For M = inf the bias and variance asymptotes are shown; for
M = 1000 the shot-noise floor Var(w_O)/M = 2/(3M).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FixedLocator, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_mse_data  # noqa: E402
from markers import diamond, hexagon, marker_inner_style, marker_outer_style, square, styled, triangle  # noqa: E402

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

# Run parameters, as in POVM_random.py.
D = 2
N_TRAIN, N_TEST, N_OBS = 256, 200, 100

# n_out -> (color, shape, markersize, legend label). `square` (from
# markers.py) renders as a diamond and `diamond` renders as a square.
STYLE = {
    8: ("#8FB032", square, 13, r"$n_{\mathrm{out}} = 8$"),  # green, diamond
    16: ("#5E81B5", diamond, 13, r"$n_{\mathrm{out}} = 16$"),  # blue, square
    32: ("#E19C24", triangle, 12, r"$n_{\mathrm{out}} = 32$"),  # orange, triangle
    64: ("#A64CB8", hexagon, 11, r"$n_{\mathrm{out}} = 64$"),  # purple, hexagon
}
EDGEWIDTH = 1.3

# Per-figure settings: test shots M -> (output file, x limits (None =
# autoscale), y limits, n_out legend position).
FIGURES = {
    "inf": ("POVM_random_M=inf.pdf", None, (1.6e-8, 3e-1), [0.02, 0.02]),
    1000: ("POVM_random_M=1000.pdf", (2, 1.3e5), (5e-4, 6e-2), [0.62, 0.505]),
}


def data_path(n_out, M):
    return DATA_DIR / (
        f"MSE_vs_N_POVM_d={D}_nout={n_out}_M={M}_ntrain={N_TRAIN}_ntest={N_TEST}_nobs={N_OBS}.txt"
    )


def plot(M, output_name, x_limits, y_limits, nout_legend_loc):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []
    for n_out, (color, shape, size, label) in STYLE.items():
        stat_list, res_mse = load_mse_data(data_path(n_out, M))
        p10, p50, p90 = res_mse[:, 0], res_mse[:, 1], res_mse[:, 2]
        m = styled(shape)

        ax.fill_between(stat_list, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(stat_list, p50, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            stat_list, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            stat_list, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=size * 15 / 35),
        )
        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

    if M == "inf":
        # Small-N bias and large-N variance asymptotes for a random qubit POVM.
        N_small = np.linspace(2, 5e3, 100)
        (bias_line,) = ax.plot(N_small, 16 / 3 / N_small**2, color="r", linestyle="-.", linewidth=2, zorder=10)
        N_large = np.linspace(2e2, 2e5, 100)
        variance = 2 / 3 / (N_large * N_TRAIN) * (1 + 3 * (N_large / (N_large + 8)) ** 2)
        (variance_line,) = ax.plot(N_large, variance, color="k", linestyle="--", linewidth=2, zorder=10)
        theory_handles = [bias_line, variance_line]
        theory_labels = [r"$\mathbb{E}_{\sigma,\mathcal{O}}[b^2]/N^2$", r"$\mathbb{E}_{\sigma,\mathcal{O}}[V]/(Nn_{tr})$"]
        theory_legend_loc = [0.57, 0.7]
    else:
        floor_line = ax.hlines(2 / 3 / M, 1e2, 1.3e5, color="k", linestyle="--", linewidth=2, zorder=10)
        theory_handles = [floor_line]
        theory_labels = [r"Var$(\boldsymbol{w}_{\mathcal{O}})/M$"]
        theory_legend_loc = [0.25, 0.8]

    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [10, 100, 1000, 1e4, 1e5]
    x_labels = [rf"$10^{{{i}}}$" for i in range(1, 6)]
    ax.set_xscale("log")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    nout_legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=nout_legend_loc,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )
    ax.add_artist(nout_legend)
    ax.legend(
        theory_handles,
        theory_labels,
        loc=theory_legend_loc,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    output_path = BASE_DIR / output_name
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


def main():
    for M, settings in FIGURES.items():
        plot(M, *settings)


if __name__ == "__main__":
    main()
