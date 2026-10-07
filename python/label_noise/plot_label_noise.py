"""Plot Fig. 5 (MSE vs N, for training label noise gamma = 0.0, 0.1, 0.2)
from the files in ``data/`` produced by ``label_noise.py``.

For each gamma, the median MSE (p50) is plotted against the training shot
budget N on a log-log scale, with a shaded band spanning the [p10, p90]
quantiles, together with the large-N floor Var(w_O)/M and the label-noise
amplification gamma^2 N C / M. For gamma > 0 the curves dip and then rise
again at large N.
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
from markers import diamond, marker_inner_style, marker_outer_style, square, styled, triangle  # noqa: E402

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
OUTPUT_PATH = BASE_DIR / "label_noise.pdf"

# Run parameters, as in label_noise.py (qubit, MUB measurement).
D = 2
N_OUT = D * (D + 1)
N_TRAIN = 100
M = 100

# gamma -> (color, shape, markersize, legend label). `square` (from
# markers.py) renders as a diamond and `diamond` renders as a square.
STYLE = {
    0.0: ("#8FB032", square, 11, r"$\gamma = 0.0$"),  # green, diamond
    0.1: ("#5E81B5", diamond, 11, r"$\gamma = 0.1$"),  # blue, square
    0.2: ("#E19C24", triangle, 13, r"$\gamma = 0.2$"),  # orange, triangle
}
EDGEWIDTH = 1.3

Y_LIMITS = (0.004, 6.0)


def load_curves():
    curves = {}
    for path in DATA_DIR.glob(f"MSE_vs_N_MUB_d={D}_gamma_train=*_M={M}_*.txt"):
        gamma = float(re.search(r"gamma_train=([0-9.]+)_", path.name).group(1))
        curves[gamma] = load_mse_data(path)
    return curves


def main():
    curves = load_curves()
    missing = [g for g in STYLE if g not in curves]
    if missing:
        raise FileNotFoundError(f"no data file in {DATA_DIR} for gamma={missing}")

    var_wO = (D - 1) * (D + 2) / (D * (D + 1))
    C = (N_OUT - D**2) / (N_TRAIN - N_OUT - 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []
    for gamma, (color, shape, size, label) in STYLE.items():
        stat_list, res_mse = curves[gamma]
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

        if gamma > 0:
            N_large = np.linspace(8e3, 5e6, 100)
            (amplification_line,) = ax.plot(
                N_large, gamma**2 * N_large * C / M, color="r", linestyle="-.", linewidth=2, zorder=3
            )

    floor_line = ax.hlines(var_wO / M, 100, 2e6, color="k", linestyle="--", linewidth=2, zorder=3)

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
    ax.tick_params(which="both", direction="in", top=True, right=True, pad=7)

    gamma_legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.09, 0.64],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )
    ax.add_artist(gamma_legend)
    ax.legend(
        [floor_line, amplification_line],
        [r"Var$(\boldsymbol{w}_{\mathcal{O}})/M$", r"$\gamma^2 N C/ M$"],
        loc=[0.09, 0.36],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
