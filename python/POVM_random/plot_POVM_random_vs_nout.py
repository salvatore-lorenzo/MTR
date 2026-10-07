"""Plot Figs. SM5 (M = inf) and SM6 (M = 1000): MSE vs the number of POVM
outcomes n_out, at fixed training shot budgets N = 10^2, ..., 10^5, from
the files in ``data/`` produced by ``POVM_random.py``.

This is the transpose of ``plot_POVM_random.py``: that script fixes n_out
and shows MSE vs N; this one fixes N and shows MSE vs n_out (one curve
per N), with the median as markers and a shaded [p10, p90] band.
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
from markers import circle, diamond, marker_inner_style, marker_outer_style, pentagon, styled, triangle  # noqa: E402

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

# Training shot budget N -> (color, shape, markersize, legend label), one
# curve each, in legend order. `diamond` (from markers.py) renders as a square.
STYLE = {
    100: ("#5E81B5", diamond, 11, r"$N = 10^2$"),  # blue, square
    1000: ("#E19C24", triangle, 13, r"$N = 10^3$"),  # orange, triangle
    10000: ("#EB6235", circle, 11, r"$N = 10^4$"),  # red, circle
    100000: ("#2FA672", pentagon, 11, r"$N = 10^5$"),  # teal, pentagon
}
EDGEWIDTH = 1.3

# Per-figure settings: test shots M -> (output file, y limits).
FIGURES = {
    "inf": ("POVM_random_vs_nout_M=inf.pdf", (5e-8, 1e2)),
    1000: ("POVM_random_vs_nout_M=1000.pdf", (5e-4, 1e2)),
}


def load_curves(M):
    """{n_out: (stat_list, res_mse)} for all n_out values run at test shots M."""
    curves = {}
    pattern = f"MSE_vs_N_POVM_d={D}_nout=*_M={M}_ntrain={N_TRAIN}_ntest={N_TEST}_nobs={N_OBS}.txt"
    for path in DATA_DIR.glob(pattern):
        n_out = int(re.search(r"nout=(\d+)", path.name).group(1))
        curves[n_out] = load_mse_data(path)
    if not curves:
        raise FileNotFoundError(f"no {pattern} file found in {DATA_DIR}")
    return curves


def plot(M, output_name, y_limits):
    curves = load_curves(M)
    nout_list = sorted(curves)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []
    for N, (color, shape, size, label) in STYLE.items():
        # Row of each n_out's quantile table at this N.
        rows = np.array([curves[n_out][1][list(curves[n_out][0]).index(N)] for n_out in nout_list])
        p10, p50, p90 = rows[:, 0], rows[:, 1], rows[:, 2]
        m = styled(shape)

        ax.fill_between(nout_list, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(nout_list, p50, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            nout_list, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            nout_list, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=size * 15 / 35),
        )
        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{out}}$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*y_limits)

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [4, 8, 16, 32, 64, 128, 256]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: str(int(val)))
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.15, 0.65],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        ncols=2,
    )

    fig.tight_layout()
    output_path = BASE_DIR / output_name
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


def main():
    for M, (output_name, y_limits) in FIGURES.items():
        plot(M, output_name, y_limits)


if __name__ == "__main__":
    main()
