"""MSE vs n_out (POVM outcome number), for various fixed training shot
budgets N, from the ``MSE_vs_N_POVM_d=2..._M=1000_..._.txt`` files
produced by ``fig8_POVM_random_M.py`` in this same folder.

This is the transpose of ``plot_POVM_random_M.py``: that script fixes
n_out and shows MSE vs N (one curve per n_out); this one fixes N (see
``N_VALUES``) and shows MSE vs n_out (one curve per N), reusing the same
``RUN_PARAMS`` disambiguation since this folder can hold several M=1000
runs that differ only in ntrain/ntest/nobs.

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
from markers import circle, diamond, marker_inner_style, marker_outer_style, pentagon, square, styled, triangle  # noqa: E402
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb,mathpazo,bm}"

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
OUTPUT_PATH = DATA_DIR / "POVM_random_M_vs_nout_plot.pdf"

STAT_SIGMA = 1000 #"inf" #1000

# Which fig8_POVM_random_M.py run to plot (must match that script's
# N_TRAIN / N_TEST / N_OBS at the time it produced the data files) --
# update this to switch between coexisting runs.
RUN_PARAMS = {"ntrain": 256, "ntest": 200, "nobs": 100}

FILENAME_RE = re.compile(
    r"MSE_vs_N_POVM_d=(?P<d>\d+)_nout=(?P<nout>\d+)_M=(?P<M>[^_]+)"
    r"_ntrain=(?P<ntrain>\d+)_ntest=(?P<ntest>\d+)_nobs=(?P<nobs>\d+)_\.txt$"
)

# Fixed training shot budgets N to show, one curve each, in legend order.
N_VALUES = (100, 1000, 10000, 100000)

STYLE = {
    10: ("#8FB032", square, r"$N = 10$"),  # green
    100: ("#5E81B5", diamond, r"$N = 10^2$"),  # blue
    1000: ("#E19C24", triangle, r"$N = 10^3$"),  # orange
    10000: ("#EB6235", circle, r"$N = 10^4$"),  # red
    100000: ("#2FA672", pentagon, r"$N = 10^5$"),  # teal
}

MARKERSIZE = [ 11, 13, 11, 11]
EDGEWIDTH = 1.3

if STAT_SIGMA==1000:
    Y_LIMITS = (5e-4, 1e2)
else:
    Y_LIMITS = (5e-8, 1e2)


def load_curves():
    """Load {n_out: (stat_list, res_mse)} for M=1000 files matching
    `RUN_PARAMS`, plus the other (ntrain, ntest, nobs) tuples seen for
    files that match on n_out/M but not on `RUN_PARAMS`.
    """
    curves = {}
    other_runs = {}
    for path in DATA_DIR.glob(f"MSE_vs_N_POVM_d=*_M={STAT_SIGMA}_*_.txt"):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        n_out = int(match["nout"])
        params = {
            "ntrain": int(match["ntrain"]),
            "ntest": int(match["ntest"]),
            "nobs": int(match["nobs"]),
        }
        if params == RUN_PARAMS:
            curves[n_out] = load_mse_data(path)
        else:
            other_runs.setdefault(n_out, set()).add(tuple(sorted(params.items())))
    return curves, other_runs


def main():
    curves, other_runs = load_curves()
    if not curves:
        raise FileNotFoundError(
            f"no MSE_vs_N_POVM_d=..._M={STAT_SIGMA}_..._.txt file matching "
            f"RUN_PARAMS={RUN_PARAMS} found in {DATA_DIR}"
        )
    for n_out, param_sets in other_runs.items():
        for params in param_sets:
            print(f"note: ignoring other run for nout={n_out}: {dict(params)}")

    nout_list = sorted(curves)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []

    for n, N in enumerate(N_VALUES):
        xs, p10s, p50s, p90s = [], [], [], []
        for n_out in nout_list:
            stat_list, res_mse = curves[n_out]
            idx = np.where(stat_list == N)[0]
            if idx.size == 0:
                print(f"note: no N={N} entry for nout={n_out}, skipping that point")
                continue
            i = idx[0]
            xs.append(n_out)
            p10s.append(res_mse[i, 0])
            p50s.append(res_mse[i, 1])
            p90s.append(res_mse[i, 2])

        if not xs:
            print(f"note: no data at all for N={N}, skipping curve")
            continue

        color, shape, label = STYLE[N]
        m = styled(shape)

        ax.fill_between(xs, p10s, p90s, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(xs, p50s, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            xs, p50s, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=MARKERSIZE[n], edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            xs, p50s, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=MARKERSIZE[n] * 15 / 35),
        )

        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{out}}$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*Y_LIMITS)

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [4, 8, 16, 32, 64, 128, 256]
    #ax.set_xlim(3, 170)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: str(int(val)))
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.15,0.65],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        ncols=2
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
