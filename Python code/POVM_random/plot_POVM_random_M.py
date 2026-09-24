"""Reproduce the Figure 8 plot (MSE vs N, for n_out = 4, 8, 16, 32, 64, 128)
from the ``MSE_vs_N_POVM_d=..._M=1000_..._.txt`` files produced by
``fig8_POVM_random_M.py`` in this same folder.

Same as ``plot_POVM_random.py`` (Figure 7), except the test-side
probabilities were estimated from ``M = 1000`` shots per test state
instead of being exact, so this only picks up the ``M=1000`` data files
and leaves the ``M=inf`` ones (Figure 7) alone.

``fig8_POVM_random_M.py`` has been run several times with different
``N_TRAIN``/``N_TEST``/``N_OBS`` values, so this folder can hold several
``M=1000`` files for the same ``nout`` that differ only in those fields.
``RUN_PARAMS`` below pins down exactly which run to plot; `load_curves`
only matches files whose ``ntrain``/``ntest``/``nobs`` agree with it, and
`main` reports any other run parameter sets it finds on disk so a stale
choice doesn't silently plot the wrong (or a mixed) run.

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
from markers import circle, diamond, hexagon, marker_inner_style, marker_outer_style, pentagon, square, styled, triangle  # noqa: E402
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
OUTPUT_PATH = DATA_DIR / "POVM_random_M_plot.pdf"

STAT_SIGMA = 1000  #"inf" 

RUN_PARAMS = {"ntrain": 256, "ntest": 200, "nobs": 100}

FILENAME_RE = re.compile(
    r"MSE_vs_N_POVM_d=(?P<d>\d+)_nout=(?P<nout>\d+)_M=(?P<M>[^_]+)"
    r"_ntrain=(?P<ntrain>\d+)_ntest=(?P<ntest>\d+)_nobs=(?P<nobs>\d+)_\.txt$"
)

STYLE = {
    4:   ("#8FB032", square,   r"$n_{\mathrm{out}} = 4$"  ),  # green, diamond-look
    8:   ("#5E81B5", diamond,  r"$n_{\mathrm{out}} = 8$"  ),  # blue, square-look
    16:  ("#E19C24", triangle, r"$n_{\mathrm{out}} = 16$" ),  # orange, triangle
    32:  ("#EB6235", circle,   r"$n_{\mathrm{out}} = 32$" ),  # red, circle
    64:  ("#2FA672", pentagon, r"$n_{\mathrm{out}} = 64$" ),  # teal, pentagon
    128: ("#A64CB8", hexagon,  r"$n_{\mathrm{out}} = 128$"),  # purple, hexagon
}

MARKERSIZE = [ 11, 13, 11, 11, 11]
EDGEWIDTH = 1.3


DIM_OUT_ORDER = ( 8, 16, 32, 64, 128)

if STAT_SIGMA==1000:
    Y_LIMITS = (5e-4, 0.1)
else:
    Y_LIMITS = (1e-8, 0.1)



def load_curves():
    """Load the M=1000 curves matching `RUN_PARAMS`, plus the (nout ->
    other (ntrain, ntest, nobs) tuples seen) for files that match on
    `nout`/`M` but not on `RUN_PARAMS`, so `main` can warn about them.
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
    dim_out_order = [n for n in DIM_OUT_ORDER if n in curves]
    if not dim_out_order:
        raise FileNotFoundError(
            f"no MSE_vs_N_POVM_d=..._M={STAT_SIGMA}_..._.txt file matching "
            f"RUN_PARAMS={RUN_PARAMS} found in {DATA_DIR}"
        )
    missing = [n for n in DIM_OUT_ORDER if n not in curves]
    if missing:
        print(f"note: no RUN_PARAMS={RUN_PARAMS} data for nout={missing}, skipping")
    for n_out, param_sets in other_runs.items():
        if n_out not in dim_out_order:
            continue
        for params in param_sets:
            print(f"note: ignoring other run for nout={n_out}: {dict(params)}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []

    for n, n_out in enumerate(dim_out_order):
        stat_list, res_mse = curves[n_out]
        p10, p50, p90 = res_mse[:, 0], res_mse[:, 1], res_mse[:, 2]
        color, shape, label = STYLE[n_out]
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

        dashed_artist = ax.hlines(2/3/1000, 4*1e1, 5*1e6, color='k', linestyle='--', linewidth=2, zorder=3)
        
        if n == 0:
            dashed_handle = dashed_artist

        legend_handles.append((line, inner, outer))
        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*Y_LIMITS)

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [10, 100, 1000, 1e4, 1e5]
    x_labels = [rf"$10^{{{i}}}$" for i in range(1, 6)]
    ax.set_xlim(2, 1.4e5)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.6,0.4],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
