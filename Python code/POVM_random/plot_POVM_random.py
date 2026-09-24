import re
import sys
from pathlib import Path
import numpy as np
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
OUTPUT_PATH = DATA_DIR / "POVM_random_plot.pdf"

STAT_SIGMA = 1000

RUN_PARAMS = {"ntrain": 256, "ntest": 200, "nobs": 100}

FILENAME_RE = re.compile(
    r"MSE_vs_N_POVM_d=(?P<d>\d+)_nout=(?P<nout>\d+)_M=(?P<M>[^_]+)"
    r"_ntrain=(?P<ntrain>\d+)_ntest=(?P<ntest>\d+)_nobs=(?P<nobs>\d+)_\.txt$"
)

STYLE = {
    8:   ("#8FB032", square,  13, r"$n_{\mathrm{out}} = 8$"  ),  # green, diamond-look
    16:   ("#5E81B5", diamond, 13,r"$n_{\mathrm{out}} = 16$"  ),  # blue, square-look
    32:  ("#E19C24", triangle, 12, r"$n_{\mathrm{out}} = 32$" ),  # orange, triangle
    128:  ("#EB6235", circle,   9.5, r"$n_{\mathrm{out}} = 128$" ),  # red, circle
    168: ("#2FA672", pentagon, 11, r"$n_{\mathrm{out}} = 168$" ),  # teal, pentagon
    64: ("#A64CB8", hexagon,  11, r"$n_{\mathrm{out}} = 64$"),  # purple, hexagon
}

MARKERSIZE = [13, 9.5,11, 8.5, 11, 11]
EDGEWIDTH = 1.3


#DIM_OUT_ORDER = (4,8,16, 32,64, 128)
DIM_OUT_ORDER = (8,16,32,64)

X_LIMITS = (2, 1.3e5)
if STAT_SIGMA=="inf":
    Y_LIMITS = (1.6e-8, 3e-1)
else:
    Y_LIMITS = (5e-4, 6e-2)


def load_curves():
    """Load the M=1000 curves matching `RUN_PARAMS`, plus the (nout ->
    other (ntrain, ntest, nobs) tuples seen) for files that match on
    `nout`/`M` but not on `RUN_PARAMS`, so `main` can warn about them.
    """
    curves = {}
    other_runs = {}
    for path in DATA_DIR.glob(f"MSE_vs_N_POVM_d=*_M={STAT_SIGMA}_*_.txt"):
        print(path)
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
    missing = [n for n in DIM_OUT_ORDER if n not in curves]
    if missing:
        raise FileNotFoundError(f"no MSE_vs_N_POVM_d=..._.txt file found for nout={missing}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    legend_handles, legend_labels = [], []

    for n, n_out in enumerate(DIM_OUT_ORDER):
        stat_list, res_mse = curves[n_out]
        p10, p50, p90 = res_mse[:, 0], res_mse[:, 1], res_mse[:, 2]
        color, shape, size, label = STYLE[n_out]
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
        interval=np.linspace(2*1e2, 2*1e5, 100)
        (dashed_artist,) = ax.plot(interval,2/3/(interval*256)*(1+3*(interval/(interval+8))**2), color='k', linestyle='--', linewidth=2, zorder=10)
        
        if n == 0:
            if STAT_SIGMA=="inf":
                interval=np.linspace(2*1e0, 5*1e3, 100)
                (dashdot_line,) = ax.plot(interval, 16/3/interval**2,
                                          color='r', linestyle='-.',linewidth=2, zorder=10)
                dashed_handle, dashdot_handle = dashed_artist, dashdot_line
            
        if n == 0:
            if STAT_SIGMA==1000:
                dashdot_line = ax.hlines( 2/3/STAT_SIGMA,1e2, 1.3e5,
                                          color='k', linestyle='--',linewidth=2, zorder=10)
                dashed_handle, dashdot_handle = dashed_artist, dashdot_line

        legend_handles.append((line, inner, outer))
        legend_labels.append(label)
    
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(*Y_LIMITS)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    x_ticks = [10, 100, 1000, 1e4, 1e5]
    x_labels = [rf"$10^{{{i}}}$" for i in range(1, 6)]
    ax.set_xscale("log")
    ax.set_xlim(*X_LIMITS)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    nout_legend=ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc=[0.62,0.505],
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )
    ax.add_artist(nout_legend)
    if STAT_SIGMA=="inf":
        ax.legend(
                [dashdot_handle,dashed_handle],
                #[r"$b^2/N^2$",r"$V/(Nn_{tr})$"],
                [r"$\mathbb{E}_{\sigma,\mathcal{O}}[b^2]/N^2$",r"$\mathbb{E}_{\sigma,\mathcal{O}}[V]/(Nn_{tr})$"],
                loc=[0.57,0.7],
                frameon=True,
                framealpha=0.9,
                edgecolor="0.8",
            )
    if STAT_SIGMA==1000:
        ax.legend(
                [dashdot_handle,dashed_handle],
                [r"Var$(\boldsymbol{w}_{\mathcal{O}})/M$"],
                loc=[0.25,0.8],
                frameon=True,
                framealpha=0.9,
                edgecolor="0.8",
            )
    

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
