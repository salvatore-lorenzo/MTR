"""Plot several `check_fit.py` results files on the same axes.

Same as `plot_check_fit.py`, but the curves relative to different values
of GAMMA, GAMMA_TEST and M are drawn together. Set the parameters below to match previous
`check_fit.py` runs -- DIM/N_OUT/N_TRAIN must be identical, and for
each (GAMMA, GAMMA_TEST) in GAMMA_PAIRS and each M in MS there must be a
run with those values, since together they determine the results filename. For each
curve the figure shows the empirical MSE(N) of the fixed instance (mean
over the realizations, with its standard error, and the 10th-90th
percentile band of the squared errors) and the fit f(N).

Run with ``python3 plot_check_fit_2.py`` from anywhere; it locates its
input/output directories next to itself.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent))  # check_fit.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import check_fit as cf  # noqa: E402
import noise_comparison as nc  # noqa: E402
from markers import circle, diamond, hexagon, marker_inner_style, marker_outer_style, pentagon, square, styled, triangle  # noqa: E402

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

OUTPUT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Parameters identifying which saved results files to load -- must match
# previous check_fit.py runs exactly (one run per (GAMMA, GAMMA_TEST) in
# GAMMA_PAIRS and M in MS).
# ----------------------------------------------------------------------

DIM = 2
N_OUT = 16
N_TRAIN = 256
MS = [10000,None]  # None = exact test statistics
GAMMA_PAIRS = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)]  # (gamma, gamma_test)

COLORS = ["#5E81B5", "#E19C24", "#8FB032", "#EB6235", "#A64CB8", "#3AA39B"]  # cycled over the curves
SHAPES = [circle, square, triangle, diamond, pentagon, hexagon]  # cycled over the curves (`square` looks like a diamond, see markers.py)
MARKERSIZE = [9.5, 11, 13, 11, 11, 11]  # cycled over the curves
EDGEWIDTH = 1.3


def plot_one(ax, results, color, shape, size):
    """Draw the empirical MSE(N) and the fit f(N) of one results file; return
    the positive values used to set the y-limits."""
    N = results["N_values"].astype(float)
    sq = results["sq_errors"]
    coeffs = results["coeffs"]
    mse = sq.mean(axis=0)
    sem = sq.std(axis=0, ddof=1) / np.sqrt(sq.shape[0])
    p10, p90 = np.percentile(sq, [10, 90], axis=0)
    N_dense = np.logspace(np.log10(N[0]), np.log10(N[-1]), 400)

    ax.plot(N_dense, cf.mse_fit(N_dense, coeffs), color="k", linestyle="--", linewidth=2.0, zorder=4)

    ax.fill_between(N, p10, p90, color=color, alpha=0.2, linewidth=0, zorder=1)

    m = styled(shape)
    ax.errorbar(N, mse, yerr=sem, color=color, linestyle="None", elinewidth=1.2, capsize=2, zorder=3)
    ax.plot(N, mse, color=color, linewidth=1.4, zorder=3)
    ax.plot(N, mse, marker=m, linestyle="None", zorder=5, **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH))
    ax.plot(N, mse, marker=m, linestyle="None", zorder=6, **marker_outer_style(color, size=size * 15 / 35))

    return N, np.concatenate([mse, cf.mse_fit(N, coeffs), p10, p90])


LEGEND_COLUMN_WIDTHS = ("2.4em", "2.8em", "3.6em")  # gamma_tr, gamma_test, M


def legend_row(*cells):
    """One row of the table-style legend: each cell centred in a fixed-width box."""
    return "".join(rf"\makebox[{w}]{{{c}}}" for w, c in zip(LEGEND_COLUMN_WIDTHS, cells))


def plot_fit_check(all_results):
    fig, ax = nc.plt.subplots(figsize=(8.0, 5.8))

    # The legend is a table: a header row with no marker, then one row of values per curve.
    handles = [Line2D([], [], linestyle="None")]
    labels = [legend_row(r"$\gamma_{\rm tr}$", r"$\gamma_{\rm test}$", r"$M$")]
    positive, N_all = [], []
    for i, results in enumerate(all_results.values()):
        color = COLORS[i % len(COLORS)]
        shape = SHAPES[i % len(SHAPES)]
        size = MARKERSIZE[i % len(MARKERSIZE)]
        N, values = plot_one(ax, results, color, shape, size)
        N_all.append(N)
        positive.append(values)
        m = styled(shape)
        handles.append((
            Line2D([], [], color=color, linewidth=1.4, marker=m, **marker_inner_style(color, size=size, edgewidth=EDGEWIDTH)),
            Line2D([], [], linestyle="None", marker=m, **marker_outer_style(color, size=size * 15 / 35)),
        ))
        M_str = r"\infty" if results["M"] is None else r"10^4"#str(results["M"])
        labels.append(legend_row(f"${results['gamma']}$", f"${results['gamma_test']}$", f"${M_str}$"))

    N_all = np.concatenate(N_all)
    positive = np.concatenate(positive)
    positive = positive[positive > 0]
    x_ticks = tuple(10**k for k in range(1, int(np.ceil(np.log10(N_all.max()))) + 1))
    nc._style_axes(ax, r"$N$", "MSE", log_y=True, x_ticks=x_ticks)
    #ax.set_xlim(N_all.min() / 2, N_all.max() * 2)
    ax.set_xlim(2,1.2e6)
    ax.set_ylim(1e-8, 0.3*1e0)

    ax.legend(handles, 
              labels, 
              #handler_map={tuple: HandlerTuple(ndivide=1)}, 
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True, framealpha=0.9, edgecolor="0.8")

    fig.savefig(OUTPUT_DIR / f"{output_stem()}.pdf", bbox_inches="tight")
    nc.plt.close(fig)
    print(f"  wrote {output_stem()}.pdf")


def output_stem():
    #M_str = "inf" if M is None else str(M)
    #gammas_str = ",".join(str(g) for g in GAMMAS)
    return f"check_fit_2_d={DIM}_nout={N_OUT}_ntrain={N_TRAIN}"#_M={M_str}_gammas={gammas_str}"


def main():
    all_results = {}
    for gamma, gamma_test in GAMMA_PAIRS:
        for M in MS:
            path = cf.RESULTS_DIR / cf.results_filename(DIM, N_OUT, N_TRAIN, M, gamma, gamma_test)
            print(f"Loading {path}")
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- run check_fit.py with these same parameters first.")
            results = cf.load_results(path)
            print(f"  gamma={gamma} gamma_test={gamma_test} M={M} fit coefficients: " + "  ".join(f"{k}={v:.4e}" for k, v in results["coeffs"].items()))
            all_results[gamma, gamma_test, M] = results
    plot_fit_check(all_results)


if __name__ == "__main__":
    main()
