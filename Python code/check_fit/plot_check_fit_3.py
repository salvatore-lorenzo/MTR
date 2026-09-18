"""Plot a `check_fit_3.py` results file: mean-MSE comparison + merged box plot.

Set the parameters below to match a previous `check_fit_3.py` run --
POVM_TYPE/DIM/N_OUT/N_TRAIN/M_FINITE/GAMMA_NONZERO must be identical,
since together they determine the results filename
(`check_fit_3.results_filename`). This script locates and loads that
pickle file from ``check_fit_3.RESULTS_DIR`` and reproduces the two
figures -- `check_fit_3_mean.pdf` (mean MSE per context vs. fit) and
`check_fit_3_boxplot.pdf` (merged per-context box plot) -- without
re-running the (potentially expensive) simulation. The `contexts` list
used for plotting (gamma/gamma_test/M/color/label per context) comes
from the loaded file itself, not from a hardcoded constant here, so the
figures always match exactly what was actually simulated.

Run with ``python3 plot_check_fit_3.py`` from anywhere; it locates its
input/output directories next to itself.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # check_fit.py, check_fit_3.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import check_fit as cf  # noqa: E402  (mse_fit, load_results)
import check_fit_3 as cf3  # noqa: E402  (results_filename, RESULTS_DIR)
import noise_comparison as nc  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Parameters identifying which saved results file to load -- must match
# a previous check_fit_3.py run's parameters exactly.
# ----------------------------------------------------------------------

DIM = 4
POVM_TYPE = "random"  # "mub" (complete MUB POVM) or "random" (Haar-random rank-one POVM)
N_OUT = 512
N_TRAIN = 4092
M_FINITE = 10000
GAMMA_NONZERO = 0.05
shot_noise = "mult"  # "gauss" (Gaussian) or "mult" (multinomial) -- must match the original run

GROUP_OFFSET_LOG10 = 0.055  # log10(N) shift between adjacent contexts' boxes in the merged box plot
GROUP_BOX_WIDTH_FRACTION = 0.12  # box width as a fraction of its (offset) position
BOX_YLIM = (1e-9, 1e0)  # fixed range for the box plot (as in check_fit_single.py); without it, a rare
# near-zero outlier (e.g. context 1's M=infinite, no noise floor) auto-scales the axis over many extra
# decades, squeezing every box down to a sliver even though the underlying IQR spread hasn't changed

def results_filename(dim, povm_type, n_out, n_train, m_finite, gamma_nonzero):
    """Filename encoding exactly the parameters needed to locate/recompute
    this run's fits -- prefixed "check_fit_3_gauss_" (distinct from
    `check_fit_3.results_filename`'s "check_fit_3_") so the multinomial
    and Gaussian results for the same parameters never collide. Not
    encoded: N_REPEATS/SEED (simulation settings that don't enter the
    fit) -- re-running with those changed but the same identifying
    parameters overwrites the previous file.
    """
    if shot_noise=="gauss":
        return (
            f"check_fit_3_gauss_povm={povm_type}_d={dim}_nout={n_out}_ntrain={n_train}"
            f"_M={m_finite}_gamma={gamma_nonzero}.pkl"
        )
    else:
        return (
            f"check_fit_3_povm={povm_type}_d={dim}_nout={n_out}_ntrain={n_train}"
            f"_M={m_finite}_gamma={gamma_nonzero}.pkl"
        ) 

def savefig(fig, stem):
    """Save a figure as both PDF and PNG next to this script. `bbox_inches
    ="tight"` so a legend/title extending past the axes never gets clipped.
    """
    if shot_noise=="gauss":
        fig.savefig(OUTPUT_DIR / f"{stem}_gauss.pdf", bbox_inches="tight")
        #fig.savefig(OUTPUT_DIR / f"{stem}_gauss.png", dpi=150, bbox_inches="tight")
    else:
        fig.savefig(OUTPUT_DIR / f"{stem}_mult.pdf", bbox_inches="tight")
        #fig.savefig(OUTPUT_DIR / f"{stem}_mult.png", dpi=150, bbox_inches="tight")
    #fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    nc.plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


def plot_mean_comparison(N_values, contexts, mean_results, dim, n_tr):
    """All three contexts' mean empirical MSE(N) overlaid against their
    fits (`mean_results`: (mse_empirical, mse_fit_values) per context),
    each context plotted in its own color: empirical as a solid marked
    line, fit as a dash-dot line.
    """
    fig, ax = nc.plt.subplots(figsize=(8.3, 6.0))

    context_handles = []
    for ctx, (mse_empirical, mse_fit_values) in zip(contexts, mean_results):
        color = ctx["color"]
        (emp_line,) = ax.plot(N_values, mse_empirical, color=color, marker="o", markersize=5, linewidth=1.6, zorder=3)
        ax.plot(N_values, mse_fit_values, color=color, linestyle="-.", linewidth=2.0, zorder=3)
        context_handles.append(emp_line)

    x_ticks = tuple(10**k for k in range(1, int(np.ceil(np.log10(N_values[-1]))) + 1))
    nc._style_axes(ax, r"$N$", "MSE (mean)", log_y=True, x_ticks=x_ticks)
    ax.set_title(rf"Same experiment, three noise contexts ($d={dim}$, $n_{{\rm tr}}={n_tr}$)")
    ax.set_xlim(N_values[0] / 3, N_values[-1] * 3)

    context_legend = ax.legend(
        context_handles, [ctx["label"] for ctx in contexts],
        title="context", loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.add_artist(context_legend)  # kept when the legend below replaces the "current" legend
    empirical_proxy = Line2D([], [], color="black", marker="o", markersize=5, linewidth=1.6)
    fit_proxy = Line2D([], [], color="black", linestyle="-.", linewidth=2.0)
    ax.legend(
        [empirical_proxy, fit_proxy], ["empirical MSE (mean)", r"${\rm MSE}_{\rm fit}(N)$"],
        loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8",
    )

    fig.tight_layout()
    savefig(fig, "check_fit_3_mean")


def plot_boxplot_grouped(N_values, contexts, boxplot_results, dim, n_tr, n_repeats):
    """All three contexts' box plots merged into a single panel: at each
    N, one box per context, offset slightly in log10(N) (`GROUP_OFFSET_
    LOG10`) so they sit side by side instead of overlapping, colored by
    context, each with its own fit line drawn through its own (offset)
    box cluster. `boxplot_results` is a list of (sq_errors, mse_fit_
    values) pairs, one per entry of `contexts`.
    """
    fig, ax = nc.plt.subplots(figsize=(9.6, 6.6))

    n_ctx = len(contexts)
    N_arr = np.asarray(N_values, dtype=float)
    offsets = (np.arange(n_ctx) - (n_ctx - 1) / 2) * GROUP_OFFSET_LOG10

    context_handles = []
    for ctx, offset, (sq_errors, mse_fit_values) in zip(contexts, offsets, boxplot_results):
        color = ctx["color"]
        positions = N_arr * 10**offset
        widths = positions * GROUP_BOX_WIDTH_FRACTION
        data = [sq_errors[:, j] for j in range(len(N_values))]

        ax.boxplot(
            data, positions=positions, widths=widths,
            manage_ticks=False, patch_artist=True, showfliers=True, zorder=3,
            boxprops=dict(facecolor=color, alpha=0.5, edgecolor=color, linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.3),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=color, markeredgecolor="none", alpha=0.5),
        )
        ax.plot(positions, mse_fit_values, color=color, linestyle="-.", linewidth=1.6, zorder=4)
        context_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor=color, linewidth=1.0))

    x_ticks = tuple(10**k for k in range(1, int(np.ceil(np.log10(N_values[-1]))) + 1))
    nc._style_axes(ax, r"$N$", "squared error", log_y=True, x_ticks=x_ticks)
    ax.set_xlim(N_values[0] / 3, N_values[-1] * 3)
    ax.set_ylim(*BOX_YLIM)
    ax.set_title(
        rf"($d={dim}$, $n_{{\rm tr}}={n_tr}$, $n_{{\rm out}}={N_OUT}$, $N_{{\rm repeats}}={n_repeats}$ shot-noise draws)"
    )

    context_legend = ax.legend(
        context_handles, [ctx["label"] for ctx in contexts],
        #title="context", 
        loc=[0.18,0.8], frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.add_artist(context_legend)  # kept when the legend below replaces the "current" legend

    # Generic anatomy legend, neutral gray -- box color/position already
    # identifies the context (see legend above), so this only explains
    # what the (otherwise colorless) median/whiskers/outliers/fit mean.
    median_line = Line2D([], [], color="black", linewidth=1.3)
    whisker_line = Line2D([], [], color="0.5", linewidth=1.0)
    outlier_marker = Line2D(
        [], [], marker="o", linestyle="None", markersize=3, markerfacecolor="0.5", markeredgecolor="none", alpha=0.5
    )
    fit_proxy = Line2D([], [], color="0.3", linestyle="-.", linewidth=1.6)
    ax.legend(
        [median_line, whisker_line, outlier_marker, fit_proxy],
        ["median", r"whiskers ($1.5\times$IQR)", "outliers", r"${\rm MSE}_{\rm fit}(N)$"],
        loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=11,
    )

    fig.tight_layout()
    savefig(fig, "check_fit_3_boxplot")


def main():
    filename = results_filename(DIM, POVM_TYPE, N_OUT, N_TRAIN, M_FINITE, GAMMA_NONZERO)
    path = cf3.RESULTS_DIR / filename
    print(f"Loading {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run check_fit_3.py with these same parameters "
            f"(povm={POVM_TYPE}, d={DIM}, n_out={N_OUT}, n_train={N_TRAIN}, "
            f"M={M_FINITE}, gamma={GAMMA_NONZERO}) first."
        )
    results = cf.load_results(path)

    N_values = results["N_values"]
    contexts = results["contexts"]
    sq_errors_list = results["sq_errors_list"]
    dim, n_tr, n_out = results["dim"], results["n_train"], results["n_out"]
    n_repeats = results["n_repeats"]
    print(
        f"loaded {len(contexts)} contexts, {len(N_values)} N-values "
        f"(n_repeats={n_repeats}, seed={results['seed']})"
    )

    mean_results, boxplot_results = [], []
    for ctx, sq_errors in zip(contexts, sq_errors_list):
        mse_fit_values = cf.mse_fit(N_values, dim, n_tr, n_out, ctx["gamma"], ctx["gamma_test"], ctx["M"])
        mean_results.append((sq_errors.mean(axis=0), mse_fit_values))
        boxplot_results.append((sq_errors, mse_fit_values))

    print("\n=== plot: mean MSE vs. fit, per context ===")
    plot_mean_comparison(N_values, contexts, mean_results, dim, n_tr)

    print("\n=== plot: squared-error box plots (merged), per context, vs. fit ===")
    plot_boxplot_grouped(N_values, contexts, boxplot_results, dim, n_tr, n_repeats)

    print("\nDone. Figures written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
