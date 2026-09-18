"""Plot a `check_fit.py` results file as a box plot per N, against the fit.

Same loading convention as `plot_check_fit.py`: set the parameters below
to match a previous `check_fit.py` run -- POVM_TYPE/DIM/N_OUT/N_TRAIN/
GAMMA/GAMMA_TEST/M_TEST must be identical, since together they determine
the results filename (`results_filename` in `check_fit.py`). Instead of
one thin line per experiment, this version draws one box plot per N
value, summarizing the spread of `mse_empirical[:, j]` (the
`N_EXPERIMENTS` independent MSE(N) values at that N) against
`check_fit.mse_fit(N, ...)` -- without re-running the (potentially
expensive) simulation.

Run with ``python3 plot_check_fit_2.py`` from anywhere; it locates its
input/output directories next to itself.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # check_fit.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import check_fit as cf  # noqa: E402
import noise_comparison as nc  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Parameters identifying which saved results file to load -- must match
# a previous check_fit.py run's fit parameters exactly.
# ----------------------------------------------------------------------

DIM = 8
POVM_TYPE = "random"  # "mub" (complete MUB POVM) or "random" (Haar-random rank-one POVM)
N_OUT = 640
N_TRAIN = 6400
GAMMA = 0.05
GAMMA_TEST = 0.05
M_TEST = 10000

BOX_COLOR = nc.MODEL_STYLE["mult"][0]
BOX_WIDTH_FRACTION = 0.25  # box width as a fraction of N (so boxes look even in log-x)


def savefig(fig, stem):
    """Save a figure as both PDF and PNG next to this script."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150)
    nc.plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


def plot_mse_fit_boxplot(N_values, mse_empirical, mse_fit_values, dim, n_tr, gamma, gamma_test, M, povm_type, n_out):
    """`mse_empirical` has shape (n_experiments, len(N_values)): for each N
    (column), a box plot summarizes the `n_experiments` independent MSE
    values at that N, against the single `mse_fit_values` curve.
    """
    fig, ax = nc.plt.subplots(figsize=(2 * 6.4, 2 * 4.6))

    n_experiments = mse_empirical.shape[0]
    data = [mse_empirical[:, j] for j in range(len(N_values))]
    widths = np.asarray(N_values, dtype=float) * BOX_WIDTH_FRACTION

    ax.boxplot(
        data, positions=N_values, widths=widths,
        manage_ticks=False, patch_artist=True, showfliers=True, 
         zorder=3,
        boxprops=dict(facecolor=BOX_COLOR, alpha=0.5, edgecolor=BOX_COLOR, linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color=BOX_COLOR, linewidth=1.2),
        capprops=dict(color=BOX_COLOR, linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=BOX_COLOR, markeredgecolor="none", alpha=0.6),
    )
    (fit_line,) = ax.plot(N_values, mse_fit_values, color="k", linestyle="-.", linewidth=2.2, zorder=4)

    x_ticks = tuple(10**k for k in range(1, int(np.ceil(np.log10(N_values[-1]))) + 1))
    nc._style_axes(ax, r"$N$", "MSE", log_y=True, x_ticks=x_ticks)
    M_label = r"\infty" if M is None else str(M)
    povm_label = "MUB" if povm_type == "mub" else rf"random, $n_{{\rm out}}={n_out}$"
    ax.set_title(
        rf"MSE fit check ({povm_label}, $d={dim}$, $n_{{\rm tr}}={n_tr}$, $\gamma={gamma}$, $M={M_label}$)"
    )

    box_patch = Patch(facecolor=BOX_COLOR, alpha=0.5, edgecolor=BOX_COLOR, linewidth=1.2)
    median_line = Line2D([], [], color="black", linewidth=1.6)
    whisker_line = Line2D([], [], color=BOX_COLOR, linewidth=1.2)
    outlier_marker = Line2D(
        [], [], marker="o", linestyle="None", markersize=4, markerfacecolor=BOX_COLOR, markeredgecolor="none", alpha=0.6
    )
    ax.legend(
        [box_patch, median_line, whisker_line, outlier_marker, fit_line],
        [
            "IQR (Q1" + "–" + f"Q3), {n_experiments} experiments",
            "median",
            r"whiskers ($1.5\times$IQR)",
            "outliers",
            r"${\rm MSE}_{\rm fit}(N)$",
        ],
        loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.set_xlim(N_values[0] / 3, N_values[-1] * 3)
    ax.set_ylim(1e-5, 1e-1)
    fig.tight_layout()
    savefig(fig, "MSE_fit_check_boxplot")


def main():
    filename = cf.results_filename(DIM, POVM_TYPE, N_OUT, N_TRAIN, GAMMA, GAMMA_TEST, M_TEST)
    path = cf.RESULTS_DIR / filename
    print(f"Loading {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run check_fit.py with these same parameters "
            f"(povm={POVM_TYPE}, d={DIM}, n_out={N_OUT}, n_train={N_TRAIN}, "
            f"gamma={GAMMA}, gamma_test={GAMMA_TEST}, M={M_TEST}) first."
        )
    results = cf.load_results(path)

    N_values = results["N_values"]
    mse_empirical = results["mse_empirical"]
    print(
        f"loaded {mse_empirical.shape[0]} experiments, {len(N_values)} N-values "
        f"(n_real={results['n_real']}, seed={results['seed']})"
    )

    mse_fit_values = cf.mse_fit(N_values, DIM, N_TRAIN, N_OUT, GAMMA, GAMMA_TEST, M_TEST)

    print("\n=== plot: empirical MSE box plot (per N) vs. closed-form fit ===")
    plot_mse_fit_boxplot(
        N_values, mse_empirical, mse_fit_values, DIM, N_TRAIN, GAMMA, GAMMA_TEST, M_TEST, POVM_TYPE, N_OUT
    )

    print("\nDone. Figure written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
