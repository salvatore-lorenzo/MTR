"""Plot a `check_fit.py` results file against the closed-form MSE fit.

Set the parameters below to match a previous `check_fit.py` run --
POVM_TYPE/DIM/N_OUT/N_TRAIN/GAMMA/GAMMA_TEST/M_TEST must be identical,
since together they determine the results filename (`results_filename`
in `check_fit.py`). This script locates and loads that pickle file from
``results/`` and plots the saved empirical MSE(N) curves (one per
experiment, colored on a purple gradient) against
`check_fit.mse_fit(N, ...)` -- without re-running the (potentially
expensive) simulation.

Run with ``python3 plot_check_fit.py`` from anywhere; it locates its
input/output directories next to itself.
"""

import sys
from pathlib import Path

import numpy as np

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

EXPERIMENT_CMAP = "Purples"
EXPERIMENT_CMAP_RANGE = (0.35, 0.95)  # avoid the near-white/near-black ends of the colormap


def savefig(fig, stem):
    """Save a figure as both PDF and PNG next to this script."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150)
    nc.plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


def plot_mse_fit_check(N_values, mse_empirical, mse_fit_values, dim, n_tr, gamma, gamma_test, M, povm_type, n_out):
    """`mse_empirical` has shape (n_experiments, len(N_values)): each row is
    one independent experiment's MSE(N) curve, plotted as its own thin line
    (no averaging/quantile band) against the single `mse_fit_values` curve.
    """
    fig, ax = nc.plt.subplots(figsize=(2 * 6.4, 2 * 4.6))

    n_experiments = mse_empirical.shape[0]
    cmap = nc.plt.get_cmap(EXPERIMENT_CMAP)
    lo, hi = EXPERIMENT_CMAP_RANGE
    colors = [cmap(t) for t in np.linspace(lo, hi, n_experiments)]
    for i, row in enumerate(mse_empirical):
        ax.plot(N_values, row, color=colors[i], marker='o', linestyle='', linewidth=1.5, alpha=0.9, zorder=3)
    (fit_line,) = ax.plot(N_values, mse_fit_values, color="k", linestyle="-.", linewidth=2.2, zorder=4)

    x_ticks = tuple(10**k for k in range(1, int(np.ceil(np.log10(N_values[-1]))) + 1))
    nc._style_axes(ax, r"$N$", "MSE", log_y=True, x_ticks=x_ticks)
    M_label = r"\infty" if M is None else str(M)
    povm_label = "MUB" if povm_type == "mub" else rf"random, $n_{{\rm out}}={n_out}$"
    ax.set_title(
        rf"MSE fit check ({povm_label}, $d={dim}$, $n_{{\rm tr}}={n_tr}$, $\gamma={gamma}$, $M={M_label}$)"
    )

    ax.legend(
        [fit_line], [r"${\rm MSE}_{\rm fit}(N)$"],
        loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.set_xlim(1, 1000000)
    ax.set_ylim(1e-5, 1e-1)
    fig.tight_layout()
    savefig(fig, "MSE_fit_check")


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

    print("\n=== plot: empirical MSE vs. closed-form fit ===")
    plot_mse_fit_check(
        N_values, mse_empirical, mse_fit_values, DIM, N_TRAIN, GAMMA, GAMMA_TEST, M_TEST, POVM_TYPE, N_OUT
    )

    print("\nDone. Figure written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
