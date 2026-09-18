"""Same plot as the upper panel of ``1_MSE_vs_N.pdf`` (see
``noise_comparison.py``) -- multinomial vs. Gaussian-approximation MSE(N),
plus the theoretical asymptote -- repeated as a small-multiples grid over
several values of N_TRAIN, to show that the Gaussian approximation tracks
the true multinomial noise well across training-set sizes, not just at the
single n_tr=500 used in the main script.

For each n_tr in `N_TRAIN_VALUES`, `N_REPS` independent regression-scenario
instances are drawn (fresh Haar states/POVM/observables every time, reusing
`noise_comparison.repeated_regression_sweep`) and the median MSE(N) curve
for both noise models, with its [p10, p90] instance band, is plotted in its
own panel together with that n_tr's theoretical asymptote
A_d/N^2 + B_d/(N n_tr) (`noise_comparison.theoretical_bias2_variance`).
Color/shape/legend per model reuse `noise_comparison.MODEL_STYLE` (blue
diamond = multinomial, orange triangle = Gaussian approx.), so each panel
looks exactly like the upper panel of ``1_MSE_vs_N.pdf``.

Each n_tr panel uses its own seeded RNG stream (reset to `SEED` before its
sweep), so panels are reproducible independently of what other values
appear in `N_TRAIN_VALUES` or in what order.

Run with ``python3 noise_comparison_2.py`` from anywhere; it locates its
output directory next to itself.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_comparison as nc  # noqa: E402

N_TRAIN_VALUES = [1, 10, 100, 1000]
NCOLS = 2


def plot_mse_vs_N_grid(N_values, results_by_ntrain, dim, n_reps, n_train_values):
    n = len(n_train_values)
    ncols = min(NCOLS, n)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = nc.plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    b2, variance = nc.theoretical_bias2_variance(dim)
    N_theory = np.logspace(np.log10(3), np.log10(3000), 200)

    handles_by_model, theory_handle = None, None
    for ax, n_train in zip(axes, n_train_values):
        results = results_by_ntrain[n_train]
        local_handles = {}
        for model in ("mult", "gauss"):
            color, shape, _ = nc.MODEL_STYLE[model]
            local_handles[model] = nc._styled_band_and_line(ax, N_values, results[model]["mse"], color, shape)

        mse_theory = b2 / N_theory**2 + variance / (N_theory * n_train)
        (theory_line,) = ax.plot(N_theory, mse_theory, color="k", linestyle="-.", linewidth=1.4, zorder=1)

        nc._style_axes(ax, r"$N$", "MSE", log_y=True)
        ax.set_title(rf"$n_{{\rm tr}} = {n_train}$")

        if handles_by_model is None:
            handles_by_model, theory_handle = local_handles, theory_line

    for ax in axes[n:]:
        ax.set_visible(False)

    nc._model_legend(
        axes[0], handles_by_model, loc="upper right",
        extra=[(theory_handle, r"$A_d/N^2 + B_d/(N n_{\rm tr})$")],
    )

    fig.suptitle(rf"MSE vs $N$: multinomial vs.\ Gaussian approx.\ ($d={dim}$, {n_reps} instances)")
    fig.tight_layout()
    nc.savefig(fig, "1_MSE_vs_N_ntrain_sweep")


def main():
    print("=" * 72)
    print("MSE vs N: multinomial vs. Gaussian approx., swept over N_TRAIN")
    print("=" * 72)

    results_by_ntrain = {}
    for n_train in N_TRAIN_VALUES:
        nc.set_seed(nc.SEED)
        print(
            f"\n=== n_train={n_train}: {nc.N_REPS} instances x "
            f"{nc.N_NOISE_REALIZATIONS} inner noise realizations ==="
        )
        results_by_ntrain[n_train] = nc.repeated_regression_sweep(
            nc.DIM, nc.N_VALUES, n_train, nc.N_TEST_STATES, nc.N_OBSERVABLES,
            nc.N_NOISE_REALIZATIONS, nc.N_REPS, M_test=nc.M_TEST,
        )

    print("\n=== plot: one panel per n_train, median + [p10,p90] instance band ===")
    plot_mse_vs_N_grid(nc.N_VALUES, results_by_ntrain, nc.DIM, nc.N_REPS, N_TRAIN_VALUES)

    print("\nDone. Figure written to:", nc.OUTPUT_DIR)


if __name__ == "__main__":
    main()
