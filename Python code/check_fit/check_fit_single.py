"""Box-plot the shot-noise spread of a SINGLE fixed POVM experiment.

Unlike `check_fit.py` (which draws `N_EXPERIMENTS` independent scenarios
-- a fresh POVM/states/observable/label-noise each time -- and
box-plots the spread *across experiments*, see `plot_check_fit_2.py`),
this script fixes ONE scenario: one draw of the POVM, training states,
test state, observable, training-target label noise (`gamma`),
test-target label noise (`gamma_test`), and test-side M-shot noise --
exactly `check_fit.run_single_experiment`'s setup.

The only thing repeated is the shot-noise sampling itself: for each N,
`N_REPEATS` independent multinomial(N, P_train) realizations are drawn,
and every individual squared error is kept (not averaged into a single
MSE number as `check_fit.py` does via `bias_variance_mse`). Box-plotting
that per-N distribution shows how much a single experiment's squared
error fluctuates from one shot-noise draw to the next, which is exactly
what `MSE_fit(N)` predicts the mean of.

Run with ``python3 check_fit_single.py`` from anywhere; it locates its
output directory next to itself.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # check_fit.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import check_fit as cf  # noqa: E402  (build_povm, build_regression_scenario, mse_fit)
import noise_comparison as nc  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Parameters (all easy to change).
# ----------------------------------------------------------------------

SEED = None  # fixes the ONE scenario (POVM/states/observable/label noise/test-shot noise); None = fresh each run

DIM = 2
POVM_TYPE = "random"  # "mub" (complete MUB POVM) or "random" (Haar-random rank-one POVM)
N_OUT_RANDOM = 512  # number of POVM outcomes when POVM_TYPE == "random" (ignored for "mub")
N_OUT = DIM * (DIM + 1) if POVM_TYPE == "mub" else N_OUT_RANDOM
N_TRAIN = 1024  # must be > N_OUT + 1 for the fit's cross term to be defined
GAMMA = 0.01  # additive Gaussian label noise std on the training targets
GAMMA_TEST = 0.01  # additive Gaussian label noise std on the test target (0 = exact test target)
M_TEST = 5000  # test-side shot budget (None = exact test statistics)

N_VALUES = sorted(set(np.round(np.logspace(np.log10(5), np.log10(1000000), 16)).astype(int).tolist()))
N_REPEATS = 20  # shot-noise realizations of the SAME fixed experiment, per N

BOX_COLOR = nc.MODEL_STYLE["mult"][0]
BOX_WIDTH_FRACTION = 0.25  # box width as a fraction of N (so boxes look even in log-x)

# term name -> (color, linestyle, legend label). Colors reuse the same
# palette `POVM_random/plot_POVM_random.py` uses for sweeping n_out
# (green/blue/orange/red/purple), just repurposed here for the fit's
# additive terms instead of n_out values.
TERM_STYLE = {
    "bias": ("#8FB032", "--", r"bias$^2$"),
    "train_shot_variance": ("#5E81B5", "--", "train shot variance"),
    "train_label_noise": ("#E19C24", ":", "train label noise"),
    "test_noise": ("#EB6235", ":", "test noise"),  # test label noise + test shot noise
    "cross": ("#A64CB8", "--", "cross term"),
}


def savefig(fig, stem):
    """Save a figure as both PDF and PNG next to this script."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150)
    nc.plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


def run_repeated_shot_noise(seed):
    """One fixed regression instance -- POVM, training states, test state,
    observable, one draw of the training-target label noise (`gamma`),
    one draw of the test-target label noise (`gamma_test`), and one draw
    of the test-side M-shot noise -- all fixed once here, up front. For
    each N, `N_REPEATS` independent multinomial(N, P_train) shot-noise
    realizations are drawn and every individual squared error is kept.

    Returns an (N_REPEATS, len(N_VALUES)) array.
    """
    nc.set_seed(seed)
    M_mu = cf.build_povm(DIM, POVM_TYPE, N_OUT)
    scenario = cf.build_regression_scenario(M_mu, DIM, N_TRAIN, 1, 1)
    P_train, P_test = scenario["P_train"], scenario["P_test"]
    y_train = scenario["y_train"] + GAMMA * np.random.randn(*scenario["y_train"].shape)
    y_test = scenario["y_test"] + GAMMA_TEST * np.random.randn(*scenario["y_test"].shape)

    # Fixed once, not redrawn as N sweeps: the single M-shot noisy read of
    # the (single) test state's probabilities.
    P_test_used = P_test if M_TEST is None else nc.noisy_test_probs(P_test, M_TEST, "mult", tol=nc.EIG_TOL)

    sq_errors = np.empty((N_REPEATS, len(N_VALUES)))
    for j, N in enumerate(N_VALUES):
        mult_stack = nc.multinomial_phat_batch(P_train, N, N_REPEATS)  # (N_REPEATS, n_out, n_train)
        o_hat_stack = nc.predict_batch(y_train, mult_stack, P_test_used, rcond=nc.PINV_RCOND)  # (N_REPEATS, 1, 1)
        sq_errors[:, j] = (o_hat_stack[:, 0, 0] - y_test[0, 0]) ** 2
        print(f"  N={N:>6d}  median={np.median(sq_errors[:, j]):.4e}  mean={np.mean(sq_errors[:, j]):.4e}")

    return sq_errors


def plot_boxplot(
    N_values, sq_errors, mse_fit_values, mse_fit_term_values, dim, n_tr, gamma, gamma_test, M, povm_type, n_out,
    n_repeats,
):
    """`sq_errors` has shape (n_repeats, len(N_values)): for each N
    (column), a box plot summarizes the `n_repeats` shot-noise
    realizations' squared errors, against the single `mse_fit_values`
    curve (which predicts their mean) and its individual additive terms
    `mse_fit_term_values` (from `check_fit.mse_fit_terms`).
    """
    fig, ax = nc.plt.subplots(figsize=(2 * 6.4, 2 * 4.6))

    data = [sq_errors[:, j] for j in range(len(N_values))]
    widths = np.asarray(N_values, dtype=float) * BOX_WIDTH_FRACTION

    ax.boxplot(
        data, positions=N_values, widths=widths,
        manage_ticks=False, patch_artist=True, showfliers=True, zorder=3,
        boxprops=dict(facecolor=BOX_COLOR, alpha=0.5, edgecolor=BOX_COLOR, linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color=BOX_COLOR, linewidth=1.2),
        capprops=dict(color=BOX_COLOR, linewidth=1.2),
        flierprops = dict(marker="o", markersize=4, markerfacecolor=BOX_COLOR, markeredgecolor="none", alpha=0.6),
    )
    (fit_line,) = ax.plot(N_values, mse_fit_values, color="k", linestyle="-.", linewidth=2.2, zorder=4)

    term_lines, term_labels = [], []
    for name, (color, linestyle, label) in TERM_STYLE.items():
        values = mse_fit_term_values[name]
        if np.allclose(values, 0):
            continue  # e.g. test_shot_noise/cross when M is None
        (term_line,) = ax.plot(
            N_values, np.broadcast_to(values, np.shape(N_values)), color=color, linestyle=linestyle,
            linewidth=1.4, alpha=0.85, zorder=2,
        )
        term_lines.append(term_line)
        term_labels.append(label)

    x_ticks = tuple(10**k for k in range(1, int(np.ceil(np.log10(N_values[-1]))) + 1))
    nc._style_axes(ax, r"$N$", "squared error", log_y=True, x_ticks=x_ticks)
    M_label = r"\infty" if M is None else str(M)
    povm_label = "MUB" if povm_type == "mub" else rf"random, $n_{{\rm out}}={n_out}$"
    ax.set_title(
        rf"Single-experiment shot-noise spread ({povm_label}, $d={dim}$, $n_{{\rm tr}}={n_tr}$, "
        rf"$\gamma={gamma}$, $M={M_label}$)"
    )

    box_patch = Patch(facecolor=BOX_COLOR, alpha=0.5, edgecolor=BOX_COLOR, linewidth=1.2)
    median_line = Line2D([], [], color="black", linewidth=1.6)
    whisker_line = Line2D([], [], color=BOX_COLOR, linewidth=1.2)
    outlier_marker = Line2D(
        [], [], marker="o", linestyle="None", markersize=4, markerfacecolor=BOX_COLOR, markeredgecolor="none", alpha=0.6
    )
    data_legend = ax.legend(
        [box_patch, median_line, whisker_line, outlier_marker, fit_line],
        [
            f"IQR (Q1–Q3), {n_repeats} shot-noise draws",
            "median",
            r"whiskers ($1.5\times$IQR)",
            "outliers",
            r"${\rm MSE}_{\rm fit}(N)$",
        ],
        loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.add_artist(data_legend)  # kept when the second ax.legend() below replaces the "current" legend

    ax.legend(
        term_lines, term_labels,
        title=r"${\rm MSE}_{\rm fit}$ terms",
        loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=12,
    )
    ax.set_xlim(N_values[0] / 3, N_values[-1] * 3)
    ax.set_ylim(1e-6, 1e-0)
    fig.tight_layout()
    savefig(fig, "check_fit_single_boxplot")


def main():
    print("=" * 72)
    print(f"Single-experiment shot-noise check -- {N_REPEATS} shot-noise draws per N (n_test=1, n_obs=1)")
    print("=" * 72)
    print(
        f"povm={POVM_TYPE}  d={DIM}  n_out={N_OUT}  n_train={N_TRAIN}  "
        f"gamma={GAMMA}  gamma_test={GAMMA_TEST}  M={M_TEST}  seed={SEED}"
    )

    sq_errors = run_repeated_shot_noise(SEED)
    mse_fit_term_values = cf.mse_fit_terms(N_VALUES, DIM, N_TRAIN, N_OUT, GAMMA, GAMMA_TEST, M_TEST)
    mse_fit_values = sum(mse_fit_term_values.values())

    print("\n=== plot: squared-error box plot (per N) vs. closed-form fit ===")
    plot_boxplot(
        N_VALUES, sq_errors, mse_fit_values, mse_fit_term_values, DIM, N_TRAIN, GAMMA, GAMMA_TEST, M_TEST,
        POVM_TYPE, N_OUT, N_REPEATS,
    )

    print("\nDone. Figure written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
