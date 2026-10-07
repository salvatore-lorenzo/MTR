"""Multinomial shot noise vs. its Gaussian approximation, for POVM-based
quantum linear regression / QELM -- run and save.

The central scientific question:

    Does replacing the true multinomial shot noise

        p_hat = Multinomial(N, p) / N

    with the singular Gaussian approximation

        p_hat = p + xi / sqrt(N),      xi ~ N(0, Sigma(p))
        Sigma(p) = diag(p) - p p^T

    preserve the bias, variance and MSE of the nonlinear pseudoinverse
    regression estimator used throughout this repository?

Sections:

  1. Basic noise models: sanity check that both models reproduce
     E[p_hat] = p and Cov[p_hat] = Sigma(p)/N.
  3. Quantum setting: Haar-random pure states measured by a POVM chosen
     with `POVM_TYPE`, used to build the regression scenario:
       "mub"    -- the complete MUB POVM (``common.mub``), n_out = d(d+1)
                   fixed by `d`;
       "random" -- a Haar-random rank-one POVM (``common.random_povm``) with
                   `N_OUT_RANDOM` outcomes, drawn afresh for every instance.
  4. Regression-level comparison: bias^2 / variance / MSE of the
     pseudoinverse-regression estimator, trained on noisy POVM statistics,
     for both noise models. For every n_tr in `N_TRAIN_VALUES`, `N_REPS`
     independent *instances* of the whole regression scenario are drawn
     (fresh Haar states/POVM/observables every time), so the saved arrays
     expose the instance-to-instance fluctuation, not just one random draw.
  6. Optional finite test statistics (disabled by default): the test-side
     probabilities are also estimated from M shots instead of exact.

Each n_tr uses its own seeded RNG stream (reset to `SEED` before its
sweep), so its results do not depend on the other values in
`N_TRAIN_VALUES` or on their order. The per-instance bias^2 / variance /
MSE arrays of both models, and the MSE of every single noise realization
(one simulated experiment), are saved, one pickle per n_tr, under
``results/`` (see `results_filename`). Run ``plot_noise_comparison.py``
with the same parameters to make the figure.

This module is also imported as a library by the ``check_fit*/`` scripts
(noise models, regression helpers, `_style_axes`).

Only NumPy and Matplotlib are used. The heavy lifting (singular Gaussian
sampling, the multinomial/Gaussian noise models, and the pseudoinverse
regression weights) lives in ``common.py`` as reusable functions.

Run with ``python3 noise_comparison.py`` from anywhere; it locates its
output directory next to itself.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from common import (  # noqa: E402
    chop,
    covariance_rank_info,
    mub,
    multinomial_phat,
    multinomial_phat_batch,
    random_kets,
    random_povm,
    sample_gaussian_from_covariance,
    sigma_batch,
    sigma_from_p,
)
from markers import diamond, triangle  # noqa: E402

# Matplotlib settings used by the plotting helper below, which the
# `check_fit*/` scripts import from this module (`nc.plt`, `nc._style_axes`).
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

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

# model -> (color, shape, legend label); `check_fit*/` scripts reuse the
# multinomial color.
MODEL_STYLE = {
    "mult": ("#5E81B5", diamond, "multinomial"),  # blue, square-look
    "gauss": ("#E19C24", triangle, "Gaussian approx."),  # orange, triangle
}
# ----------------------------------------------------------------------
# Default parameters (all easy to change).
# ----------------------------------------------------------------------

SEED = 0

DIM = 2
POVM_TYPE = "random"  # "mub" (complete MUB POVM) or "random" (Haar-random rank-one POVM)
N_OUT_RANDOM = 16  # number of POVM outcomes when POVM_TYPE == "random" (ignored for "mub")
N_OUT = DIM * (DIM + 1) if POVM_TYPE == "mub" else N_OUT_RANDOM
#N_TRAIN_VALUES = [1,2,3,4,5,6,7,8,9]  # one results file per value
N_TRAIN_VALUES = [1,2,3,4,5,10,15,20,32]  # one results file per value
#N_VALUES = [5, 10, 20, 50, 100, 200, 500, 1000, 2000,10000, 20000, 50000, 100000]
N_VALUES = [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,1000,2000,5000,10000]
N_NOISE_REALIZATIONS = 50  # inner shot-noise realizations per instance
N_REPS = 1  # outer instances (fresh states/POVM/observables each time)
N_TEST_STATES = 50
N_OBSERVABLES = 20
PINV_RCOND = 1e-30  # pseudoinverse tolerance -- same for both noise models (NumPy default; 1e-16 inverts rounding noise)
EIG_TOL = 1e-30  # Sigma eigenvalue tolerance (Gaussian sampling / rank)

# Section 6 (optional finite test statistics): disabled by default.
M_TEST = None


def set_seed(seed=SEED):
    """Seed both RNG mechanisms used across this codebase: the legacy
    global state (``random_kets`` / ``random_unitary`` in common.py use
    ``np.random.randn``) and the Generator instance (``common.rng``, used
    by the multinomial/Gaussian noise-model functions).
    """
    np.random.seed(seed)
    common.rng = np.random.default_rng(seed)


# ========================================================================
# 3. QUANTUM SETTING: Haar states + complete MUB POVM or random POVM.
# ========================================================================
#
# mu_{b,k} = |b,k><b,k| / (d+1), for the d+1 mutually unbiased bases of a
# d-dimensional Hilbert space; n_out = d(d+1). ``common.mub`` builds this
# from the discrete Weyl-Heisenberg (shift/clock) operators X, Z, which
# gives a *complete* set of MUBs when d is prime -- in particular for
# d=2, where the three bases are the eigenbases of the Pauli operators
# X, Z and (up to phase) Y.


def mub_povm_matrix(dim):
    """(n_out, dim**2) row-major-flattened MUB POVM elements; n_out = dim*(dim+1)."""
    return np.array([m.flatten() for m in mub(dim)])


def build_povm(dim, povm_type, n_out):
    """The measurement POVM, as an (n_out, dim**2) row-major-flattened
    matrix: the complete MUB POVM (`n_out` ignored) or a Haar-random
    rank-one POVM with `n_out` outcomes (same as `check_fit.build_povm`).
    """
    if povm_type == "mub":
        return mub_povm_matrix(dim)
    elif povm_type == "random":
        return random_povm(dim, n_out)
    raise ValueError(f"POVM_TYPE must be 'mub' or 'random', got {povm_type!r}")


def flattened_states(dim, n):
    """`n` Haar-random pure-state density matrices, flattened row-major, as rows."""
    kets = random_kets(dim, n)
    return np.array([np.outer(k, k.conj()).flatten() for k in kets])


def povm_probabilities(M_povm, M_states):
    """p_a(rho) = Tr(mu_a rho) for every POVM element a (rows of `M_povm`)
    and every state rho (rows of `M_states`), as a real (n_out, n_states)
    matrix. Tr(mu_a rho) is real because mu_a and rho are both Hermitian;
    `chop` removes the residual floating-point imaginary part.
    """
    return chop(M_povm.conj() @ M_states.T).real


def verify_quantum_setup(dim, povm_type=POVM_TYPE, n_out=N_OUT, n_check=2000):
    """Sanity check: sum_a p_a(rho) = 1 for many Haar-random states."""
    M_mu = build_povm(dim, povm_type, n_out)
    n_out = M_mu.shape[0]
    M_states = flattened_states(dim, n_check)
    P = povm_probabilities(M_mu, M_states)
    max_dev = np.max(np.abs(P.sum(axis=0) - 1))
    print(
        f"[Section 3] d={dim}: n_out={n_out} ({povm_type} POVM), "
        f"max|sum_a p_a - 1| over {n_check} Haar states = {max_dev:.2e}"
    )
    return M_mu, P


# ========================================================================
# 1. BASIC NOISE MODELS -- sanity check on the first two moments.
# ========================================================================


def section1_moment_checks(p, N=100, n_realizations=20_000):
    """Monte Carlo check that, for both noise models,

        E[p_hat] = p                     (unbiasedness)
        Cov[p_hat] = Sigma(p) / N        (correct fluctuation scale)

    ``Sigma(p) = diag(p) - p p^T`` is singular (rank n_out - 1, since
    probabilities sum to 1); `sample_gaussian_from_covariance` handles
    this via an eigendecomposition that discards the numerically-zero
    eigenvalue.
    """
    print("\n=== Section 1: E[p_hat] = p, Cov[p_hat] = Sigma(p)/N ===")
    Sigma = sigma_from_p(p)
    rank, lam_min = covariance_rank_info(Sigma, tol=EIG_TOL)
    print(
        f"p has n_out={len(p)} outcomes; rank(Sigma)={rank} "
        f"(expected {len(p) - 1}), smallest nonzero eigenvalue={lam_min:.3e}"
    )

    mult = multinomial_phat(p, N, size=n_realizations)
    gauss = p + sample_gaussian_from_covariance(Sigma, size=n_realizations, tol=EIG_TOL) / np.sqrt(N)

    for name, sample in (("multinomial", mult), ("Gaussian", gauss)):
        mean_err = np.max(np.abs(sample.mean(axis=0) - p))
        cov_err = np.max(np.abs(np.cov(sample, rowvar=False) - Sigma / N))
        print(
            f"  {name:11s} (N={N}, R={n_realizations}): "
            f"max|E[p_hat]-p|={mean_err:.2e}   max|Cov[p_hat]-Sigma/N|={cov_err:.2e}"
        )


# ========================================================================
# 4. REGRESSION-LEVEL COMPARISON.
# ========================================================================


def build_regression_scenario(dim, n_train, n_test, n_obs, povm_type=POVM_TYPE, n_out=N_OUT):
    """Haar-random training states, test states, and rank-one target
    observables, measured by the POVM `build_povm(dim, povm_type, n_out)`
    in dimension `dim` (for "random", a fresh POVM on every call, i.e. on
    every instance).

    Returns the exact training/test probability matrices (rows = POVM
    outcomes, columns = states) and the exact target values
    y_train[k, i] = Tr(O_k rho_i), y_test[k, j] = Tr(O_k sigma_j) --
    following the same (n_obs, n_states) convention as the sibling
    `fig*_POVM_random.py` / `figN_scaling.py` scripts, so several
    observables and test states are handled at once and the reported MSE
    is averaged over all of them.
    """
    M_mu = build_povm(dim, povm_type, n_out)
    M_rho = flattened_states(dim, n_train)
    M_sigma = flattened_states(dim, n_test)
    M_obs = flattened_states(dim, n_obs)  # rank-one observables |phi><phi|

    P_train = povm_probabilities(M_mu, M_rho)  # (n_out, n_train)
    P_test = povm_probabilities(M_mu, M_sigma)  # (n_out, n_test)

    y_train = chop(M_obs.conj() @ M_rho.T).real  # (n_obs, n_train)
    y_test = chop(M_obs.conj() @ M_sigma.T).real  # (n_obs, n_test) = o_true

    return dict(n_out=M_mu.shape[0], P_train=P_train, P_test=P_test, y_train=y_train, y_test=y_test)


def noisy_training_batches(P_train, N, R, tol=EIG_TOL):
    """R independent noise realizations of the training probability
    matrix, under both models -- each returned as (R, n_out, n_train).

    MULTINOMIAL: P_hat[:,i] = Multinomial(N, P_train[:,i]) / N.
    GAUSSIAN:    P_hat[:,i] = P_train[:,i] + xi_i / sqrt(N),
                 xi_i ~ N(0, Sigma_i),  Sigma_i = diag(P_train[:,i]) - P_train[:,i] P_train[:,i]^T.
    """
    mult_stack = multinomial_phat_batch(P_train, N, R)  # (R, n_out, n_train)

    Sigma_train = sigma_batch(P_train)  # (n_train, n_out, n_out)
    xi = sample_gaussian_from_covariance(Sigma_train, size=R, tol=tol)  # (R, n_train, n_out)
    gauss_stack = P_train[None, :, :] + xi.transpose(0, 2, 1) / np.sqrt(N)  # (R, n_out, n_train)

    return mult_stack, gauss_stack


def noisy_test_probs(P_test, M, model, tol=EIG_TOL):
    """Section 6 (optional finite test statistics): one realization of
    M-shot test-side probabilities, shared across all training-noise
    realizations for a given N (matching the convention used in
    ``fig5_M_saturation.py``).
    """
    if model == "mult":
        return multinomial_phat_batch(P_test, M, R=1)[0]
    Sigma_test = sigma_batch(P_test)
    xi = sample_gaussian_from_covariance(Sigma_test, size=1, tol=tol)[0]  # (n_test, n_out)
    return P_test + xi.T / np.sqrt(M)


def predict_batch(y_train, P_hat_stack, P_test, rcond=PINV_RCOND):
    """W_hat[r] = y_train @ pinv(P_hat_stack[r]);  o_hat[r] = W_hat[r] @ P_test.

    Vectorized over the realization axis R via a batched pseudoinverse
    (``np.linalg.pinv`` supports stacked matrices) instead of a Python
    loop over realizations.
    """
    pinv_stack = np.linalg.pinv(P_hat_stack, rcond=rcond)  # (R, n_train, n_out)
    W_stack = np.einsum("oi,rik->rok", y_train, pinv_stack)  # (R, n_obs, n_out)
    return np.einsum("rok,kt->rot", W_stack, P_test)  # (R, n_obs, n_test)


def bias_variance_mse(o_hat_stack, o_true):
    """bias^2, variance and MSE of o_hat over the realization axis
    (axis 0), averaged over all (observable, test-state) pairs. Checked
    numerically to satisfy MSE = bias^2 + variance.
    """
    mean_o = o_hat_stack.mean(axis=0)
    bias2 = np.mean((mean_o - o_true) ** 2)
    variance = np.mean(o_hat_stack.var(axis=0))
    mse = np.mean((o_hat_stack - o_true[None, :, :]) ** 2)
    return bias2, variance, mse


def regression_sweep(
    dim, N_values, n_train, n_test, n_obs, R, rcond=PINV_RCOND, tol=EIG_TOL, M_test=M_TEST,
    povm_type=POVM_TYPE, n_out=N_OUT, verbose=False,
):
    """Bias^2/variance/MSE of both noise models, swept over `N_values`,
    for a single (dim, n_train, n_test, n_obs) regression scenario (one
    *instance*: a fresh draw of training states, test states and
    observables). Returns {model: {metric: array of shape (len(N_values),)}},
    plus {model: {"mse_real": array of shape (len(N_values), R)}}: the MSE of
    each single noise realization (one "experiment"), averaged over the
    (observable, test-state) pairs only.
    """
    scenario = build_regression_scenario(dim, n_train, n_test, n_obs, povm_type=povm_type, n_out=n_out)
    P_train, P_test = scenario["P_train"], scenario["P_test"]
    y_train, y_test = scenario["y_train"], scenario["y_test"]

    if verbose:
        Sigma_train = sigma_batch(P_train)
        rank, lam_min = covariance_rank_info(Sigma_train, tol=tol)
        print(
            f"[Section 4] d={dim}: {povm_type} POVM, n_out={scenario['n_out']}, n_train={n_train}, "
            f"Sigma_i rank in [{rank.min()},{rank.max()}] (expected {scenario['n_out'] - 1}), "
            f"smallest nonzero eigenvalue over i: {lam_min.min():.3e}"
        )

    results = {model: {"bias2": [], "variance": [], "mse": [], "mse_real": []} for model in ("mult", "gauss")}

    for N in N_values:
        mult_stack, gauss_stack = noisy_training_batches(P_train, N, R, tol=tol)
        stacks = {"mult": mult_stack, "gauss": gauss_stack}

        if verbose and N == N_values[0]:
            sv = np.linalg.svd(mult_stack[0], compute_uv=False)
            print(f"    singular values of P_hat_mult at N={N} (rep. 0): {np.array2string(sv, precision=3)}")

        for model, stack in stacks.items():
            P_test_used = P_test if M_test is None else noisy_test_probs(P_test, M_test, model, tol=tol)
            o_hat_stack = predict_batch(y_train, stack, P_test_used, rcond=rcond)
            bias2, variance, mse = bias_variance_mse(o_hat_stack, y_test)
            results[model]["bias2"].append(bias2)
            results[model]["variance"].append(variance)
            results[model]["mse"].append(mse)
            results[model]["mse_real"].append(np.mean((o_hat_stack - y_test[None, :, :]) ** 2, axis=(1, 2)))

        if verbose:
            check = {
                m: abs(results[m]["mse"][-1] - (results[m]["bias2"][-1] + results[m]["variance"][-1]))
                for m in results
            }
            print(
                f"    N={N:>6d}  mult MSE={results['mult']['mse'][-1]:.3e} "
                f"gauss MSE={results['gauss']['mse'][-1]:.3e}  "
                f"|MSE-(bias2+var)|: mult={check['mult']:.1e} gauss={check['gauss']:.1e}"
            )

    for model in results:
        for key in results[model]:
            results[model][key] = np.array(results[model][key])
    return results


def repeated_regression_sweep(
    dim, N_values, n_train, n_test, n_obs, R, n_reps, rcond=PINV_RCOND, tol=EIG_TOL, M_test=M_TEST,
    povm_type=POVM_TYPE, n_out=N_OUT,
):
    """Run `regression_sweep` `n_reps` independent times -- fresh Haar
    training/test states, POVM and observables every time -- to expose
    the *instance-to-instance* fluctuation of bias^2/variance/MSE, not
    just their value for one particular random scenario.

    Returns {model: {metric: array of shape (n_reps, len(N_values))}}, and
    (n_reps, len(N_values), R) for the per-realization "mse_real".
    """
    all_results = {model: {metric: [] for metric in ("bias2", "variance", "mse", "mse_real")} for model in ("mult", "gauss")}
    for rep in range(n_reps):
        res = regression_sweep(
            dim, N_values, n_train, n_test, n_obs, R, rcond=rcond, tol=tol, M_test=M_test,
            povm_type=povm_type, n_out=n_out, verbose=(rep == 0),
        )
        for model in all_results:
            for metric in all_results[model]:
                all_results[model][metric].append(res[model][metric])
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  instance {rep + 1}/{n_reps} done")

    for model in all_results:
        for metric in all_results[model]:
            all_results[model][metric] = np.array(all_results[model][metric])  # (n_reps, len(N_values)[, R])
    return all_results


# ========================================================================
# 5. PLOTTING HELPER (imported by the check_fit*/ scripts) and the
#    asymptotic theory coefficients.
# ========================================================================


def _style_axes(ax, xlabel, ylabel, log_y=True, x_ticks=(10, 100, 1000)):
    """Grid/tick/locator styling matching the sibling `plot_*.py` scripts."""
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        # Major ticks only at exact decades, labeled "$10^n$"; minor ticks
        # unlabeled. (LogLocator's default would otherwise also label
        # non-decade minor ticks, e.g. "2e-2", whenever the y-range spans
        # less than ~2 decades, as several of these plots do -- and that
        # coefficient+"times"+exponent label mis-renders under usetex,
        # silently dropping the leading digit.)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10), numticks=15))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_labels = [rf"$10^{{{int(np.log10(t))}}}$" for t in x_ticks]
    ax.set_xlim(x_ticks[0] / 3, x_ticks[-1] * 3)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(lambda val, pos: x_labels[x_ticks.index(val)] if val in x_ticks else "")
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def theoretical_bias2_variance(d):
    """Population-averaged bias^2 and variance of the pseudoinverse
    regression estimator for the complete MUB POVM only (not valid for
    POVM_TYPE="random"), exact in the joint
    large-N, large-n_tr limit (eq. 48 of the draft, "Mind the Rank:
    Asymptotic Regime of QLR"):

        E_{sigma,O}[b^2] = (d-1)(d+2)^2 / (d+1)
        E_{sigma,O}[V]   = d(d-1)(d+2) / (d+1)

    These are exactly the coefficients A_d, B_d of the asymptotic scaling
    bias^2 ~ A_d/N^2, variance ~ B_d/(N n_tr) (the draft's fig. 3), so
    N^2*bias^2 and N*n_tr*variance approach these constants as N grows
    (at fixed, large enough n_tr) -- verified numerically to hold at
    n_tr=500 for N in [10^3, 10^4] before finite-N_REALIZATIONS Monte
    Carlo noise starts to dominate the bias^2 estimator at much larger N.
    """
    b2 = (d - 1) * (d + 2) ** 2 / (d + 1)
    variance = d * (d - 1) * (d + 2) / (d + 1)
    return b2, variance


def results_filename(dim, n_train, M_test, povm_type="mub", n_out=None):
    """MUB results keep the original name; random-POVM results also carry
    the POVM type and its number of outcomes, so the two never overwrite
    each other.
    """
    M_str = "inf" if M_test is None else str(M_test)
    povm_str = "mub" if povm_type == "mub" else f"_POVM={povm_type}_nout={n_out}"
    return f"noise_comparison_d={dim}{povm_str}_ntrain={n_train}_M={M_str}.pkl"


def save_results(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  wrote {path}")


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    set_seed(SEED)

    print("=" * 72)
    print("Multinomial shot noise vs. Gaussian approximation -- QELM regression")
    print("=" * 72)

    # Section 3: build a realistic example probability vector p from a
    # single Haar-random state measured by the chosen POVM.
    M_mu, _ = verify_quantum_setup(DIM, POVM_TYPE, N_OUT)
    p = povm_probabilities(M_mu, flattened_states(DIM, 1))[:, 0]
    print(f"[Section 3] example p (d={DIM}, n_out={len(p)}): {np.array2string(p, precision=4)}")

    # Section 1.
    section1_moment_checks(p, N=100, n_realizations=20_000)

    # Section 4: N_REPS independent instances of the regression scenario, per n_tr.
    for n_train in N_TRAIN_VALUES:
        set_seed(SEED)
        print(
            f"\n=== Section 4: regression-level comparison (d={DIM}, {POVM_TYPE} POVM, n_out={N_OUT}, n_tr={n_train}, "
            f"{N_REPS} instances x {N_NOISE_REALIZATIONS} inner noise realizations) ==="
        )
        results = repeated_regression_sweep(
            DIM, N_VALUES, n_train, N_TEST_STATES, N_OBSERVABLES, N_NOISE_REALIZATIONS, N_REPS, M_test=M_TEST,
            povm_type=POVM_TYPE, n_out=N_OUT,
        )
        save_results(
            RESULTS_DIR / results_filename(DIM, n_train, M_TEST, POVM_TYPE, N_OUT),
            dict(
                results=results,  # {model: {metric: array of shape (n_reps, len(N_values))}}
                N_values=np.asarray(N_VALUES),
                dim=DIM, povm_type=POVM_TYPE, n_out=N_OUT, n_train=n_train, M_test=M_TEST,
                n_reps=N_REPS, n_noise_realizations=N_NOISE_REALIZATIONS,
                n_test_states=N_TEST_STATES, n_observables=N_OBSERVABLES, seed=SEED,
            ),
        )

    print("\nDone. Run plot_noise_comparison.py with the same parameters to plot these results.")


if __name__ == "__main__":
    main()
