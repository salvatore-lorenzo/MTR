"""Fig. SM1: multinomial shot noise vs. its Gaussian approximation, for
pseudoinverse quantum linear regression -- run and save.

Does replacing the true multinomial shot noise

    p_hat = Multinomial(N, p) / N

with the singular Gaussian approximation

    p_hat = p + xi / sqrt(N),      xi ~ N(0, Sigma(p)),
    Sigma(p) = diag(p) - p p^T

preserve the error statistics of the pseudoinverse regression estimator?

For each measurement in `RUNS` (the qubit MUB POVM, and a Haar-random
rank-one POVM with 16 outcomes, drawn afresh for every instance) and each
n_tr in `N_TRAIN_VALUES`, Haar-random training states, test states and
rank-one observables are drawn, and for every N in `N_VALUES` both noise
models are applied to the same exact training probabilities. For each
noise realization (one simulated experiment) the MSE averaged over test
states and observables is stored, together with bias^2 / variance / MSE
over realizations. Optionally (`M_TEST`), the test-side probabilities are
also estimated from M shots instead of being exact.

Each (run, n_tr) pair uses its own RNG stream, reset to the run's seed,
so its results do not depend on the other values in `N_TRAIN_VALUES`.
Results are saved, one pickle per (run, n_tr), under ``data/`` (see
`results_filename`). Run ``plot_noise_comparison.py`` to make the figure.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    covariance_rank_info,
    flattened_states,
    mub,
    multinomial_phat,
    multinomial_phat_batch,
    povm_probabilities,
    random_povm,
    sample_gaussian_from_covariance,
    set_seed,
    sigma_batch,
    sigma_from_p,
)

RESULTS_DIR = Path(__file__).resolve().parent / "data"

# ----------------------------------------------------------------------
# Parameters.
# ----------------------------------------------------------------------

DIM = 2
# (POVM type, number of outcomes, seed) of each run: "mub" is the complete
# MUB POVM (n_out = d(d+1)), "random" a Haar-random rank-one POVM.
RUNS = (
    ("mub", DIM * (DIM + 1), 142),
    ("random", 16, 0),
)
N_TRAIN_VALUES = [1, 2, 3, 4, 5, 10, 15, 20, 32]  # one results file per value
N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
N_NOISE_REALIZATIONS = 50  # shot-noise realizations (experiments) per instance
N_REPS = 1  # instances (fresh states/POVM/observables each time)
N_TEST_STATES = 50
N_OBSERVABLES = 20
PINV_RCOND = 1e-30  # pseudoinverse tolerance, same for both noise models
EIG_TOL = 1e-30  # Sigma eigenvalue tolerance (Gaussian sampling / rank)
M_TEST = None  # test shots (None = exact test probabilities)


# ========================================================================
# Measurement and regression scenario.
# ========================================================================


def build_povm(dim, povm_type, n_out):
    """The measurement POVM, as an (n_out, dim**2) row-major-flattened
    matrix: the complete MUB POVM (`n_out` ignored) or a Haar-random
    rank-one POVM with `n_out` outcomes.
    """
    if povm_type == "mub":
        return np.array([m.flatten() for m in mub(dim)])
    elif povm_type == "random":
        return random_povm(dim, n_out)
    raise ValueError(f"povm_type must be 'mub' or 'random', got {povm_type!r}")


def verify_quantum_setup(dim, povm_type, n_out, n_check=2000):
    """Sanity check: sum_a p_a(rho) = 1 for many Haar-random states."""
    M_mu = build_povm(dim, povm_type, n_out)
    n_out = M_mu.shape[0]
    P = povm_probabilities(M_mu, flattened_states(dim, n_check))
    max_dev = np.max(np.abs(P.sum(axis=0) - 1))
    print(
        f"d={dim}: n_out={n_out} ({povm_type} POVM), "
        f"max|sum_a p_a - 1| over {n_check} Haar states = {max_dev:.2e}"
    )
    return M_mu, P


def moment_checks(p, N=100, n_realizations=20_000):
    """Monte Carlo check that, for both noise models, E[p_hat] = p and
    Cov[p_hat] = Sigma(p) / N. ``Sigma(p)`` is singular (rank n_out - 1);
    `sample_gaussian_from_covariance` handles this by discarding the
    numerically-zero eigenvalue.
    """
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


def build_regression_scenario(dim, n_train, n_test, n_obs, povm_type, n_out):
    """Haar-random training states, test states and rank-one target
    observables, measured by `build_povm(dim, povm_type, n_out)` (for
    "random", a fresh POVM on every call, i.e. on every instance).

    Returns the exact training/test probability matrices (rows = POVM
    outcomes, columns = states) and the exact targets
    y_train[k, i] = Tr(O_k rho_i), y_test[k, j] = Tr(O_k sigma_j).
    """
    M_mu = build_povm(dim, povm_type, n_out)
    M_rho = flattened_states(dim, n_train)
    M_sigma = flattened_states(dim, n_test)
    M_obs = flattened_states(dim, n_obs)  # rank-one observables |phi><phi|

    P_train = povm_probabilities(M_mu, M_rho)  # (n_out, n_train)
    P_test = povm_probabilities(M_mu, M_sigma)  # (n_out, n_test)
    y_train = povm_probabilities(M_obs, M_rho)  # (n_obs, n_train)
    y_test = povm_probabilities(M_obs, M_sigma)  # (n_obs, n_test)

    return dict(n_out=M_mu.shape[0], P_train=P_train, P_test=P_test, y_train=y_train, y_test=y_test)


# ========================================================================
# Noise models and regression.
# ========================================================================


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
    """One realization of M-shot test-side probabilities (under the given
    noise model), shared across all training-noise realizations.
    """
    if model == "mult":
        return multinomial_phat_batch(P_test, M, R=1)[0]
    Sigma_test = sigma_batch(P_test)
    xi = sample_gaussian_from_covariance(Sigma_test, size=1, tol=tol)[0]  # (n_test, n_out)
    return P_test + xi.T / np.sqrt(M)


def predict_batch(y_train, P_hat_stack, P_test, rcond=PINV_RCOND):
    """W_hat[r] = y_train @ pinv(P_hat_stack[r]);  o_hat[r] = W_hat[r] @ P_test,
    vectorized over the realization axis r.
    """
    pinv_stack = np.linalg.pinv(P_hat_stack, rcond=rcond)  # (R, n_train, n_out)
    W_stack = np.einsum("oi,rik->rok", y_train, pinv_stack)  # (R, n_obs, n_out)
    return np.einsum("rok,kt->rot", W_stack, P_test)  # (R, n_obs, n_test)


def bias_variance_mse(o_hat_stack, o_true):
    """bias^2, variance and MSE of o_hat over the realization axis
    (axis 0), averaged over all (observable, test-state) pairs.
    """
    mean_o = o_hat_stack.mean(axis=0)
    bias2 = np.mean((mean_o - o_true) ** 2)
    variance = np.mean(o_hat_stack.var(axis=0))
    mse = np.mean((o_hat_stack - o_true[None, :, :]) ** 2)
    return bias2, variance, mse


def regression_sweep(dim, N_values, n_train, n_test, n_obs, R, povm_type, n_out, M_test=M_TEST, verbose=False):
    """Bias^2/variance/MSE of both noise models, swept over `N_values`, for
    one instance of the regression scenario. Returns {model: {metric:
    array of shape (len(N_values),)}}, plus {model: {"mse_real": array of
    shape (len(N_values), R)}}: the MSE of each single noise realization
    (one experiment), averaged over the (observable, test-state) pairs.
    """
    scenario = build_regression_scenario(dim, n_train, n_test, n_obs, povm_type, n_out)
    P_train, P_test = scenario["P_train"], scenario["P_test"]
    y_train, y_test = scenario["y_train"], scenario["y_test"]

    if verbose:
        rank, lam_min = covariance_rank_info(sigma_batch(P_train), tol=EIG_TOL)
        print(
            f"  Sigma_i rank in [{rank.min()},{rank.max()}] (expected {scenario['n_out'] - 1}), "
            f"smallest nonzero eigenvalue over i: {lam_min.min():.3e}"
        )

    results = {model: {"bias2": [], "variance": [], "mse": [], "mse_real": []} for model in ("mult", "gauss")}

    for N in N_values:
        mult_stack, gauss_stack = noisy_training_batches(P_train, N, R)

        for model, stack in (("mult", mult_stack), ("gauss", gauss_stack)):
            P_test_used = P_test if M_test is None else noisy_test_probs(P_test, M_test, model)
            o_hat_stack = predict_batch(y_train, stack, P_test_used)
            bias2, variance, mse = bias_variance_mse(o_hat_stack, y_test)
            results[model]["bias2"].append(bias2)
            results[model]["variance"].append(variance)
            results[model]["mse"].append(mse)
            results[model]["mse_real"].append(np.mean((o_hat_stack - y_test[None, :, :]) ** 2, axis=(1, 2)))

        if verbose:
            print(
                f"    N={N:>6d}  mult MSE={results['mult']['mse'][-1]:.3e} "
                f"gauss MSE={results['gauss']['mse'][-1]:.3e}"
            )

    for model in results:
        for key in results[model]:
            results[model][key] = np.array(results[model][key])
    return results


def repeated_regression_sweep(dim, N_values, n_train, n_test, n_obs, R, n_reps, povm_type, n_out, M_test=M_TEST):
    """Run `regression_sweep` on `n_reps` independent instances. Returns
    {model: {metric: array of shape (n_reps, len(N_values))}}, and
    (n_reps, len(N_values), R) for the per-realization "mse_real".
    """
    all_results = {model: {metric: [] for metric in ("bias2", "variance", "mse", "mse_real")} for model in ("mult", "gauss")}
    for rep in range(n_reps):
        res = regression_sweep(
            dim, N_values, n_train, n_test, n_obs, R, povm_type, n_out, M_test=M_test, verbose=(rep == 0),
        )
        for model in all_results:
            for metric in all_results[model]:
                all_results[model][metric].append(res[model][metric])

    for model in all_results:
        for metric in all_results[model]:
            all_results[model][metric] = np.array(all_results[model][metric])
    return all_results


# ========================================================================
# Saving / loading.
# ========================================================================


def results_filename(dim, n_train, M_test, povm_type, n_out):
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
    for povm_type, n_out, seed in RUNS:
        print(f"\n=== {povm_type} POVM, n_out={n_out}, seed={seed} ===")
        set_seed(seed)

        # Sanity checks on an example probability vector p.
        M_mu, _ = verify_quantum_setup(DIM, povm_type, n_out)
        p = povm_probabilities(M_mu, flattened_states(DIM, 1))[:, 0]
        moment_checks(p)

        for n_train in N_TRAIN_VALUES:
            set_seed(seed)
            print(f"n_tr={n_train}: {N_REPS} instance(s) x {N_NOISE_REALIZATIONS} noise realizations")
            results = repeated_regression_sweep(
                DIM, N_VALUES, n_train, N_TEST_STATES, N_OBSERVABLES, N_NOISE_REALIZATIONS, N_REPS,
                povm_type, n_out, M_test=M_TEST,
            )
            save_results(
                RESULTS_DIR / results_filename(DIM, n_train, M_TEST, povm_type, n_out),
                dict(
                    results=results,
                    N_values=np.asarray(N_VALUES),
                    dim=DIM, povm_type=povm_type, n_out=n_out, n_train=n_train, M_test=M_TEST,
                    n_reps=N_REPS, n_noise_realizations=N_NOISE_REALIZATIONS,
                    n_test_states=N_TEST_STATES, n_observables=N_OBSERVABLES, seed=seed,
                ),
            )


if __name__ == "__main__":
    main()
