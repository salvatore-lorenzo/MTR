"""Check the closed-form MSE fit against a single simulated experiment.

A single Random-POVM regression experiment -- one draw of Haar training
states, POVM, a single test state and a single observable, i.e. one
instance with n_test = n_obs = 1 -- is swept over the training shot
budget N (true multinomial noise, `N_REAL` shot-noise realizations per
N). The resulting MSE(N) curve is plotted against the closed-form
prediction

    MSE_fit(N) = A_d / [N + d(d+2)]^2
               + C_d / (N n_tr) * [1 + (d^2-1) (N / (N + d(d+2)))^2]
               + d^2 gamma^2 / n_tr
               + C_d / M
               + (N gamma^2 / M) * (n_out - d^2) / (n_tr - n_out - 1)

with

    A_d = (d-1)(d+2)^2 / (d+1)
    B_d = d(d-1)(d+2) / (d+1)
    C_d = B_d / d^2 = (d-1)(d+2) / (d(d+1))

n_tr is the training-set size, gamma is the additive Gaussian label
(target) noise std on the training targets (`y -> y + gamma * N(0,1)`,
test targets exact), and M is the number of shots used to estimate the
(single) test state's probabilities (`M=None` means exact/infinite-shot
test statistics, for which the two M-dependent terms vanish).

The measurement POVM (`POVM_TYPE`) is either:
  "mub"    -- the complete MUB POVM (same as noise_comparison.py),
              n_out = d(d+1) fixed by `d`.
  "random" -- a fresh Haar-random rank-one POVM (``common.random_povm``)
              with an arbitrary `N_OUT_RANDOM` outcomes.
Note the fit's A_d/B_d/C_d coefficients (`theoretical_bias2_variance`)
were derived for the complete MUB POVM specifically; n_out only enters
the fit explicitly through the last (cross) term, so running with
POVM_TYPE="random" is itself a check of whether that MUB-derived fit
still tracks a generic (non-MUB) rank-one POVM.

`N_EXPERIMENTS` independent repetitions of the single-instance experiment
above (each a fresh draw of the POVM, training/test states, observable,
label noise and test-shot noise) are run. This script only runs the
simulation and saves the raw results (every experiment's MSE(N) curve,
plus every parameter `mse_fit` needs) to a pickle file under
``results/``, named after the fit parameters -- see `results_filename`.
Run ``plot_check_fit.py`` (same parameters) to load that file and make
the plot, without re-running the simulation.

Run with ``python3 check_fit.py`` from anywhere; it locates its output
directory next to itself.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import common  # noqa: E402
import noise_comparison as nc  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

# ----------------------------------------------------------------------
# Parameters (all easy to change).
# ----------------------------------------------------------------------

SEED = None

DIM = 2
POVM_TYPE = "random"  # "mub" (complete MUB POVM) or "random" (Haar-random rank-one POVM)
N_OUT_RANDOM = 64  # number of POVM outcomes when POVM_TYPE == "random" (ignored for "mub")
N_OUT = DIM * (DIM + 1) if POVM_TYPE == "mub" else N_OUT_RANDOM
N_TRAIN = 640  # must be > N_OUT + 1 for the fit's cross term to be defined
GAMMA = 0.05  # additive Gaussian label noise std on the training targets
GAMMA_TEST = 0.05  # additive Gaussian label noise std on the test target (0 = exact test target)
M_TEST = 1000  # test-side shot budget (None = exact test statistics)

N_VALUES = sorted(set(np.round(np.logspace(np.log10(5), np.log10(1000000), 16)).astype(int).tolist()))
N_REAL = 100  # shot-noise realizations per N (single test state/observable -> needs many to smooth out)
N_EXPERIMENTS = 10  # independent repetitions of the single-instance experiment


def results_filename(dim, povm_type, n_out, n_train, gamma, gamma_test, M):
    """Filename encoding exactly the parameters `mse_fit` needs -- so a
    `plot_check_fit.py` run with the same parameters locates the same
    file. Not encoded: N_EXPERIMENTS/N_REAL/SEED (simulation settings
    that don't enter the fit) -- re-running with those changed but the
    same fit parameters overwrites the previous file.
    """
    M_str = "inf" if M is None else str(M)
    return (
        f"mse_povm={povm_type}_d={dim}_nout={n_out}_ntrain={n_train}"
        f"_gamma={gamma}_gammatest={gamma_test}_M={M_str}.pkl"
    )


def save_results(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  wrote {path}")


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def mse_fit_terms(N, d, n_tr, n_out, gamma, gamma_test, M):
    """The individual terms of the closed-form MSE prediction -- see module
    docstring -- as a dict {term_name: array}; `mse_fit` is their sum.
    """
    Ad, Bd = nc.theoretical_bias2_variance(d)
    Cd = Bd / d**2
    N = np.asarray(N, dtype=float)

    test_shot_noise = 0.0 if M is None else Cd / M
    terms = {
        "bias": Ad / (N + d * (d + 2)) ** 2,
        "train_shot_variance": Cd / (N * n_tr) * (1 + (d**2 - 1) * (N / (N + d * (d + 2))) ** 2),
        "train_label_noise": np.full_like(N, d**2 * gamma**2 / n_tr),
        "test_noise": np.full_like(N, 0*gamma_test**2 + test_shot_noise),  # test label noise + test shot noise
    }
    terms["cross"] = np.zeros_like(N) if M is None else (N * gamma**2 / M) * (n_out - d**2) / (n_tr - n_out - 1)
    return terms


def mse_fit(N, d, n_tr, n_out, gamma, gamma_test, M):
    """Closed-form MSE prediction -- see module docstring."""
    terms = mse_fit_terms(N, d, n_tr, n_out, gamma, gamma_test, M)
    return sum(terms.values())


def build_povm(dim, povm_type, n_out):
    """The measurement POVM, as an (n_out, dim**2) row-major-flattened
    matrix (same convention as `noise_comparison.mub_povm_matrix`).
    """
    if povm_type == "mub":
        return nc.mub_povm_matrix(dim)
    elif povm_type == "random":
        return common.random_povm(dim, n_out)
    raise ValueError(f"POVM_TYPE must be 'mub' or 'random', got {povm_type!r}")


def build_regression_scenario(M_mu, dim, n_train, n_test, n_obs):
    """Same as `noise_comparison.build_regression_scenario`, but with the
    POVM matrix `M_mu` passed in rather than always the complete MUB POVM,
    so a random rank-one POVM (`build_povm`) can be used as well.
    """
    M_rho = nc.flattened_states(dim, n_train)
    M_sigma = nc.flattened_states(dim, n_test)
    M_obs = nc.flattened_states(dim, n_obs)

    P_train = nc.povm_probabilities(M_mu, M_rho)
    P_test = nc.povm_probabilities(M_mu, M_sigma)

    y_train = (M_obs.conj() @ M_rho.T).real
    y_test = (M_obs.conj() @ M_sigma.T).real

    return dict(n_out=M_mu.shape[0], P_train=P_train, P_test=P_test, y_train=y_train, y_test=y_test)


def run_single_experiment(seed):
    """One Haar-random regression instance -- POVM, single training states,
    single test state, single observable, one draw of the training-target
    label noise (`gamma`), one draw of the test-target label noise
    (`gamma_test`), and one draw of the test-side M-shot noise -- all fixed
    once here, up front. The N-sweep below then changes exactly one thing,
    the training-side shot-noise sampling: `N_REAL` multinomial(N, P_train)
    realizations for each N in `N_VALUES`.
    """
    nc.set_seed(seed)
    M_mu = build_povm(DIM, POVM_TYPE, N_OUT)
    scenario = build_regression_scenario(M_mu, DIM, N_TRAIN, 1, 1)
    P_train, P_test = scenario["P_train"], scenario["P_test"]
    y_train = scenario["y_train"] + GAMMA * np.random.randn(*scenario["y_train"].shape)
    y_test = scenario["y_test"] + GAMMA_TEST * np.random.randn(*scenario["y_test"].shape)

    # Fixed once, not redrawn as N sweeps: the single M-shot noisy read of
    # the (single) test state's probabilities.
    P_test_used = P_test if M_TEST is None else nc.noisy_test_probs(P_test, M_TEST, "mult", tol=nc.EIG_TOL)

    mse_empirical = []
    for N in N_VALUES:
        mult_stack = nc.multinomial_phat_batch(P_train, N, N_REAL)  # (N_REAL, n_out, n_train)
        o_hat_stack = nc.predict_batch(y_train, mult_stack, P_test_used, rcond=nc.PINV_RCOND)
        _, _, mse = nc.bias_variance_mse(o_hat_stack, y_test)
        mse_empirical.append(mse)
        print(f"  N={N:>6d}  MSE_empirical={mse:.4e}")

    return np.array(mse_empirical)


def run_experiments(n_experiments):
    """`n_experiments` independent repetitions of `run_single_experiment`,
    each with its own seed (so they're independent even when `SEED` is
    fixed). Returns an (n_experiments, len(N_VALUES)) array.
    """
    results = []
    for i in range(n_experiments):
        seed = None if SEED is None else SEED + i
        print(f"\n--- experiment {i + 1}/{n_experiments} (seed={seed}) ---")
        results.append(run_single_experiment(seed))
    return np.array(results)


def main():
    print("=" * 72)
    print(f"MSE fit check -- {N_EXPERIMENTS} independent POVM experiments (n_test=1, n_obs=1)")
    print("=" * 72)
    print(
        f"povm={POVM_TYPE}  d={DIM}  n_out={N_OUT}  n_train={N_TRAIN}  "
        f"gamma={GAMMA}  gamma_test={GAMMA_TEST}  M={M_TEST}  N_REAL={N_REAL}"
    )

    mse_empirical = run_experiments(N_EXPERIMENTS)

    results = dict(
        N_values=np.asarray(N_VALUES),
        mse_empirical=mse_empirical,
        dim=DIM,
        povm_type=POVM_TYPE,
        n_out=N_OUT,
        n_train=N_TRAIN,
        gamma=GAMMA,
        gamma_test=GAMMA_TEST,
        M_test=M_TEST,
        n_experiments=N_EXPERIMENTS,
        n_real=N_REAL,
        seed=SEED,
    )
    filename = results_filename(DIM, POVM_TYPE, N_OUT, N_TRAIN, GAMMA, GAMMA_TEST, M_TEST)
    print("\n=== saving results ===")
    save_results(RESULTS_DIR / filename, results)

    print("\nDone. Run plot_check_fit.py with the same parameters to plot these results.")


if __name__ == "__main__":
    main()
