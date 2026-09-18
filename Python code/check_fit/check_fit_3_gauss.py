"""Same as `check_fit_3.py`, but with Gaussian instead of multinomial shot noise.

One fixed experiment, three noise contexts -- run and save the results.

A single scenario -- one draw of the POVM, training states, test state
and observable (`check_fit.build_povm` / `build_regression_scenario`,
n_test=1, n_obs=1) -- is fixed ONCE, then swept over the training shot
budget N under three different noise contexts applied on top of that
SAME scenario:

  1) M = infinite (exact test statistics), gamma_train = 0, gamma_test = 0
  2) M = M_FINITE,                          gamma_train = 0, gamma_test = 0
  3) M = M_FINITE,                          gamma_train = gamma_test = GAMMA_NONZERO

The only difference from `check_fit_3.py` is the training-side shot
noise itself: instead of true multinomial shot noise, each training
state's estimated probabilities use the singular Gaussian approximation
(see `noise_comparison.py`'s module docstring / `common.py`'s
`sample_gaussian_from_covariance`):

    p_hat = p + xi / sqrt(N),      xi ~ N(0, Sigma(p))
    Sigma(p) = diag(p) - p p^T

For each context, N_REPEATS such Gaussian-noise realizations are drawn
per N and every individual squared error is kept (as in
`check_fit_single.py`), not just their mean. Since training states/test
state/POVM/observable are shared across all three contexts, any
difference between their results comes only from the noise model (M,
gamma, gamma_test), not from a different underlying quantum scenario --
and any difference from `check_fit_3.py`'s results (same parameters)
isolates the effect of the multinomial-vs-Gaussian shot-noise model
itself.

This script only runs the simulation and saves the raw results (every
context's raw squared-error array, plus every parameter needed to
recompute their fits) to a pickle file under ``results/``, named after
the identifying parameters -- see `results_filename` (prefixed
"check_fit_3_gauss_", distinct from `check_fit_3.py`'s "check_fit_3_",
so the two noise models' results for the same parameters never collide
and can be compared side by side).

Run with ``python3 check_fit_3_gauss.py`` from anywhere; it locates its
output directory next to itself.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # check_fit.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import check_fit as cf  # noqa: E402  (build_povm, build_regression_scenario, save_results)
import noise_comparison as nc  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

# ----------------------------------------------------------------------
# Parameters (all easy to change) -- same as check_fit_3.py by default,
# so the two scripts' results are directly comparable.
# ----------------------------------------------------------------------

SEED = None  # fixes the ONE scenario (POVM/train states/test state/observable) shared by all three contexts

DIM = 2
POVM_TYPE = "mub"  # "mub" (complete MUB POVM) or "random" (Haar-random rank-one POVM)
N_OUT_RANDOM = 6  # number of POVM outcomes when POVM_TYPE == "random" (ignored for "mub")
N_OUT = DIM * (DIM + 1) if POVM_TYPE == "mub" else N_OUT_RANDOM
N_TRAIN = 1024  # must be > N_OUT + 1 for the fit's cross term to be defined

M_FINITE = 10000  # finite test-shot budget used in contexts 2 and 3
GAMMA_NONZERO = 0.05  # gamma_train = gamma_test used in context 3

N_VALUES = sorted(set(np.round(np.logspace(np.log10(5), np.log10(1000000), 16)).astype(int).tolist()))
N_REPEATS = 100  # shot-noise realizations of the SAME context, per N (as in check_fit_single.py)

# context -> (gamma_train, gamma_test, M, color, label). Colors reuse the
# blue/orange/green from `noise_comparison.MODEL_STYLE`/the nout=4 entry
# in `POVM_random/plot_POVM_random.py`.
CONTEXTS = [
    dict(gamma=0.0, gamma_test=0.0, M=None, color="#5E81B5", label=r"1) $M=\infty$, $\gamma=0$"),
    dict(gamma=0.0, gamma_test=0.0, M=M_FINITE, color="#E19C24", label=rf"2) $M={M_FINITE}$, $\gamma=0$"),
    dict(
        gamma=GAMMA_NONZERO, gamma_test=GAMMA_NONZERO, M=M_FINITE, color="#8FB032",
        label=rf"3) $M={M_FINITE}$, $\gamma=\gamma_{{\rm test}}={GAMMA_NONZERO}$",
    ),
]


def results_filename(dim, povm_type, n_out, n_train, m_finite, gamma_nonzero):
    """Filename encoding exactly the parameters needed to locate/recompute
    this run's fits -- prefixed "check_fit_3_gauss_" (distinct from
    `check_fit_3.results_filename`'s "check_fit_3_") so the multinomial
    and Gaussian results for the same parameters never collide. Not
    encoded: N_REPEATS/SEED (simulation settings that don't enter the
    fit) -- re-running with those changed but the same identifying
    parameters overwrites the previous file.
    """
    return (
        f"check_fit_3_gauss_povm={povm_type}_d={dim}_nout={n_out}_ntrain={n_train}"
        f"_M={m_finite}_gamma={gamma_nonzero}.pkl"
    )


def run_context(P_train, Sigma_train, P_test, y_train_exact, y_test_exact, gamma, gamma_test, M):
    """The shared scenario's (P_train, Sigma_train, P_test, y_train_exact,
    y_test_exact) under one noise context: apply this context's label
    noise (once) and test-side M-shot noise (once), then for each N draw
    N_REPEATS independent Gaussian shot-noise realizations -- p_hat = p +
    xi/sqrt(N), xi ~ N(0, Sigma(p)) -- and keep every individual squared
    error (not averaged), exactly as `check_fit_3.run_context` does for
    the multinomial model.

    `Sigma_train = noise_comparison.sigma_batch(P_train)` doesn't depend
    on the context (only on the exact P_train), so it's computed once in
    `main` and shared across all three contexts' calls to this function.

    Returns an (N_REPEATS, len(N_VALUES)) array.
    """
    y_train = y_train_exact + gamma * np.random.randn(*y_train_exact.shape)
    y_test = y_test_exact + gamma_test * np.random.randn(*y_test_exact.shape)
    P_test_used = P_test if M is None else nc.noisy_test_probs(P_test, M, "mult", tol=nc.EIG_TOL)

    sq_errors = np.empty((N_REPEATS, len(N_VALUES)))
    for j, N in enumerate(N_VALUES):
        xi = nc.sample_gaussian_from_covariance(Sigma_train, size=N_REPEATS, tol=nc.EIG_TOL)  # (N_REPEATS, n_train, n_out)
        gauss_stack = P_train[None, :, :] + xi.transpose(0, 2, 1) / np.sqrt(N)  # (N_REPEATS, n_out, n_train)
        o_hat_stack = nc.predict_batch(y_train, gauss_stack, P_test_used, rcond=nc.PINV_RCOND)  # (N_REPEATS, 1, 1)
        sq_errors[:, j] = (o_hat_stack[:, 0, 0] - y_test[0, 0]) ** 2

    return sq_errors


def main():
    print("=" * 72)
    print("One experiment, three noise contexts, Gaussian shot noise (n_test=1, n_obs=1)")
    print("=" * 72)
    print(f"povm={POVM_TYPE}  d={DIM}  n_out={N_OUT}  n_train={N_TRAIN}  N_REPEATS={N_REPEATS}  seed={SEED}")

    nc.set_seed(SEED)
    M_mu = cf.build_povm(DIM, POVM_TYPE, N_OUT)
    scenario = cf.build_regression_scenario(M_mu, DIM, N_TRAIN, 1, 1)
    P_train, P_test = scenario["P_train"], scenario["P_test"]
    y_train_exact, y_test_exact = scenario["y_train"], scenario["y_test"]
    Sigma_train = nc.sigma_batch(P_train)  # (n_train, n_out, n_out); shared by every context

    sq_errors_list = []
    for ctx in CONTEXTS:
        print(f"\n--- {ctx['label']} ---")
        sq_errors = run_context(
            P_train, Sigma_train, P_test, y_train_exact, y_test_exact, ctx["gamma"], ctx["gamma_test"], ctx["M"]
        )
        for N, mse in zip(N_VALUES, sq_errors.mean(axis=0)):
            print(f"  N={N:>6d}  MSE_empirical={mse:.4e}")
        sq_errors_list.append(sq_errors)

    results = dict(
        N_values=np.asarray(N_VALUES),
        contexts=CONTEXTS,
        sq_errors_list=sq_errors_list,
        dim=DIM,
        povm_type=POVM_TYPE,
        n_out=N_OUT,
        n_train=N_TRAIN,
        n_repeats=N_REPEATS,
        seed=SEED,
        m_finite=M_FINITE,
        gamma_nonzero=GAMMA_NONZERO,
        noise_model="gauss",
    )
    filename = results_filename(DIM, POVM_TYPE, N_OUT, N_TRAIN, M_FINITE, GAMMA_NONZERO)
    print("\n=== saving results ===")
    cf.save_results(RESULTS_DIR / filename, results)

    print(
        "\nDone. Update plot_check_fit_3.py to import check_fit_3_gauss "
        "(and use its results_filename) to plot these results."
    )


if __name__ == "__main__":
    main()
