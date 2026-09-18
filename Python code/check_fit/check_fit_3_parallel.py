"""Parallelized version of `check_fit_3.py` -- same output, run across cores.

Same experiment/contexts as `check_fit_3.py` (see its docstring): one
shared scenario (POVM, training states, test state, observable), swept
over N under three noise contexts, N_REPEATS multinomial shot-noise
realizations kept per (context, N) point.

`check_fit_3.py` runs its 3 contexts x len(N_VALUES) points sequentially.
Here, the one-time-per-context setup (label noise, test-shot noise --
must stay sequential, since it advances shared global RNG state) still
runs in the main process, but the expensive per-(context, N) work --
drawing N_REPEATS multinomial(N, P_train) realizations and the batched
pseudoinverse regression -- is farmed out across a `ProcessPoolExecutor`,
one task per (context, N) pair (`len(CONTEXTS) * len(N_VALUES)` tasks
total). Each worker uses its own local `numpy.random.Generator` (seeded
deterministically from `(SEED, context_index, N_index)` when `SEED` is
set) instead of the shared global RNG the sequential script uses, since
worker processes don't share memory -- so with a fixed SEED this is
reproducible run-to-run, just not bit-identical to `check_fit_3.py`'s
own sequential draws for that inner loop.

The output pickle has exactly the same schema (same keys/shapes,
written via `check_fit_3.results_filename` / `check_fit.save_results`)
as `check_fit_3.py`'s, so `plot_check_fit_3.py` (unchanged) can load and
plot it exactly the same way, regardless of which of the two scripts
produced it.

Run with ``python3 check_fit_3_parallel.py`` from anywhere; it locates
its output directory next to itself.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # check_fit.py, check_fit_3.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import check_fit as cf  # noqa: E402  (build_povm, build_regression_scenario, save_results)
import check_fit_3 as cf3  # noqa: E402  (results_filename, RESULTS_DIR -- reused so the output matches exactly)
import noise_comparison as nc  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

# ----------------------------------------------------------------------
# Parameters (all easy to change) -- same values as check_fit_3.py, so a
# default run targets the same results file.
# ----------------------------------------------------------------------

SEED = None  # fixes the ONE scenario (POVM/train states/test state/observable) shared by all three contexts

DIM = 10
POVM_TYPE = "random"  # "mub" (complete MUB POVM) or "random" (Haar-random rank-one POVM)
N_OUT_RANDOM = 1000  # number of POVM outcomes when POVM_TYPE == "random" (ignored for "mub")
N_OUT = DIM * (DIM + 1) if POVM_TYPE == "mub" else N_OUT_RANDOM
N_TRAIN = 5000  # must be > N_OUT + 1 for the fit's cross term to be defined

M_FINITE = 10000  # finite test-shot budget used in contexts 2 and 3
GAMMA_NONZERO = 0.05  # gamma_train = gamma_test used in context 3

N_VALUES = sorted(set(np.round(np.logspace(np.log10(5), np.log10(1000000), 16)).astype(int).tolist()))
N_REPEATS = 50  # shot-noise realizations of the SAME context, per N (as in check_fit_single.py)

N_WORKERS = 8 # os.cpu_count() or 4  # worker processes in the pool

# context -> (gamma_train, gamma_test, M, color, label). Same as check_fit_3.CONTEXTS.
CONTEXTS = [
    dict(gamma=0.0, gamma_test=0.0, M=None, color="#5E81B5", label=r"1) $M=\infty$, $\gamma=0$"),
    dict(gamma=0.0, gamma_test=0.0, M=M_FINITE, color="#E19C24", label=rf"2) $M={M_FINITE}$, $\gamma=0$"),
    dict(
        gamma=GAMMA_NONZERO, gamma_test=GAMMA_NONZERO, M=M_FINITE, color="#8FB032",
        label=rf"3) $M={M_FINITE}$, $\gamma=\gamma_{{\rm test}}={GAMMA_NONZERO}$",
    ),
]


def _multinomial_phat_batch_local(rng, P, N, R):
    """Same as `common.multinomial_phat_batch`, but takes an explicit
    `numpy.random.Generator` instead of using the shared global
    `common.rng` -- required for correctness here, since each worker
    process has its own independent copy of that global, not seeded the
    way the main process's was.
    """
    Pt = np.real(np.asarray(P)).T  # (n_cols, n_out)
    pvals = np.broadcast_to(Pt, (R,) + Pt.shape)
    counts = rng.multinomial(N, pvals)  # (R, n_cols, n_out)
    return counts.transpose(0, 2, 1) / N


def _sq_errors_for_task(task):
    """One (context, N) unit of work, run in a worker process: draw
    `n_repeats` independent multinomial(N, P_train) shot-noise
    realizations and return their squared errors -- the same computation
    as one iteration of `check_fit_3.run_context`'s N-loop, just with an
    explicit local RNG instead of the shared global one.
    """
    ctx_idx, n_idx, P_train, y_train, P_test_used, y_test_scalar, N, n_repeats, seed = task
    rng = np.random.default_rng(seed)
    mult_stack = _multinomial_phat_batch_local(rng, P_train, N, n_repeats)  # (n_repeats, n_out, n_train)
    o_hat_stack = nc.predict_batch(y_train, mult_stack, P_test_used, rcond=nc.PINV_RCOND)  # (n_repeats, 1, 1)
    sq_errors = (o_hat_stack[:, 0, 0] - y_test_scalar) ** 2
    return ctx_idx, n_idx, sq_errors


def setup_context(P_train, P_test, y_train_exact, y_test_exact, gamma, gamma_test, M):
    """Everything that must happen ONCE per context, sequentially, in the
    main process -- it advances the shared global RNG state
    (`np.random`/`common.rng`), so it can't safely be parallelized the
    way the N-loop below is. Returns (y_train_noisy, y_test_scalar,
    P_test_used): the fixed inputs shared by every N in that context's
    sweep -- exactly what `check_fit_3.run_context` computes up front.
    """
    y_train = y_train_exact + gamma * np.random.randn(*y_train_exact.shape)
    y_test = y_test_exact + gamma_test * np.random.randn(*y_test_exact.shape)
    P_test_used = P_test if M is None else nc.noisy_test_probs(P_test, M, "mult", tol=nc.EIG_TOL)
    return y_train, y_test[0, 0], P_test_used


def main():
    print("=" * 72)
    print("One experiment, three noise contexts, parallelized (n_test=1, n_obs=1)")
    print("=" * 72)
    print(
        f"povm={POVM_TYPE}  d={DIM}  n_out={N_OUT}  n_train={N_TRAIN}  "
        f"N_REPEATS={N_REPEATS}  seed={SEED}  workers={N_WORKERS}"
    )

    nc.set_seed(SEED)
    M_mu = cf.build_povm(DIM, POVM_TYPE, N_OUT)
    scenario = cf.build_regression_scenario(M_mu, DIM, N_TRAIN, 1, 1)
    P_train, P_test = scenario["P_train"], scenario["P_test"]
    y_train_exact, y_test_exact = scenario["y_train"], scenario["y_test"]

    # -- sequential: one-time per-context label-noise / test-shot-noise draws --
    context_setups = [
        setup_context(P_train, P_test, y_train_exact, y_test_exact, ctx["gamma"], ctx["gamma_test"], ctx["M"])
        for ctx in CONTEXTS
    ]

    # -- parallel: the expensive per-(context, N) shot-noise sampling + pinv regression --
    tasks = []
    for i, (y_train, y_test_scalar, P_test_used) in enumerate(context_setups):
        for j, N in enumerate(N_VALUES):
            seed = None if SEED is None else (SEED, i, j)
            tasks.append((i, j, P_train, y_train, P_test_used, y_test_scalar, N, N_REPEATS, seed))

    sq_errors_list = [np.empty((N_REPEATS, len(N_VALUES))) for _ in CONTEXTS]

    print(f"\n=== dispatching {len(tasks)} (context, N) tasks across {N_WORKERS} worker processes ===")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_sq_errors_for_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            ctx_idx, n_idx, sq_errors = future.result()
            sq_errors_list[ctx_idx][:, n_idx] = sq_errors
            print(
                f"  [{done:>2d}/{len(tasks)}] {CONTEXTS[ctx_idx]['label']}  "
                f"N={N_VALUES[n_idx]:>6d}  MSE_empirical={sq_errors.mean():.4e}"
            )
    print(f"=== done in {time.time() - t0:.1f}s ===")

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
    )
    filename = cf3.results_filename(DIM, POVM_TYPE, N_OUT, N_TRAIN, M_FINITE, GAMMA_NONZERO)
    print("\n=== saving results ===")
    cf.save_results(cf3.RESULTS_DIR / filename, results)

    print("\nDone. Run plot_check_fit_3.py with the same parameters to plot these results.")


if __name__ == "__main__":
    main()
