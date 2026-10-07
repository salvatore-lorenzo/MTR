"""Figs. SM2, SM3, SM5, SM6: MSE vs training shot number N, for random
rank-one POVMs.

Python translation of ``mathematica/POVM_random.nb``.

Unlike the MUB-based figures, the measurement is a fresh Haar-random
rank-one POVM with `dim_out` outcomes (see ``random_povm`` in
``common.py``), redrawn independently for every repetition, along with
the training/test/observable ensembles. The number of POVM outcomes is
swept over `DIM_OUT_VALUES`, and for each value the training shot budget
N is swept over `STAT_LIST`. The test-side probabilities are either exact
(M = "inf", Figs. SM2 and SM5) or estimated from M = 1000 shots per test
state (Figs. SM3 and SM6), see `STAT_SIGMA_VALUES`.

Output: one ``MSE_vs_N_POVM_d=..._nout=..._M=....txt`` file per (n_out,
M) pair, written to ``data/``, each containing the swept N values and
the [p10, p50, p90, mean] quantiles over ``N_REPS`` repetitions of the
MSE averaged over ``N_REAL`` shot-noise realizations.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import chop, dirt_multinomial, flattened_states, quantiles, random_povm, save_mse_data  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "data"

DIM_IN = 2
N_REPS = 100
N_REAL = 10
N_TRAIN = 256
N_TEST = 200
N_OBS = 100
DIM_OUT_VALUES = (4, 8, 16, 32, 64, 128, 168, 208, 256, 296, 336)
STAT_SIGMA_VALUES = ("inf", 1000)  # test shots M ("inf" = exact test probabilities)

STAT_LIST = np.round(10.0 ** np.arange(0.5, 5.0 + 1e-9, 0.25)).astype(int)


def mse_single_repetition(dim_in, dim_out, n_train, n_test, n_obs, N, stat_sigma, n_real):
    """MSE averaged over `n_real` shot-noise realizations, for one draw of
    the POVM and of the training/test/observable ensembles."""
    M_rho = flattened_states(dim_in, n_train)
    M_sigma = flattened_states(dim_in, n_test)
    M_obs = flattened_states(dim_in, n_obs)
    M_mu = random_povm(dim_in, dim_out)

    y_rho = chop(M_obs.conj() @ M_rho.T)
    y_sigma = chop(M_obs.conj() @ M_sigma.T)
    P_rho = chop(M_mu.conj() @ M_rho.T)
    P_sigma = chop(M_mu.conj() @ M_sigma.T)

    P_rho = P_rho / P_rho.sum(axis=0, keepdims=True)
    P_sigma = P_sigma / P_sigma.sum(axis=0, keepdims=True)
    if stat_sigma == "inf":
        P_sigma_M = P_sigma
    else:
        P_sigma_M = dirt_multinomial(P_sigma.T, stat_sigma).T

    sq_errors = []
    for _ in range(n_real):
        P_rho_N = dirt_multinomial(P_rho.T, N).T
        W_N = y_rho @ np.linalg.pinv(P_rho_N)
        diff = W_N @ P_sigma_M - y_sigma
        sq_errors.append(np.abs(diff) ** 2)
    return np.mean(sq_errors)


def main():
    for stat_sigma in STAT_SIGMA_VALUES:
        for dim_out in DIM_OUT_VALUES:
            res = np.empty((len(STAT_LIST), N_REPS))
            for i, N in enumerate(STAT_LIST):
                for rep in range(N_REPS):
                    res[i, rep] = mse_single_repetition(
                        DIM_IN, dim_out, N_TRAIN, N_TEST, N_OBS, N, stat_sigma, N_REAL
                    )
                print(f"M={stat_sigma}  nout={dim_out}  N={N}  done")

            res_mse = np.array([quantiles(res[i]) for i in range(len(STAT_LIST))])

            filename = (
                f"MSE_vs_N_POVM_d={DIM_IN}_nout={dim_out}_M={stat_sigma}"
                f"_ntrain={N_TRAIN}_ntest={N_TEST}_nobs={N_OBS}.txt"
            )
            save_mse_data(OUTPUT_DIR / filename, STAT_LIST, res_mse)
            print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
