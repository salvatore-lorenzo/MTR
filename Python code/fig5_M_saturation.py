"""Figure 5: MSE vs test shot number M, at fixed training-set size.

Python translation of
``Mathematica code/M_saturation/M_saturation.nb``.

The training shot budget N is swept over `STAT_LIST`, and for each of a
few fixed test shot budgets M (`STAT_SIGMA_VALUES`) both the training
probabilities (with N shots) and the test probabilities (with M shots)
are estimated from finite statistics.

Output: one ``MSE_vs_M_MUB_d=..._.txt`` file per M value, written to
``./M_saturation/`` next to this script. As in the original notebook,
the fixed M value for each file is recorded in the filename under the
``N=`` field (kept for consistency with the exported Mathematica files).
"""

from pathlib import Path

import numpy as np

from common import chop, dirt_multinomial, mub, quantiles, random_kets, save_mse_data

OUTPUT_DIR = Path(__file__).resolve().parent / "M_saturation"

DIM_IN = 2
N_REPS = 50
N_REAL = 10
N_TRAIN = 1000
N_TEST = 100
N_OBS = 50
STAT_SIGMA_VALUES = (10**2, 10**3, 10**4, 10**5)

STAT_LIST = np.round(10.0 ** np.arange(0.3, 4.1 + 1e-9, 0.2)).astype(int)


def flattened_states(dim, n):
    """`n` random pure-state density matrices, flattened row-major, as rows."""
    kets = random_kets(dim, n)
    return np.array([np.outer(k, k.conj()).flatten() for k in kets])


def mse_single_repetition(M_mu, dim_in, n_train, n_test, n_obs, stat_rho, stat_sigma, n_real):
    M_rho = flattened_states(dim_in, n_train)
    M_sigma = flattened_states(dim_in, n_test)
    M_obs = flattened_states(dim_in, n_obs)

    y_rho = chop(M_obs.conj() @ M_rho.T)
    y_sigma = chop(M_obs.conj() @ M_sigma.T)
    P_rho = chop(M_mu.conj() @ M_rho.T)
    P_sigma = chop(M_mu.conj() @ M_sigma.T)

    P_rho = P_rho / P_rho.sum(axis=0, keepdims=True)
    P_sigma = P_sigma / P_sigma.sum(axis=0, keepdims=True)

    P_sigma_M = dirt_multinomial(P_sigma.T, stat_sigma).T

    sq_errors = []
    for _ in range(n_real):
        P_rho_N = dirt_multinomial(P_rho.T, stat_rho).T
        W_N = y_rho @ np.linalg.pinv(P_rho_N)
        diff = W_N @ P_sigma_M - y_sigma
        sq_errors.append(np.abs(diff) ** 2)
    return np.mean(np.concatenate([e.flatten() for e in sq_errors]))


def main():
    M_mu = np.array([p.flatten() for p in mub(DIM_IN)])
    dim_out = DIM_IN * (DIM_IN + 1)

    for stat_sigma in STAT_SIGMA_VALUES:
        res = np.empty((len(STAT_LIST), N_REPS))
        for i, stat_rho in enumerate(STAT_LIST):
            for rep in range(N_REPS):
                res[i, rep] = mse_single_repetition(
                    M_mu, DIM_IN, N_TRAIN, N_TEST, N_OBS, stat_rho, stat_sigma, N_REAL
                )
            print(f"M={stat_sigma}  N={stat_rho}  done")

        res_mse = np.array([quantiles(res[i]) for i in range(len(STAT_LIST))])

        filename = (
            f"MSE_vs_M_MUB_d={DIM_IN}_nout={dim_out}_N={stat_sigma}"
            f"_ntrain={N_TRAIN}_ntest={N_TEST}_nobs={N_OBS}_.txt"
        )
        save_mse_data(OUTPUT_DIR / filename, STAT_LIST, res_mse)
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
