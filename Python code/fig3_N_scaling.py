"""Figure 3: MSE vs training shot number N, in the M -> infinity limit.

Python translation of ``Mathematica code/M_inf/N_scaling.nb``.

For a few training-set sizes n_tr, the MUB measurement is used to build
the training/test feature matrices, and the training-side shot noise
(N measurements per training state) is swept over `stat_list`. The test
side is evaluated with the exact (infinite-statistics) probabilities, so
the resulting MSE isolates the effect of finite training statistics.

Output: one ``MSE_vs_N_MUB_d=..._.txt`` file per n_tr value, written to
``./M_inf/`` next to this script, each containing the swept N values and
the [p10, p50, p90, mean] MSE quantiles over ``n_reps`` repetitions.
"""

from pathlib import Path

import numpy as np

from common import chop, dirt_multinomial, mub, quantiles, random_kets, save_mse_data

OUTPUT_DIR = Path(__file__).resolve().parent / "M_inf"

DIM_IN = 2
N_REPS = 50
N_REAL = 100
N_TEST = 100
N_OBS = 50
N_TRAIN_VALUES = (100, 1000, 10000)

STAT_LIST = np.round(10.0 ** np.arange(0.5, 6.0 + 1e-9, 0.5)).astype(int)


def flattened_states(dim, n):
    """`n` random pure-state density matrices, flattened row-major, as rows."""
    kets = random_kets(dim, n)
    return np.array([np.outer(k, k.conj()).flatten() for k in kets])


def mse_single_repetition(M_mu, dim_in, n_train, n_test, n_obs, N, n_real):
    M_rho = flattened_states(dim_in, n_train)
    M_sigma = flattened_states(dim_in, n_test)
    M_obs = flattened_states(dim_in, n_obs)

    y_rho = chop(M_obs.conj() @ M_rho.T)
    y_sigma = chop(M_obs.conj() @ M_sigma.T)
    P_rho = chop(M_mu.conj() @ M_rho.T)
    P_sigma = chop(M_mu.conj() @ M_sigma.T)

    P_rho = P_rho / P_rho.sum(axis=0, keepdims=True)
    P_sigma = P_sigma / P_sigma.sum(axis=0, keepdims=True)

    sq_errors = []
    for _ in range(n_real):
        P_rho_N = dirt_multinomial(P_rho.T, N).T
        W_N = y_rho @ np.linalg.pinv(P_rho_N)
        diff = W_N @ P_sigma - y_sigma
        sq_errors.append(np.abs(diff) ** 2)
    return np.mean(np.concatenate([e.flatten() for e in sq_errors]))


def main():
    M_mu = np.array([p.flatten() for p in mub(DIM_IN)])
    dim_out = DIM_IN * (DIM_IN + 1)

    for n_train in N_TRAIN_VALUES:
        res = np.empty((len(STAT_LIST), N_REPS))
        for i, N in enumerate(STAT_LIST):
            for rep in range(N_REPS):
                res[i, rep] = mse_single_repetition(
                    M_mu, DIM_IN, n_train, N_TEST, N_OBS, N, N_REAL
                )
            print(f"ntrain={n_train}  N={N}  done")

        res_mse = np.array([quantiles(res[i]) for i in range(len(STAT_LIST))])

        filename = (
            f"MSE_vs_N_MUB_d={DIM_IN}_nout={dim_out}_M=inf"
            f"_ntrain={n_train}_ntest={N_TEST}_nobs={N_OBS}_.txt"
        )
        save_mse_data(OUTPUT_DIR / filename, STAT_LIST, res_mse)
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
