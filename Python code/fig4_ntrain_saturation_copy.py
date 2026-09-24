"""Figure 4: MSE vs training-set size n_tr, at fixed training shot budget N.

Python translation of
``Mathematica code/ntrain_saturation/ntrain_saturation.nb``.

For each input dimension d in `DIMS`, n_tr is swept over a dedicated
geometric grid (see `TRAINING_LISTS`; each dimension is paired with its
own grid, following the original notebook, since larger d needs a larger
minimum n_tr to be well posed) while the training shot budget N is held
fixed at `STAT_RHO`.

Output: one ``MSE_vs_ntrain_MUB_d=..._copy.txt`` file per dimension,
written to ``./ntrain_saturation/`` next to this script, each containing
the swept n_tr values and the [p10, p50, p90, mean] quantiles of the
per-realization MSEs, pooled over ``N_REPS`` repetitions x ``N_REAL``
realizations.

Note: the largest n_tr values (up to 2**14 = 16384) combined with 50
repetitions make this the most expensive of the four scripts; reduce
`N_REPS` or `MM` for a quick test run.
"""

from pathlib import Path

import numpy as np

from common import chop, dirt_multinomial, mub, quantiles, random_kets, save_mse_data

OUTPUT_DIR = Path(__file__).resolve().parent / "ntrain_saturation"

DIMS = (2, 3, 5, 7)
MM = 14
N_REPS = 50
N_REAL = 100
N_TEST = 100
N_OBS = 50
STAT_RHO = 100

_k = np.arange(0, MM + 1)
TRAINING_LISTS = (
    np.round(2.0 ** _k).astype(int),
    np.round(2.0 ** (0.8 * _k + MM * 0.2)).astype(int),
    np.round(2.0 ** (0.7 * _k + MM * 0.3)).astype(int),
    np.round(2.0 ** (0.6 * _k + MM * 0.4)).astype(int),
)


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
        sq_errors.append(np.mean(np.abs(diff) ** 2))
    return np.array(sq_errors)


def main():
    for dim_in, train_list in zip(DIMS, TRAINING_LISTS):
        M_mu = np.array([p.flatten() for p in mub(dim_in)])
        dim_out = dim_in * (dim_in + 1)

        res = np.empty((len(train_list), N_REPS, N_REAL))
        for i, n_train in enumerate(train_list):
            for rep in range(N_REPS):
                res[i, rep] = mse_single_repetition(
                    M_mu, dim_in, int(n_train), N_TEST, N_OBS, STAT_RHO, N_REAL
                )
            print(f"d={dim_in}  ntrain={n_train}  done")

        res_mse = np.array([quantiles(np.concatenate(res[i])) for i in range(len(train_list))])

        # As in the original notebook, the filename records the largest
        # n_tr in this dimension's grid (the loop variable's final value).
        filename = (
            f"MSE_vs_ntrain_MUB_d={dim_in}_nout={dim_out}_N={STAT_RHO}"
            f"_ntrain={train_list[-1]}_ntest={N_TEST}_nobs={N_OBS}_copy.txt"
        )
        save_mse_data(OUTPUT_DIR / filename, train_list, res_mse)
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
