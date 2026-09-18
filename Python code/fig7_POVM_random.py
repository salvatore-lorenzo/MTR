"""Figure 7: MSE vs training shot number N, for random rank-one POVMs.

Python translation of ``Mathematica code/POVM_random/POVM_random.nb``.

Unlike the MUB-based figures, the measurement is a fresh Haar-random
rank-one POVM on `dim_out` outcomes (see ``random_povm`` in ``common.py``),
redrawn independently for every repetition, along with the training/test/
observable ensembles. The number of POVM outcomes is swept over
`DIM_OUT_VALUES`, and for each value the training-side shot noise (N
measurements per training state) is swept over `STAT_LIST`. The test side
is evaluated with the exact (infinite-statistics) probabilities, so the
resulting MSE isolates the effect of finite training statistics.

Output: one ``MSE_vs_N_POVM_d=..._.txt`` file per `dim_out` value, written
to ``./POVM_random/`` next to this script, each containing the swept N
values and the [p10, p50, p90, mean] MSE quantiles over ``n_reps``
repetitions.
"""

from pathlib import Path

import numpy as np

from common import chop, dirt_multinomial, quantiles, random_kets, random_povm, save_mse_data

OUTPUT_DIR = Path(__file__).resolve().parent / "POVM_random"

DIM_IN = 2
N_REPS = 100
N_REAL = 10
N_TRAIN = 256
N_TEST = 200
N_OBS = 100
DIM_OUT_VALUES = (168,208,256,296,336)#(4, 8, 16, 32, 64, 128)

STAT_LIST = np.round(10.0 ** np.arange(0.5, 5.0 + 1e-9, 0.25)).astype(int)


def flattened_states(dim, n):
    """`n` random pure-state density matrices, flattened row-major, as rows."""
    kets = random_kets(dim, n)
    return np.array([np.outer(k, k.conj()).flatten() for k in kets])


def mse_single_repetition(dim_in, dim_out, n_train, n_test, n_obs, N, n_real):
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

    sq_errors = []
    for _ in range(n_real):
        P_rho_N = dirt_multinomial(P_rho.T, N).T
        W_N = y_rho @ np.linalg.pinv(P_rho_N)
        diff = W_N @ P_sigma - y_sigma
        sq_errors.append(np.abs(diff) ** 2)
    return np.mean(np.concatenate([e.flatten() for e in sq_errors]))


def main():
    for dim_out in DIM_OUT_VALUES:
        res = np.empty((len(STAT_LIST), N_REPS))
        for i, N in enumerate(STAT_LIST):
            for rep in range(N_REPS):
                res[i, rep] = mse_single_repetition(
                    DIM_IN, dim_out, N_TRAIN, N_TEST, N_OBS, N, N_REAL
                )
            print(f"nout={dim_out}  N={N}  done")

        res_mse = np.array([quantiles(res[i]) for i in range(len(STAT_LIST))])

        filename = (
            f"MSE_vs_N_POVM_d={DIM_IN}_nout={dim_out}_M=inf"
            f"_ntrain={N_TRAIN}_ntest={N_TEST}_nobs={N_OBS}_.txt"
        )
        save_mse_data(OUTPUT_DIR / filename, STAT_LIST, res_mse)
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
