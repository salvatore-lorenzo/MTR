"""Figure 6: MSE vs training shot number N, with additive target (label) noise.

Python translation of ``Mathematica code/label_noise/label_noise.nb``.

Both the training and test targets are corrupted by i.i.d. Gaussian
noise of standard deviation `gamma` (``y -> y + gamma * N(0,1)``), on top
of the usual finite-shot estimation of the training probabilities
(N shots, swept over `STAT_LIST`) and test probabilities (fixed M shots).
The sweep is repeated for a few values of `gamma`.

Output: one ``MSE_vs_N_MUB_d=..._.txt`` file per gamma value, written to
``./label_noise/`` next to this script.
"""

from pathlib import Path

import numpy as np

from common import chop, dirt_multinomial, mub, quantiles, random_kets, save_mse_data

OUTPUT_DIR = Path(__file__).resolve().parent / "label_noise"

DIM_IN = 2
N_REPS = 50
N_REAL = 50
N_TRAIN = 100
N_TEST = 100
N_OBS = 50
STAT_SIGMA = 100 # for infinite-shot test probabilities use "inf" 
GAMMA_VALUES = (0, 0.1, 0.2)

STAT_LIST = np.round(10.0 ** np.arange(0.5, 6.0 + 1e-9, 0.5)).astype(int)


def flattened_states(dim, n):
    """`n` random pure-state density matrices, flattened row-major, as rows."""
    kets = random_kets(dim, n)
    return np.array([np.outer(k, k.conj()).flatten() for k in kets])


def mse_single_repetition(
    M_mu, dim_in, n_train, n_test, n_obs, stat_rho, stat_sigma, gamma, gamma_test, n_real
):
    M_rho = flattened_states(dim_in, n_train)
    M_sigma = flattened_states(dim_in, n_test)
    M_obs = flattened_states(dim_in, n_obs)

    y_rho = chop(M_obs.conj() @ M_rho.T)
    y_rho_dirt = y_rho + gamma * np.random.randn(n_obs, n_train)

    y_sigma = chop(M_obs.conj() @ M_sigma.T)
    if gamma_test==0:
        y_sigma_dirt = y_sigma
    else:
        y_sigma_dirt = y_sigma + gamma_test * np.random.randn(n_obs, n_test)

    P_rho = chop(M_mu.conj() @ M_rho.T)
    P_sigma = chop(M_mu.conj() @ M_sigma.T)

    P_rho = P_rho / P_rho.sum(axis=0, keepdims=True)
    P_sigma = P_sigma / P_sigma.sum(axis=0, keepdims=True)
    if stat_sigma is "inf":
        P_sigma_M = P_sigma
    else:
        P_sigma_M = dirt_multinomial(P_sigma.T, stat_sigma).T

    sq_errors = []
    for _ in range(n_real):
        P_rho_N = dirt_multinomial(P_rho.T, stat_rho).T
        W_N = y_rho_dirt @ np.linalg.pinv(P_rho_N)
        diff = W_N @ P_sigma_M - y_sigma_dirt
        sq_errors.append(np.abs(diff) ** 2)
    return np.mean(np.concatenate([e.flatten() for e in sq_errors]))


def main():
    M_mu = np.array([p.flatten() for p in mub(DIM_IN)])
    dim_out = DIM_IN * (DIM_IN + 1)

    for gamma in GAMMA_VALUES:
        res = np.empty((len(STAT_LIST), N_REPS))
        for i, stat_rho in enumerate(STAT_LIST):
            for rep in range(N_REPS):
                res[i, rep] = mse_single_repetition(
                    M_mu, DIM_IN, N_TRAIN, N_TEST, N_OBS, stat_rho, STAT_SIGMA, gamma, 0, N_REAL
                )
            print(f"gamma={gamma}  N={stat_rho}  done")

        res_mse = np.array([quantiles(res[i]) for i in range(len(STAT_LIST))])

        filename = (
            f"MSE_vs_N_MUB_d={DIM_IN}_gamma_train={gamma}_nout={dim_out}_M={STAT_SIGMA}"
            f"_ntrain={N_TRAIN}_ntest={N_TEST}_nobs={N_OBS}_.txt"
        )
        save_mse_data(OUTPUT_DIR / filename, STAT_LIST, res_mse)
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
