"""Fig. 5: MSE vs training shot number N, with training label noise.

Python translation of ``mathematica/label_noise.nb``.

The training targets are corrupted by i.i.d. Gaussian noise of standard
deviation `gamma` (``y -> y + gamma * N(0,1)``), on top of the finite-shot
estimation of the training probabilities (N shots, swept over
`STAT_LIST`) and test probabilities (fixed `STAT_SIGMA` = M shots). Test
targets are noiseless (``gamma_test = 0``). The sweep is repeated for each
value in `GAMMA_VALUES`.

Output: one ``MSE_vs_N_MUB_d=..._gamma_train=....txt`` file per gamma
value, written to ``data/``, each containing the swept N values and the
[p10, p50, p90, mean] quantiles of the per-realization MSEs, pooled over
``N_REPS`` repetitions x ``N_REAL`` shot-noise realizations.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import chop, dirt_multinomial, flattened_states, mub, quantiles, save_mse_data  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "data"

DIM_IN = 2
N_REPS = 50
N_REAL = 50
N_TRAIN = 100
N_TEST = 100
N_OBS = 50
STAT_SIGMA = 100  # test shots M; use "inf" for exact test probabilities
GAMMA_VALUES = (0, 0.1, 0.2)
GAMMA_TEST = 0

STAT_LIST = np.round(10.0 ** np.arange(0.5, 6.0 + 1e-9, 0.5)).astype(int)


def mse_single_repetition(
    M_mu, dim_in, n_train, n_test, n_obs, stat_rho, stat_sigma, gamma, gamma_test, n_real
):
    """MSE of each of `n_real` shot-noise realizations, for one draw of
    the training/test/observable ensembles and of the label noise."""
    M_rho = flattened_states(dim_in, n_train)
    M_sigma = flattened_states(dim_in, n_test)
    M_obs = flattened_states(dim_in, n_obs)

    y_rho = chop(M_obs.conj() @ M_rho.T)
    y_rho_dirt = y_rho + gamma * np.random.randn(n_obs, n_train)

    y_sigma = chop(M_obs.conj() @ M_sigma.T)
    if gamma_test == 0:
        y_sigma_dirt = y_sigma
    else:
        y_sigma_dirt = y_sigma + gamma_test * np.random.randn(n_obs, n_test)

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
        P_rho_N = dirt_multinomial(P_rho.T, stat_rho).T
        W_N = y_rho_dirt @ np.linalg.pinv(P_rho_N)
        diff = W_N @ P_sigma_M - y_sigma_dirt
        sq_errors.append(np.mean(np.abs(diff) ** 2))
    return np.array(sq_errors)


def main():
    M_mu = np.array([p.flatten() for p in mub(DIM_IN)])
    dim_out = DIM_IN * (DIM_IN + 1)

    for gamma in GAMMA_VALUES:
        res = np.empty((len(STAT_LIST), N_REPS, N_REAL))
        for i, stat_rho in enumerate(STAT_LIST):
            for rep in range(N_REPS):
                res[i, rep] = mse_single_repetition(
                    M_mu, DIM_IN, N_TRAIN, N_TEST, N_OBS, stat_rho, STAT_SIGMA, gamma, GAMMA_TEST, N_REAL
                )
            print(f"gamma={gamma}  N={stat_rho}  done")

        res_mse = np.array([quantiles(res[i].ravel()) for i in range(len(STAT_LIST))])

        filename = (
            f"MSE_vs_N_MUB_d={DIM_IN}_gamma_train={gamma}_nout={dim_out}_M={STAT_SIGMA}"
            f"_ntrain={N_TRAIN}_ntest={N_TEST}_nobs={N_OBS}.txt"
        )
        save_mse_data(OUTPUT_DIR / filename, STAT_LIST, res_mse)
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
