"""Fig. 4 (top panel): rescaled training bias N^2 bias^2 vs n_tr.

For each input dimension d in `DIMS`, the MUB measurement is used to build
the training/test feature matrices, the training shot budget is held fixed
at `STAT_RHO`, and n_tr is swept over `TRAINING_LIST` (restricted, per
dimension, to n_tr >= `N_TR_MIN[d]`). The test side is evaluated with the
exact (infinite-statistics) probabilities.

For every repetition (an independent draw of training states, test states
and target observables) two quantities are computed:

- the empirical rescaled squared bias ``N^2 bias^2``, where ``bias`` is the
  mean over ``n_real`` shot-noise realizations of ``W_N p_sigma - y_sigma``,
  averaged (squared) over test states and observables. The squared sample
  mean is corrected by ``s^2 / n_real`` (``s^2`` the sample variance over
  realizations), which removes the O(V / (N n_tr n_real)) contamination of
  the training variance and makes the estimator unbiased for bias^2. Since
  that contamination is ~ V N / (b^2 n_tr n_real) relative to the signal,
  ``n_real = max(N_REAL_MIN, REAL_BUDGET / n_tr)`` is scaled up at small
  n_tr (where each realization is cheap) to keep the correction small;

- the asymptotic coefficient ``b^2`` of eq. (46) of the paper,
  ``b = w_eff^T Sigma_bar S^+ p_sigma``, with ``S = P P^T / n_tr``,
  ``Sigma_bar = n_tr^{-1} sum_i [diag(p_i) - p_i p_i^T]``, and
  ``w_eff = (I - U2 Sigma_bar_22^{-1} Sigma_bar_21 U1^T) w_O``, where U1
  (U2) spans the range (kernel) of P P^T and ``w_O^T = y P^+``.

Output: two files per dimension, written to ``data/``, each containing
the swept n_tr values and the [p10, p50, p90, mean] quantiles over
`N_REPS` repetitions: ``N2bias2_vs_ntrain_MUB_d=....txt`` and
``b2_vs_ntrain_MUB_d=....txt``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import chop, dirt_multinomial, flattened_states, mub, quantiles, save_mse_data, sigma_batch  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "data"

DIMS = (2, 3, 5, 7)
N_REPS = 100
N_REAL_MIN = 100
REAL_BUDGET = 100_000
N_TEST = 100
N_OBS = 50
STAT_RHO = 10000

TRAINING_LIST = np.round(2.0 ** np.arange(3.0, 11.5 + 1e-9, 0.5)).astype(int)
N_TR_MIN = {2: 8, 3: 22, 5: 45, 7: 90}

EIG_TOL = 1e-10


def b_coefficient(y_rho, P_rho, P_sigma):
    """Asymptotic bias coefficient b of eq. (46), for every
    (observable, test state) pair: returns an ``(n_obs, n_test)`` array.
    """
    n_train = P_rho.shape[1]
    S = P_rho @ P_rho.T / n_train
    Sigma_bar = sigma_batch(P_rho).mean(axis=0)

    lam, U = np.linalg.eigh(S)
    keep = lam > EIG_TOL * lam.max()
    U1, U2 = U[:, keep], U[:, ~keep]
    S_pinv = U1 @ np.diag(1.0 / lam[keep]) @ U1.T

    W = y_rho @ np.linalg.pinv(P_rho)  # rows are w_O^T
    if U2.shape[1] > 0:
        Sigma_12 = U1.T @ Sigma_bar @ U2
        Sigma_22 = U2.T @ Sigma_bar @ U2
        W = W - W @ U1 @ Sigma_12 @ np.linalg.solve(Sigma_22, U2.T)
    return W @ Sigma_bar @ S_pinv @ P_sigma


def bias_single_repetition(M_mu, dim_in, n_train, n_test, n_obs, N, n_real):
    """(N^2 bias^2, b^2) for one draw of training/test/observable ensembles."""
    M_rho = flattened_states(dim_in, n_train)
    M_sigma = flattened_states(dim_in, n_test)
    M_obs = flattened_states(dim_in, n_obs)

    y_rho = np.real(chop(M_obs.conj() @ M_rho.T))
    y_sigma = np.real(chop(M_obs.conj() @ M_sigma.T))
    P_rho = np.real(chop(M_mu.conj() @ M_rho.T))
    P_sigma = np.real(chop(M_mu.conj() @ M_sigma.T))

    P_rho = P_rho / P_rho.sum(axis=0, keepdims=True)
    P_sigma = P_sigma / P_sigma.sum(axis=0, keepdims=True)

    pred_sum = np.zeros_like(y_sigma)
    pred_sq_sum = np.zeros_like(y_sigma)
    for _ in range(n_real):
        P_rho_N = dirt_multinomial(P_rho.T, N).T
        W_N = y_rho @ np.linalg.pinv(P_rho_N)
        pred = W_N @ P_sigma
        pred_sum += pred
        pred_sq_sum += pred**2

    pred_mean = pred_sum / n_real
    pred_var = (pred_sq_sum - n_real * pred_mean**2) / (n_real - 1)
    bias2 = np.mean((pred_mean - y_sigma) ** 2 - pred_var / n_real)

    b = b_coefficient(y_rho, P_rho, P_sigma)
    return N**2 * bias2, np.mean(b**2)


def main():
    for dim_in in DIMS:
        M_mu = np.array([p.flatten() for p in mub(dim_in)])
        dim_out = dim_in * (dim_in + 1)
        train_list = TRAINING_LIST[TRAINING_LIST >= N_TR_MIN[dim_in]]

        res_bias = np.empty((len(train_list), N_REPS))
        res_b2 = np.empty((len(train_list), N_REPS))
        for i, n_train in enumerate(train_list):
            n_real = max(N_REAL_MIN, int(np.ceil(REAL_BUDGET / n_train)))
            for rep in range(N_REPS):
                res_bias[i, rep], res_b2[i, rep] = bias_single_repetition(
                    M_mu, dim_in, int(n_train), N_TEST, N_OBS, STAT_RHO, n_real
                )
            print(f"d={dim_in}  ntrain={n_train}  nreal={n_real}  done")

        suffix = (
            f"_MUB_d={dim_in}_nout={dim_out}_N={STAT_RHO}"
            f"_ntrain={train_list[-1]}_ntest={N_TEST}_nobs={N_OBS}.txt"
        )
        for stem, res in (("N2bias2_vs_ntrain", res_bias), ("b2_vs_ntrain", res_b2)):
            res_q = np.array([quantiles(res[i]) for i in range(len(train_list))])
            path = OUTPUT_DIR / (stem + suffix)
            save_mse_data(path, train_list, res_q)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
