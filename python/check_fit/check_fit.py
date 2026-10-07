"""Fig. SM4: single-instance check of the four-coefficient MSE fit -- run
and save.

One regression instance is fixed by `SEED`: a Haar-random rank-one POVM
with `N_OUT` outcomes, `N_TRAIN` Haar training states, a single Haar test
state sigma and a single target observable O. For each test-shot budget
M in `M_VALUES` and each pair of label-noise levels (gamma on the training
targets, gamma_test on the test target) in `GAMMA_PAIRS`, the training
shot budget N is swept over `N_VALUES`.

For every N, `N_REAL` independent realizations of ALL the noise sources
are drawn jointly -- training shot noise (Multinomial(N, p_i) / N per
training state), test shot noise (Multinomial(M, q) / M), training label
noise (y -> y + gamma xi) and test label noise (o -> o + gamma_test xi')
-- and every individual squared error

    ( y_hat P_hat^+ q_hat  -  o_hat )^2

is kept. Their mean over realizations is the empirical MSE(N) of this
fixed instance, to be compared with the fit

    f(N) = C0 + C1 / N + C2 / (N + d(d+2))^2 + C3 N,

    C0 = gamma_test^2 + w_O Sigma_sigma w_O^T / M + gamma^2 (1 + R) ||P^+ q||^2
    C1 = V / n_tr
    C2 = b^2
    C3 = gamma^2 R / M,           R = (n_out - d^2) / (n_tr - n_out - 1)

where P is the exact (n_out, n_tr) training probability matrix, q the
exact test probability vector, Sigma_sigma = diag(q) - q q^T, w_O^T =
y P^+ (y the exact training targets), and V, b^2 are the single-instance
asymptotic variance/bias coefficients of eq. (46) of the paper
(``bias_variance/variance.py:V_coefficient``,
``bias_variance/bias.py:b_coefficient``). With M = None (exact test
statistics) the M-dependent pieces are dropped.

The raw squared errors and the fit coefficients are saved, one pickle per
(M, gamma, gamma_test), under ``data/`` (see `results_filename`). Run
``plot_check_fit.py`` to make the figure.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bias_variance"))
import common  # noqa: E402
from bias import b_coefficient  # noqa: E402
from variance import V_coefficient  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "data"

# ----------------------------------------------------------------------
# Parameters.
# ----------------------------------------------------------------------

SEED = 1  # fixes the instance (POVM, training states, sigma, O) and the noise draws

DIM = 2
N_OUT = 16  # outcomes of the random POVM
N_TRAIN = 256  # must be > N_OUT + 1 for R to be defined
M_VALUES = (10000, None)  # test-shot budgets (None = exact test statistics)
GAMMA_PAIRS = ((0.0, 0.0), (0.1, 0.0), (0.1, 0.1))  # (gamma, gamma_test)

N_VALUES = sorted(set(np.round(np.logspace(np.log10(5), 6, 16)).astype(int).tolist()))
N_REAL = 100  # joint noise realizations per N
CHUNK_ELEMS = 2**24  # max entries of a (chunk, n_out, n_tr) stack held in memory at once
PINV_RCOND = 1e-12


def results_filename(dim, n_out, n_train, M, gamma, gamma_test):
    M_str = "inf" if M is None else str(M)
    return (
        f"check_fit_d={dim}_nout={n_out}_ntrain={n_train}"
        f"_M={M_str}_gamma={gamma}_gammatest={gamma_test}.pkl"
    )


def save_results(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  wrote {path}")


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_instance(dim, n_out, n_train):
    """The fixed instance: exact training probabilities P (n_out, n_tr),
    test probabilities q (n_out, 1), exact training targets y (1, n_tr)
    and exact test target o (1, 1).
    """
    M_mu = common.random_povm(dim, n_out)
    M_rho = common.flattened_states(dim, n_train)
    M_sigma = common.flattened_states(dim, 1)
    M_obs = common.flattened_states(dim, 1)

    P = common.povm_probabilities(M_mu, M_rho)
    q = common.povm_probabilities(M_mu, M_sigma)
    P = P / P.sum(axis=0, keepdims=True)
    q = q / q.sum(axis=0, keepdims=True)

    y = (M_obs.conj() @ M_rho.T).real
    o = (M_obs.conj() @ M_sigma.T).real
    return P, q, y, o


def fit_coefficients(P, q, y, dim, gamma, gamma_test, M):
    """C0..C3 of the fit, plus the instance quantities they are built from."""
    n_out, n_tr = P.shape
    P_pinv = np.linalg.pinv(P)
    w = y @ P_pinv  # (1, n_out), w_O^T = y P^+
    Sigma_sigma = common.sigma_from_p(q[:, 0])

    R = (n_out - dim**2) / (n_tr - n_out - 1)
    wSw = (w @ Sigma_sigma @ w.T).item()
    Pq2 = float(np.sum((P_pinv @ q) ** 2))
    V = float(V_coefficient(y, P, q)[0, 0])
    b2 = float(b_coefficient(y, P, q)[0, 0] ** 2)

    inv_M = 0.0 if M is None else 1.0 / M
    return dict(
        C0=gamma_test**2 + wSw * inv_M + gamma**2 * (1 + R) * Pq2,
        C1=V / n_tr,
        C2=b2,
        C3=gamma**2 * R * inv_M,
        R=R, wSw=wSw, Pq2=Pq2, V=V, b2=b2,
    )


def mse_fit(N, coeffs, dim):
    """f(N) = C0 + C1/N + C2/(N + d(d+2))^2 + C3 N."""
    N = np.asarray(N, dtype=float)
    return coeffs["C0"] + coeffs["C1"] / N + coeffs["C2"] / (N + dim * (dim + 2)) ** 2 + coeffs["C3"] * N


def squared_errors(P, q, y, o, N, n_real, gamma, gamma_test, M):
    """`n_real` squared errors at training shot budget N, each with its own
    independent draw of training shots, test shots and label noise.
    """
    n_out, n_tr = P.shape
    chunk = max(1, CHUNK_ELEMS // (n_out * n_tr))
    out = []
    for start in range(0, n_real, chunk):
        r = min(chunk, n_real - start)
        P_hat = common.multinomial_phat_batch(P, N, r)  # (r, n_out, n_tr)
        q_hat = q[None, :, 0] if M is None else common.multinomial_phat_batch(q, M, r)[:, :, 0]  # (r, n_out)
        y_noisy = y[0][None, :] + gamma * common.rng.standard_normal((r, n_tr))
        o_noisy = o[0, 0] + gamma_test * common.rng.standard_normal(r)

        a = np.einsum("rik,rk->ri", np.linalg.pinv(P_hat, rcond=PINV_RCOND), np.broadcast_to(q_hat, (r, n_out)))
        o_hat = np.einsum("ri,ri->r", y_noisy, a)
        out.append((o_hat - o_noisy) ** 2)
    return np.concatenate(out)


def run(M, gamma, gamma_test):
    print(f"\n=== d={DIM}  n_out={N_OUT}  n_train={N_TRAIN}  M={M}  gamma={gamma}  gamma_test={gamma_test} ===")
    common.set_seed(SEED)
    P, q, y, o = build_instance(DIM, N_OUT, N_TRAIN)
    coeffs = fit_coefficients(P, q, y, DIM, gamma, gamma_test, M)
    print("fit coefficients: " + "  ".join(f"{k}={v:.4e}" for k, v in coeffs.items()))

    sq_errors = np.empty((N_REAL, len(N_VALUES)))
    for j, N in enumerate(N_VALUES):
        sq_errors[:, j] = squared_errors(P, q, y, o, N, N_REAL, gamma, gamma_test, M)
        mse = sq_errors[:, j].mean()
        sem = sq_errors[:, j].std(ddof=1) / np.sqrt(N_REAL)
        fit = mse_fit(N, coeffs, DIM)
        print(f"  N={N:>7d}  MSE={mse:.4e} +- {sem:.1e}  fit={fit:.4e}  ratio={mse / fit:.3f}")

    results = dict(
        N_values=np.asarray(N_VALUES),
        sq_errors=sq_errors,
        coeffs=coeffs,
        dim=DIM, n_out=N_OUT, n_train=N_TRAIN, M=M, gamma=gamma, gamma_test=gamma_test,
        n_real=N_REAL, seed=SEED,
    )
    save_results(RESULTS_DIR / results_filename(DIM, N_OUT, N_TRAIN, M, gamma, gamma_test), results)


def main():
    if N_TRAIN <= N_OUT + 1:
        raise ValueError("N_TRAIN must be > N_OUT + 1 for R to be defined")
    for gamma, gamma_test in GAMMA_PAIRS:
        for M in M_VALUES:
            run(M, gamma, gamma_test)


if __name__ == "__main__":
    main()
