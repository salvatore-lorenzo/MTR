"""Single-instance check of the four-coefficient MSE fit -- run and save.

One regression instance is fixed ONCE: a Haar-random rank-one POVM with
`N_OUT` outcomes, `N_TRAIN` Haar training states, a single Haar test
state sigma and a single target observable O. The test-shot budget `M`
and the label-noise levels `GAMMA` (training targets) and `GAMMA_TEST`
(test target) are fixed too. Only the training shot budget N is swept.

For every N, `N_REAL` independent realizations of ALL the noise sources
are drawn jointly -- training shot noise (Multinomial(N, p_i) / N per
training state), test shot noise (Multinomial(M, q) / M), training label
noise (y -> y + gamma xi) and test label noise (o -> o + gamma_test xi')
-- and every individual squared error

    ( y_hat P_hat^+ q_hat  -  o_hat )^2

is kept. Their mean over realizations is the empirical MSE(N) of this
fixed instance, to be compared with the fit

    f(N) = C0 + C1 / N + C2 / N^2 + C3 N,

    C0 = gamma_test^2 + w_O Sigma_sigma w_O^T / M + gamma^2 (1 + R) ||P^+ q||^2
    C1 = V / n_tr
    C2 = b^2
    C3 = gamma^2 R / M,           R = (n_out - d^2) / (n_tr - n_out - 1)

where P is the exact (n_out, n_tr) training probability matrix, q the
exact test probability vector, Sigma_sigma = diag(q) - q q^T, w_O^T =
y P^+ (y the exact training targets), and V, b^2 are the single-instance
asymptotic variance/bias coefficients of eq. (SM119)
(`fig2_variance.V_coefficient`, `fig2_bias.b_coefficient`). With M=None
(exact test statistics) the M-dependent pieces are dropped.

The raw squared errors and every fit coefficient are saved to a pickle
under ``results/`` (see `results_filename`). Run ``plot_check_fit.py``
with the same parameters to make the figure.

Run with ``python3 check_fit.py`` from anywhere; it locates its output
directory next to itself.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, fig2_*.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import common  # noqa: E402
import noise_comparison as nc  # noqa: E402
from fig2_bias import b_coefficient  # noqa: E402
from fig2_variance import V_coefficient  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

# ----------------------------------------------------------------------
# Parameters (all easy to change).
# ----------------------------------------------------------------------

SEED = 15  # fixes the instance (POVM, training states, sigma, O) and the noise draws

DIM = 4
N_OUT = 64  # outcomes of the random POVM
N_TRAIN = 256  # must be > N_OUT + 1 for R to be defined
M = 1e7  # test-shot budget (None = exact test statistics)
GAMMA = 0.01  # label noise std on the training targets
GAMMA_TEST = 0.0  # label noise std on the test target

N_VALUES = sorted(set(np.round(np.logspace(np.log10(5), 8, 16)).astype(int).tolist()))
N_REAL = 100  # joint noise realizations per N
CHUNK_ELEMS = 2**24  # max entries of a (chunk, n_out, n_tr) stack held in memory at once
PINV_RCOND = 1e-12


def results_filename(dim, n_out, n_train, M, gamma, gamma_test):
    M_str = "inf" if M is None else str(M)
    return (
        f"check_fit_2_d={dim}_nout={n_out}_ntrain={n_train}"
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
    M_rho = nc.flattened_states(dim, n_train)
    M_sigma = nc.flattened_states(dim, 1)
    M_obs = nc.flattened_states(dim, 1)

    P = nc.povm_probabilities(M_mu, M_rho)
    q = nc.povm_probabilities(M_mu, M_sigma)
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


def fit_terms(N, coeffs):
    """The four terms of f(N), as a dict {name: array}; f(N) is their sum."""
    N = np.asarray(N, dtype=float)
    return {
        "C0": np.full_like(N, coeffs["C0"]),
        "C1/N": coeffs["C1"] / N,
        "C2/N^2": coeffs["C2"] / (N+8)**2,
        "C3 N": coeffs["C3"] * N,
    }


def mse_fit(N, coeffs):
    return sum(fit_terms(N, coeffs).values())


def averaged_fit(N, d, n_tr, n_out, gamma, gamma_test, M):
    """Closed-form MSE averaged over sigma and O (depends only on d, n_tr,
    n_out, gamma, gamma_test, M -- not on the drawn instance):

        (d-1)(d+2) / (d(d+1) n_tr N) + (d-1)(d+2)^2 / ((d+1)(d(d+2) + N)^2)
        + (d-1)(d+2) / (d(d+1) M) + gamma_test^2 + d^2 gamma^2 / n_tr
        + (n_out - d^2) N gamma^2 / ((n_tr - n_out - 1) M)
    """
    N = np.asarray(N, dtype=float)
    Cd = (d - 1) * (d + 2) / (d * (d + 1))
    inv_M = 0.0 if M is None else 1.0 / M
    return (
        Cd / (n_tr * N)
        + (d - 1) * (d + 2) ** 2 / ((d + 1) * (d * (d + 2) + N) ** 2)
        + Cd * inv_M
        + gamma_test**2
        + d**2 * gamma**2 / n_tr
        + (n_out - d**2) * N * gamma**2 * inv_M / (n_tr - n_out - 1)
    )


def squared_errors(P, q, y, o, N, n_real, gamma, gamma_test, M):
    """`n_real` squared errors at training shot budget N, each with its own
    independent draw of training shots, test shots and label noise.
    """
    n_out, n_tr = P.shape
    chunk = max(1, CHUNK_ELEMS // (n_out * n_tr))
    out = []
    for start in range(0, n_real, chunk):
        r = min(chunk, n_real - start)
        P_hat = nc.multinomial_phat_batch(P, N, r)  # (r, n_out, n_tr)
        q_hat = q[None, :, 0] if M is None else nc.multinomial_phat_batch(q, M, r)[:, :, 0]  # (r, n_out)
        y_noisy = y[0][None, :] + gamma * common.rng.standard_normal((r, n_tr))
        o_noisy = o[0, 0] + gamma_test * common.rng.standard_normal(r)

        a = np.einsum("rik,rk->ri", np.linalg.pinv(P_hat, rcond=PINV_RCOND), np.broadcast_to(q_hat, (r, n_out)))
        o_hat = np.einsum("ri,ri->r", y_noisy, a)
        out.append((o_hat - o_noisy) ** 2)
    return np.concatenate(out)


def main():
    print("=" * 72)
    print("Single-instance MSE vs N -- check of f(N) = C0 + C1/N + C2/N^2 + C3 N")
    print("=" * 72)
    print(
        f"d={DIM}  n_out={N_OUT}  n_train={N_TRAIN}  M={M}  gamma={GAMMA}  "
        f"gamma_test={GAMMA_TEST}  N_REAL={N_REAL}  seed={SEED}"
    )
    if N_TRAIN <= N_OUT + 1:
        raise ValueError("N_TRAIN must be > N_OUT + 1 for R to be defined")

    nc.set_seed(SEED)
    P, q, y, o = build_instance(DIM, N_OUT, N_TRAIN)
    coeffs = fit_coefficients(P, q, y, DIM, GAMMA, GAMMA_TEST, M)
    print("fit coefficients: " + "  ".join(f"{k}={v:.4e}" for k, v in coeffs.items()))

    sq_errors = np.empty((N_REAL, len(N_VALUES)))
    for j, N in enumerate(N_VALUES):
        sq_errors[:, j] = squared_errors(P, q, y, o, N, N_REAL, GAMMA, GAMMA_TEST, M)
        mse = sq_errors[:, j].mean()
        sem = sq_errors[:, j].std(ddof=1) / np.sqrt(N_REAL)
        fit = mse_fit(N, coeffs)
        print(f"  N={N:>7d}  MSE={mse:.4e} +- {sem:.1e}  fit={fit:.4e}  ratio={mse / fit:.3f}")

    results = dict(
        N_values=np.asarray(N_VALUES),
        sq_errors=sq_errors,
        coeffs=coeffs,
        dim=DIM, n_out=N_OUT, n_train=N_TRAIN, M=M, gamma=GAMMA, gamma_test=GAMMA_TEST,
        n_real=N_REAL, seed=SEED,
    )
    print("\n=== saving results ===")
    save_results(RESULTS_DIR / results_filename(DIM, N_OUT, N_TRAIN, M, GAMMA, GAMMA_TEST), results)
    print("\nDone. Run plot_check_fit.py with the same parameters to plot these results.")


if __name__ == "__main__":
    main()
