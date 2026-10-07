"""Shared definitions for the "Mind the Rank" numerical experiments.

The core routines (``mub``, ``dirt_multinomial``, ``quantiles``, ``to_dm``,
``random_unitary``, ``random_povm``) are Python translations of the
definitions at the top of every notebook in ``mathematica/``
(``MUB``, ``dirtmulinomial``, ``quantiles``, ``Todm``, ``RandomUnitary``,
``RandomPOVM``). The remaining helpers implement the multinomial noise
model and its Gaussian approximation (used by ``noise_comparison/`` and
``check_fit/``) and the text format of the saved MSE data.

Conventions
-----------
- A quantum state / POVM element on a ``dim``-dimensional Hilbert space is
  a ``dim x dim`` complex ``ndarray``.
- "Flattened" density matrices (as used to build the feature matrices
  ``M_mu``, ``M_rho``, ``M_sigma``, ``M_obs``) are row-major
  (``ndarray.flatten()``), matching Mathematica's ``Flatten``.
- Feature/target matrices follow the paper's convention: rows are POVM
  outcomes/observables, columns are the states of the corresponding
  ensemble (training, test, or observable set), e.g. ``P_rho`` has shape
  ``(n_out, n_train)``.

Note on ``quantiles``: Mathematica's ``Quantile`` with default settings
does not use exactly the same interpolation rule as NumPy's default
``linear`` method. With the sample sizes used here, the two conventions
differ by at most a linear interpolation between two adjacent order
statistics, which is immaterial for the reported MSE curves.
"""

import re
from decimal import Decimal
from pathlib import Path

import numpy as np

# Generator used by the shot-noise models. Random states and POVMs use the
# legacy global state (``np.random``); `set_seed` seeds both.
rng = np.random.default_rng()


def set_seed(seed):
    """Seed both random number sources: the legacy global state (used by
    `random_kets` / `random_unitary`) and the Generator `rng` (used by the
    shot-noise models).
    """
    global rng
    np.random.seed(seed)
    rng = np.random.default_rng(seed)


def mub(dim):
    """Rank-one POVM elements of a complete set of mutually unbiased bases.

    Returns a list of ``dim * (dim + 1)`` Hermitian ``dim x dim`` arrays
    (the eigenprojectors of the shift operator X, the clock operator Z,
    and X.Z^k for k = 1, ..., dim - 1, each rescaled by 1/(dim + 1)) that
    sum to the identity.
    """
    omega = np.exp(2j * np.pi / dim)

    X = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        X[(k + 1) % dim, k] = 1.0

    Z = np.diag(omega ** np.arange(1, dim + 1))

    operators = [X, Z] + [X @ np.linalg.matrix_power(Z, k) for k in range(1, dim)]

    povm = []
    for op in operators:
        _, vecs = np.linalg.eig(op)
        for j in range(dim):
            v = vecs[:, j]
            povm.append(np.outer(v, v.conj()) / (dim + 1))
    return povm


def chop(x, tol=1e-10):
    """Zero out real/imaginary parts smaller than `tol` (like Mathematica's Chop)."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        re = np.where(np.abs(x.real) < tol, 0.0, x.real)
        im = np.where(np.abs(x.imag) < tol, 0.0, x.imag)
        return re + 1j * im
    return np.where(np.abs(x) < tol, 0.0, x)


def dirt_multinomial(v, stat):
    """Empirical frequencies from `stat` multinomial draws, for a
    probability vector `v` (1D) or for every row of a matrix `v` (2D).
    """
    v = np.real(np.asarray(v))

    if v.ndim not in (1, 2):
        raise ValueError("v must be a vector or a matrix.")
    if not isinstance(stat, (int, np.integer)) or stat <= 0:
        raise ValueError("stat must be a positive integer.")

    return rng.multinomial(stat, v) / stat


def quantiles(v):
    """[10th percentile, median, 90th percentile, mean] of a 1D sample."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [
            np.percentile(v, 10),
            np.percentile(v, 50),
            np.percentile(v, 90),
            np.mean(v),
        ]
    )


def to_dm(v):
    """Unnormalized density matrix |v><v| for a ket `v`."""
    v = np.asarray(v)
    return np.outer(v, v.conj())


def random_unitary(dim):
    """Haar-random unitary matrix (Mezzadri's QR algorithm)."""
    z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return q * phases


def random_kets(dim, n):
    """`n` independent Haar-random kets, as the rows of an (n, dim) array.

    Obtained by normalizing standard complex Gaussian vectors, which is
    distributed identically to (and much cheaper than) taking the first
    column of a Haar-random unitary.
    """
    v = np.random.randn(n, dim) + 1j * np.random.randn(n, dim)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def flattened_states(dim, n):
    """`n` Haar-random pure-state density matrices, flattened row-major, as rows."""
    kets = random_kets(dim, n)
    return np.array([np.outer(k, k.conj()).flatten() for k in kets])


def povm_probabilities(M_povm, M_states):
    """p_a(rho) = Tr(mu_a rho) for every POVM element a (rows of `M_povm`)
    and every state rho (rows of `M_states`), as a real (n_out, n_states)
    matrix (not renormalized).
    """
    return chop(M_povm.conj() @ M_states.T).real


def random_povm(dim_in, dim_out):
    """Random rank-one POVM with `dim_out` outcomes on a `dim_in`-dim space.

    Returns an (dim_out, dim_in**2) array whose rows are the flattened
    (row-major) POVM elements.
    """
    v = random_unitary(dim_out)[:, :dim_in]
    return np.array([to_dm(row).flatten() for row in v])


# ----------------------------------------------------------------------
# Multinomial shot noise vs. its Gaussian approximation (used by
# ``noise_comparison/`` and ``check_fit/``).
# ----------------------------------------------------------------------


def sigma_from_p(p):
    """Per-shot covariance ``Sigma(p) = diag(p) - p p^T`` of a single
    multinomial trial with outcome probabilities `p`. The covariance of
    the empirical frequency after `N` shots is ``Sigma(p) / N``. `Sigma`
    is singular (its null space is the all-ones direction, since outcome
    probabilities are constrained to sum to 1).
    """
    p = np.real(np.asarray(p))
    return np.diag(p) - np.outer(p, p)


def sigma_batch(P):
    """`sigma_from_p` for every column of `P` (shape ``(n_out, n_cols)``),
    vectorized. Returns an ``(n_cols, n_out, n_out)`` array.
    """
    Pt = np.real(np.asarray(P)).T  # (n_cols, n_out)
    Sigma = -Pt[:, :, None] * Pt[:, None, :]
    idx = np.arange(Pt.shape[1])
    Sigma[:, idx, idx] += Pt
    return Sigma


def multinomial_phat(p, N, size=1):
    """`size` iid empirical-frequency draws ``n / N``, ``n ~
    Multinomial(N, p)``, for a single probability vector `p`. Returns
    shape ``(size, len(p))``.
    """
    p = np.real(np.asarray(p))
    return rng.multinomial(N, p, size=size) / N


def multinomial_phat_batch(P, N, R):
    """`R` iid Multinomial(N, ·)/N draws for every column of `P` (shape
    ``(n_out, n_cols)``). Returns an ``(R, n_out, n_cols)`` array.
    """
    Pt = np.real(np.asarray(P)).T  # (n_cols, n_out)
    pvals = np.broadcast_to(Pt, (R,) + Pt.shape)
    counts = rng.multinomial(N, pvals)  # (R, n_cols, n_out)
    return counts.transpose(0, 2, 1) / N


def sample_gaussian_from_covariance(Sigma, size=1, tol=1e-8):
    """Sample `size` iid draws of ``xi ~ N(0, Sigma)`` via an
    eigendecomposition of `Sigma`, discarding eigenvalues below `tol`
    times the largest one (needed because covariance matrices such as
    `sigma_from_p` are singular).

    `Sigma` has shape ``(..., n, n)``: a single covariance matrix, or a
    batch of them stacked along leading axes (e.g. from `sigma_batch`).
    Returns an array of shape ``(size, ..., n)``: `size` independent draws
    for every matrix in the batch.
    """
    Sigma = np.asarray(Sigma)
    eigvals, eigvecs = np.linalg.eigh(Sigma)  # ascending; (...,n), (...,n,n)
    max_eig = np.max(eigvals, axis=-1, keepdims=True)
    lam = np.where(eigvals > tol * np.maximum(max_eig, tol), eigvals, 0.0)
    z = rng.standard_normal(size=(size,) + Sigma.shape[:-1])
    return np.einsum("...ij,...j->...i", eigvecs, z * np.sqrt(lam))


def covariance_rank_info(Sigma, tol=1e-8):
    """``(rank, smallest_nonzero_eigenvalue)`` of `Sigma` (shape
    ``(..., n, n)``), using the same eigenvalue tolerance convention as
    `sample_gaussian_from_covariance`.
    """
    eigvals = np.linalg.eigvalsh(Sigma)
    max_eig = np.max(eigvals, axis=-1, keepdims=True)
    keep = eigvals > tol * np.maximum(max_eig, tol)
    rank = keep.sum(axis=-1)
    smallest_nonzero = np.where(keep, eigvals, np.inf).min(axis=-1)
    return rank, smallest_nonzero


# ----------------------------------------------------------------------
# Text format of the saved MSE data (Mathematica list notation).
# ----------------------------------------------------------------------


def _mathematica_real(x):
    """Format a float the way Mathematica's ToString/Export renders a
    machine real: plain decimal for magnitudes >= 1e-5, and scientific
    notation with the ``*^`` exponent marker below that.
    """
    x = float(x)
    if x == 0:
        return "0."
    sign = "-" if x < 0 else ""
    ax = abs(x)
    _, digits, exponent = Decimal(repr(ax)).as_tuple()
    exp10 = exponent + len(digits) - 1
    digit_str = "".join(str(d) for d in digits)

    if exp10 < -5:
        mantissa = digit_str[0] + "." + digit_str[1:] if len(digit_str) > 1 else digit_str[0] + "."
        return f"{sign}{mantissa}*^{exp10}"

    if exp10 >= 0:
        int_len = exp10 + 1
        if int_len >= len(digit_str):
            int_part = digit_str + "0" * (int_len - len(digit_str))
            frac_part = ""
        else:
            int_part = digit_str[:int_len]
            frac_part = digit_str[int_len:]
        return f"{sign}{int_part}.{frac_part}"

    leading_zeros = -exp10 - 1
    return f"{sign}0." + "0" * leading_zeros + digit_str


def save_mse_data(path, x_values, res_mse):
    """Save a (swept parameter, quantiles) result to a text file in
    Mathematica list notation, matching the original notebooks'
    ``Export[file, {statLIST, resMSE}]`` output:

        {x1, x2, ..., xn}
        {{p10, p50, p90, mean}, {p10, p50, p90, mean}, ...}

    `x_values` are written as plain integers; `res_mse` rows are written
    as machine reals, using ``*^`` scientific notation below 1e-5 as
    Mathematica does.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_line = "{" + ", ".join(str(int(x)) for x in x_values) + "}"
    rows = (
        "{" + ", ".join(_mathematica_real(v) for v in row) + "}"
        for row in np.asarray(res_mse)
    )
    res_line = "{" + ", ".join(rows) + "}"
    with open(path, "w") as f:
        f.write(x_line + "\n")
        f.write(res_line + "\n")


def load_mse_data(path):
    """Inverse of `save_mse_data`: read back (x_values, res_mse) from a
    file written in Mathematica list notation (``{...}`` on each line,
    numbers optionally using the ``*^`` exponent marker).
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    x_values = np.array([int(v) for v in lines[0].strip("{}").split(",")])
    rows = re.findall(r"\{([^{}]*)\}", lines[1])
    res_mse = np.array(
        [[float(tok.strip().replace("*^", "e")) for tok in row.split(",")] for row in rows]
    )
    return x_values, res_mse
