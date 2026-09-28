"""Plot a `check_fit.py` results file against f(N) = C0 + C1/N + C2/N^2 + C3 N.

Set the parameters below to match a previous `check_fit.py` run -- DIM/
N_OUT/N_TRAIN/M/GAMMA/GAMMA_TEST must be identical, since together they
determine the results filename. The figure shows the empirical MSE(N)
of the fixed instance (mean over the realizations, with its standard
error, and the 10th-90th percentile band of the squared errors), the fit
f(N), each of its four terms separately, and the sigma/O-averaged
closed form `check_fit.averaged_fit`.

Run with ``python3 plot_check_fit.py`` from anywhere; it locates its
input/output directories next to itself.
"""
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent))  # check_fit.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common.py, markers.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_comparison"))
import check_fit as cf  # noqa: E402
import noise_comparison as nc  # noqa: E402
from markers import circle, marker_inner_style, marker_outer_style, styled  # noqa: E402

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb, mathpazo,bm}"

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 17,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.0,
    }
)

OUTPUT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Parameters identifying which saved results file to load -- must match
# a previous check_fit.py run exactly.
# ----------------------------------------------------------------------

DIM = 2
N_OUT = 16
N_TRAIN = 256
M = 10000  # None = exact test statistics
GAMMA = 0.1
GAMMA_TEST = GAMMA

EMPIRICAL_COLOR = "#5E81B5"
TERM_STYLE = {  # term -> (color, label)
    "C0": ("#8FB032", r"$C_0 = \gamma_{\rm test}^2 + w_O \Sigma_\sigma w_O^{\top}/M + \gamma^2 (1+R)\,\|P^+ q\|^2$"),
    "C1/N": ("#E19C24", r"$C_1/N = V/(N n_{\rm tr})$"),
    "C2/N^2": ("#EB6235", r"$C_2/N^2 = b^2/(N+d(d+2))^2$"),
    "C3 N": ("#A64CB8", r"$C_3 N = \gamma^2 R\, N/M$, \ $R = \frac{n_{\rm out} - d^2}{n_{\rm tr} - n_{\rm out} - 1}$"),
}
MARKERSIZE = 10
EDGEWIDTH = 1.3


def plot_fit_check(results):
    N = results["N_values"].astype(float)
    sq = results["sq_errors"]
    coeffs = results["coeffs"]
    mse = sq.mean(axis=0)
    sem = sq.std(axis=0, ddof=1) / np.sqrt(sq.shape[0])
    p10, mse2, p90 = np.percentile(sq, [10, 50, 90], axis=0)
    fit = cf.mse_fit(N, coeffs)
    fit_args = (results["dim"], results["n_train"], results["n_out"], results["gamma"], results["gamma_test"], results["M"])
    N_dense = np.logspace(np.log10(N[0]), np.log10(N[-1]), 400)

    fig, ax = nc.plt.subplots(figsize=(8.0, 5.8))

    term_handles = []
    #for name, values in cf.fit_terms(N_dense, coeffs).items():
    #    color, label = TERM_STYLE[name]
    #    (h,) = ax.plot(N_dense, values, color=color, linestyle="--", linewidth=1.3, zorder=2)
    #    term_handles.append((h, label))
    (fit_line,) = ax.plot(N_dense, cf.mse_fit(N_dense, coeffs), color="k", linestyle="-.", linewidth=2.0, zorder=4)
    #(avg_line,) = ax.plot(N_dense, cf.averaged_fit(N_dense, *fit_args), color="0.45", linestyle="-", linewidth=2.0, zorder=4)

    ax.fill_between(N, p10, p90, color=EMPIRICAL_COLOR, alpha=0.2, linewidth=0, zorder=1)
    band_proxy = Patch(facecolor=EMPIRICAL_COLOR, alpha=0.2, linewidth=0)

    m = styled(circle)
    ax.errorbar(N, mse, yerr=sem, color=EMPIRICAL_COLOR, linestyle="None", elinewidth=1.2, capsize=2, zorder=3)
    ax.plot(N, mse, color=EMPIRICAL_COLOR, linewidth=1.4, zorder=3)
    ax.plot(N, mse, marker=m, linestyle="None", zorder=5, **marker_inner_style(EMPIRICAL_COLOR, size=MARKERSIZE, edgewidth=EDGEWIDTH))
    ax.plot(N, mse, marker=m, linestyle="None", zorder=6, **marker_outer_style(EMPIRICAL_COLOR, size=MARKERSIZE * 15 / 35))
    emp_proxy = Line2D([], [], color=EMPIRICAL_COLOR, marker="o", markerfacecolor="white", markersize=8, linewidth=1.4)

    x_ticks = tuple(10**k for k in range(1, int(np.ceil(np.log10(N[-1]))) + 1))
    nc._style_axes(ax, r"$N$", "MSE", log_y=True, x_ticks=x_ticks)
    ax.set_xlim(N[0] / 2, N[-1] * 2)
    positive = np.concatenate([mse, fit, cf.averaged_fit(N, *fit_args), p10, p90])
    ax.set_ylim(positive.min() / 5, positive.max() * 5)

    M_str = r"\infty" if results["M"] is None else str(results["M"])
    if results["gamma"] == results["gamma_test"]:
        gamma_str = rf"$\gamma=\gamma_{{\rm test}}={results['gamma']}$"
    else:
        gamma_str = rf"$\gamma={results['gamma']}$, $\gamma_{{\rm test}}={results['gamma_test']}$"
    #ax.set_title(
    #    rf"$d={results['dim']}$, $n_{{\rm out}}={results['n_out']}$, $n_{{\rm tr}}={results['n_train']}$, "
    #    rf"$M={M_str}$, " + gamma_str,
    #    fontsize=15,
    #)

    main_legend = ax.legend(
        [emp_proxy],# fit_line],#, avg_line],
        [rf"$\gamma=${GAMMA}"],#, r"$f(N)$", r"$f(N)$ averaged over $\sigma, O$"],
        loc="upper center", frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.add_artist(main_legend)
    ax.legend(
        [h for h, _ in term_handles], [lbl for _, lbl in term_handles],
        loc="upper left", bbox_to_anchor=(1.02, 1.0), 
        frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=12,
    )

    fig.savefig(OUTPUT_DIR / f"{results_stem()}.pdf", bbox_inches="tight")
    nc.plt.close(fig)
    print(f"  wrote {results_stem()}.pdf")


def results_stem():
    return cf.results_filename(DIM, N_OUT, N_TRAIN, M, GAMMA, GAMMA_TEST).removesuffix(".pkl")


def main():
    path = cf.RESULTS_DIR / cf.results_filename(DIM, N_OUT, N_TRAIN, M, GAMMA, GAMMA_TEST)
    print(f"Loading {path}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run check_fit.py with these same parameters first.")
    results = cf.load_results(path)
    print("fit coefficients: " + "  ".join(f"{k}={v:.4e}" for k, v in results["coeffs"].items()))
    plot_fit_check(results)


if __name__ == "__main__":
    main()
