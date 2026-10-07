"""Plot Fig. 4: N^2 bias^2 (top) and N n_tr Var (bottom) vs n_tr, for
d = 2, 3, 5, 7, stacked with a shared n_tr axis, from the files in
``data/`` produced by ``bias.py`` and ``variance.py``. In each panel:
- solid line + markers + shaded band: median [p10, p90] of the empirical
  quantity;
- dashed line + error bars: median [p10, p90] of the asymptotic
  coefficient (b^2 or V) of eq. (46);
- dash-dotted horizontal line: its tight-design average, eq. (47).
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_mse_data  # noqa: E402
from markers import circle, diamond, marker_inner_style, marker_outer_style, square, styled, triangle  # noqa: E402

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts, amssymb, mathpazo, bm}"
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "bias_variance.pdf"

# Dimension -> (color, shape, legend label). `square` (from markers.py)
# renders as a diamond and `diamond` renders as a square.
STYLE = {
    2: ("#8FB032", square,  r"$d = 2$"),  # green, diamond
    3: ("#5E81B5", diamond, r"$d = 3$"),  # blue, square
    5: ("#E19C24", triangle,r"$d = 5$"),  # orange, triangle
    7: ("#EB6235", circle,  r"$d = 7$"  ),  # red, circle
}

MARKERSIZE = {2: 11, 3: 13, 5: 13, 7: 9.5}
EDGEWIDTH = 1.3

# Legend order, left to right.
DIM_ORDER = (2, 3, 5, 7)
X_LIMITS = (6, 4000)

# Per-panel settings, top to bottom.
PANELS = (
    dict(
        data_dir=DATA_DIR,
        stems=("N2bias2", "b2"),
        average=lambda d: (d - 1) * (d + 2) ** 2 / (d + 1),
        y_limits=(1, 1200),
        labels=(
            r"$N^2\,\mathrm{bias}^2_{\mathrm{tr}}$",
            r"$b^2$",
            r"$\mathbb{E}_{\sigma,\mathcal{O}}(b^2)$",
        ),
        legend_loc=[0.68, 0.66],
    ),
    dict(
        data_dir=DATA_DIR,
        stems=("NntrVar", "V"),
        average=lambda d: d * (d - 1) * (d + 2) / (d + 1),
        y_limits=(1, 800),
        labels=(
            r"$N n_{\mathrm{tr}}\,\mathrm{Var}_{\mathrm{tr}}$",
            r"$V$",
            r"$\mathbb{E}_{\sigma,\mathcal{O}}(V)$",
        ),
        legend_loc=[0.66, 0.68],
    ),
)


def load_curves(data_dir, stem):
    curves = {}
    for path in data_dir.glob(f"{stem}_vs_ntrain_MUB_d=*.txt"):
        dim = int(re.search(r"d=(\d+)", path.name).group(1))
        curves[dim] = load_mse_data(path)
    return curves


def draw_panel(ax, data_dir, stems, average, y_limits, labels, legend_loc):
    """Draw one panel; return the (line, inner, outer) handles per dimension."""
    emp_curves = load_curves(data_dir, stems[0])
    th_curves = load_curves(data_dir, stems[1])
    missing = [d for d in DIM_ORDER if d not in emp_curves or d not in th_curves]
    if missing:
        raise FileNotFoundError(
            f"no {stems[0]}/{stems[1]} data files in {data_dir} for d={missing}"
        )

    dim_handles = []
    for dim in DIM_ORDER:
        color, shape, _ = STYLE[dim]
        m = styled(shape)
        ms = MARKERSIZE[dim]

        # Empirical quantity: median line, [p10, p90] band, ring+dot markers.
        n_tr, res = emp_curves[dim]
        p10, p50, p90 = res[:, 0], res[:, 1], res[:, 2]
        ax.fill_between(n_tr, p10, p90, color=color, alpha=0.25, linewidth=0, zorder=2)
        (line,) = ax.plot(n_tr, p50, color=color, linewidth=1.6, zorder=3)
        (inner,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=ms, edgewidth=EDGEWIDTH),
        )
        (outer,) = ax.plot(
            n_tr, p50, marker=m, linestyle="None", zorder=5,
            **marker_outer_style(color, size=ms * 15 / 35),
        )

        # Asymptotic coefficient: dashed median with [p10, p90] error bars.
        n_tr_th, res_th = th_curves[dim]
        q10, q50, q90 = res_th[:, 0], res_th[:, 1], res_th[:, 2]
        ax.errorbar(
            n_tr_th, q50, yerr=[q50 - q10, q90 - q50], color=color, linestyle="--",
            linewidth=1.2, elinewidth=1.0, capsize=3, capthick=1.0, alpha=0.8, zorder=3,
        )
        ax.plot(
            n_tr_th, q50, marker=m, linestyle="None", zorder=4,
            **marker_inner_style(color, size=ms * 0.6, edgewidth=1.0),
        )

        # Tight-design average, eq. (47).
        ax.hlines(average(dim), n_tr[0], X_LIMITS[1], color=color, linestyle="-.", linewidth=1.4, zorder=3)

        dim_handles.append((line, inner, outer))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*y_limits)

    y_ticks = [5, 10, 50, 100, 500]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(lambda val, pos: rf"${val:g}$")
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    ax.grid(True, which="major", axis="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    style_handles = [
        Line2D([], [], color="0.4", linestyle="-", linewidth=1.6),
        Line2D([], [], color="0.4", linestyle="--", linewidth=1.2),
        Line2D([], [], color="0.4", linestyle="-.", linewidth=1.4),
    ]
    style_legend = ax.legend(
        style_handles,
        labels,
        loc=legend_loc,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        labelspacing=0.2,
    )
    # Pin it, so a later ax.legend() call (the d legend) doesn't replace it.
    ax.add_artist(style_legend)
    return dim_handles


def main():
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.0), sharex=True)

    for ax, panel in zip(axes, PANELS):
        dim_handles = draw_panel(ax, **panel)

    ax_bottom = axes[-1]
    ax_bottom.set_xlabel(r"$n_{\mathrm{tr}}$")
    ax_bottom.set_xlim(*X_LIMITS)
    x_ticks = [10, 50, 100, 500, 1000]
    ax_bottom.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax_bottom.xaxis.set_major_formatter(lambda val, pos: rf"${val:g}$")
    ax_bottom.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))

    dim_legend = axes[0].legend(
        dim_handles,
        [STYLE[dim][2] for dim in DIM_ORDER],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(DIM_ORDER),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        borderaxespad=0.1,
        borderpad=0.1,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", bbox_extra_artists=[dim_legend])
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
