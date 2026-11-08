"""Appendix figure: depth sweep (E3d). 1-local mechanism, 'v has a red neighbor', OOD k-controlled test sets.
Rows: number of layers L; columns: position k_min of v in its red neighbors' lists. The staircase is the
theoretical frontier k*(L) (Prop. new:prop-rounds: AL entry k needs 2k+2 rounds, NC needs 5k); the
stratum 4-6 at L = 12 (k* = 5, so two of its three k values are solved) is outlined dashed. EL and
ECL: rows where only one orientation of each edge is covered (EL at L = 3, ECL at L = 3, 4) are outlined.
"""
import json
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from fmt import fstr, rnd  # noqa: E402
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, Rectangle

from style import INK, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, seq_cmap, text_on, vcol

W, H = TEXTWIDTH_IN, 1.95
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import is_measured, load_merged  # noqa: E402
D = load_merged()["depth_sweep"]
LS = [2, 3, 4, 6, 8, 12]
KS = ["1", "2", "3", "4-6", "7-12", "13-32"]
KLAB = ["1", "2", "3", "4--6", "7--12", "13--32"]
KSTAR = {"AL": [0, 0, 1, 2, 3, 5], "NC": [0, 0, 0, 1, 1, 2]}
PARTIAL = {"EL": [3], "ECL": [3, 4]}


def frontier_x(k):
    """Column boundary (data x) between solved strata (k_min <= k) and failed ones."""
    edges = {0: -0.5, 1: 0.5, 2: 1.5, 3: 2.5, 6: 3.5, 12: 4.5}
    if k in edges:
        return edges[k]
    if 3 < k < 6:                 # partial stratum: line at its right edge, cell outlined as partial
        return 3.5
    raise ValueError(k)


def panel(ax, fmt, first):
    cmap = seq_cmap()
    data = [[D[f"{L}|{fmt}"][k]["mean"] for k in KS] for L in LS]
    ax.imshow(data, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto", interpolation="nearest")
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.add_patch(FancyBboxPatch((j - 0.33, i - 0.27), 0.66, 0.54, boxstyle="round,pad=0,rounding_size=0.09",
                                        fc="white", ec="none", alpha=0.93, zorder=2))
            ax.text(j, i, fstr(v, 2)[1:], ha="center", va="center", fontsize=7.5, color=vcol(v),
                    zorder=3)
    if fmt in KSTAR:
        xs, ys = [], []
        for i, k in enumerate(KSTAR[fmt]):
            x = frontier_x(k)
            xs += [x, x]; ys += [i - 0.5, i + 0.5]
        ax.plot(xs, ys, color="white", lw=2.6, solid_joinstyle="miter", solid_capstyle="butt", zorder=3)
        ax.plot(xs, ys, color=SER[fmt]["color"], lw=1.3, solid_joinstyle="miter", solid_capstyle="butt", zorder=4)
        for i, k in enumerate(KSTAR[fmt]):
            if 3 < k < 6:
                ax.add_patch(Rectangle((2.5, i - 0.5), 1, 1, fc="none", ec=SER[fmt]["color"], lw=1.1,
                                       ls=(0, (2, 1.2)), zorder=4))
    for L in PARTIAL.get(fmt, []):
        i = LS.index(L)
        ax.add_patch(Rectangle((-0.5, i - 0.5), len(KS), 1, fc="none", ec=SER[fmt]["color"], lw=1.3, ls=(0, (2, 1.2)),
                               zorder=4))
    ax.set_xticks(range(len(KS)))
    # smaller tick labels: the long bins (7-12, 13-32) must not touch
    ax.set_xticklabels(KLAB, fontsize=6.4)
    ax.set_yticks(range(len(LS))); ax.set_yticklabels([str(L) for L in LS] if first else [], fontsize=8)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(rf"\textbf{{{fmt}}}", fontsize=8.5, color=SER[fmt]["color"], pad=3)


def main(out="figA_depth.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    M, gap, left, bot, top = 0.07, 0.10, 0.42, 0.36, 1.62
    w = (W - left - M - 3 * gap) / 4
    axs = [fig.add_axes([(left + i * (w + gap)) / W, bot / H, w / W, (top - bot) / H]) for i in range(4)]
    for i, (ax, f) in enumerate(zip(axs, ("AL", "NC", "EL", "ECL"))):
        panel(ax, f, i == 0)
    axs[0].set_ylabel(r"\textbf{Layers} $\boldsymbol{L}$", fontsize=8.5, labelpad=3)
    fig.text((left + (W - M)) / 2 / W, 0.03 / H, r"\textbf{Position} $\boldsymbol{k_{\min}}$ \textbf{of} $\boldsymbol{v}$"
             r" \textbf{in red neighbors' lists (larger graphs)}", ha="center", va="bottom", fontsize=8.5)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = [drawn_extent(a, r) for a in axs]
    print("ink x", f"{bb[0].x0 / fig.dpi:.3f}", f"{bb[-1].x1 / fig.dpi:.3f}", "W", f"{W:.3f}")
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
