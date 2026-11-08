"""Figure 3: positional mechanisms (E3, Section 5.2).

(a) Balanced accuracy per mechanism (rows, grouped by the logic each realizes) and task x serialization
    (columns). Each cell: in-distribution mean (upper, 8 pt) and out-of-distribution mean (lower, 7.5 pt);
    cell fill = in-distribution value. Marker left of the numbers = theory at the round budget L = 6:
    filled = a route exists for every instance, half = only for k_min <= k* (Prop. new:prop-rounds),
    hollow = no route (Theorem thm:noshift), dotted ring = route relies on an approximation (recency or relative
    positions; assumption, not a theorem).
(b) Out-of-distribution k-controlled test sets for 'v has a red neighbor': balanced accuracy by the
    position k_min of v in its red neighbors' lists, AL and NC under four mechanisms; the theoretical
    frontier k* of 1-local heads (AL 2, NC 1) is marked.
"""
import json
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from fmt import fstr  # noqa: E402
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Wedge, Circle

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, seq_cmap, style_axes, vcol

W, H = TEXTWIDTH_IN, 2.00
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import is_measured, load_merged  # noqa: E402
D = load_merged()["synthetic"]
C = D["cells"]
K = D["kcontrolled"]
VAL = INK
ARMS = [("NoPE", "NoPE", r"$\mathsf{P},\mathsf{M}$"),
        ("1-local", "1-local", r"$+\,\mathsf{Y}$"),
        ("ALiBi-only", "ALiBi only", r"$\approx\mathsf{S}$"),
        ("1-local+ALiBi", "1-local+ALiBi", r"$+\,\mathsf{Y},\ \approx\mathsf{S}$"),
        ("1-local+hard-since", "1-local+hard since", r"$+\,\mathsf{Y},\ \mathsf{S}$"),
        ("RoPE", "RoPE", r"relative positions")]
FM = ["EL", "AL", "NC", "ECL"]
TASKS = [("red", r"$\mathit{red}$"), ("dia", r"$\Diamond\mathit{red}$")]
KS = ["1", "2", "3", "4-6", "7-12", "13-32"]
KLAB = ["1", "2", "3", "4--6", "7--12", "13--32"]
KSTAR = {"AL": 2, "NC": 1}


def theory(arm, task, fmt):
    """'y' route for all, 'h' only k <= k*, 'n' none, 'a' approximate (assumption)."""
    if arm == "NoPE":
        return "n"
    if arm in ("ALiBi-only", "RoPE"):
        return "a"
    if task == "red":
        return "y"
    if fmt in ("EL", "ECL") or arm == "1-local+hard-since":
        return "y"
    return "h" if arm == "1-local" else "a"


def marker(ax, x, y, st, color, r_pt=2.6):
    """Theory glyph: true circle of radius r_pt points centered at data point (x, y)."""
    from matplotlib.transforms import ScaledTranslation
    fig = ax.figure
    tr = fig.dpi_scale_trans + ScaledTranslation(x, y, ax.transData)
    r = r_pt / 72
    lw = 0.9
    if st == "y":
        ps = [Circle((0, 0), r, fc=color, ec=color, lw=lw)]
    elif st == "n":
        ps = [Circle((0, 0), r, fc="white", ec=color, lw=lw)]
    elif st == "h":
        ps = [Circle((0, 0), r, fc="white", ec=color, lw=lw), Wedge((0, 0), r, 90, 270, fc=color, ec=color, lw=0)]
    else:
        ps = [Circle((0, 0), r, fc="white", ec=color, lw=lw, ls=(0, (1.0, 0.9)))]
    for z, p in enumerate(ps):
        p.set_transform(tr); p.set_zorder(4 + z); p.set_clip_on(False)
        ax.add_artist(p)


def panel_a(ax):
    cmap = seq_cmap()
    nrow, ncol = len(ARMS), 8
    data = [[C[f"{a}|{t}|{f}|id"]["mean"] for t, _ in TASKS for f in FM] for a, _, _ in ARMS]
    ax.imshow(data, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto", interpolation="nearest", zorder=0)
    sx = ax.get_position().width * W / ncol                 # inches per column
    sy = ax.get_position().height * H / nrow                # inches per row
    for i, (a, _, _) in enumerate(ARMS):
        for c in range(ncol):
            t, f = TASKS[c // 4][0], FM[c % 4]
            idv, oodv = C[f"{a}|{t}|{f}|id"]["mean"], C[f"{a}|{t}|{f}|ood"]["mean"]
            bw, bh = 0.84, 0.80
            ax.add_patch(FancyBboxPatch((c - bw / 2, i - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.10",
                                        fc="white", ec="none", zorder=2, alpha=1.0))
            ax.text(c + 0.16, i - 0.165, fstr(idv, 3)[1:], ha="center", va="center", fontsize=8,
                    color=vcol(idv), zorder=3)
            ax.text(c + 0.16, i + 0.205, fstr(oodv, 3)[1:], ha="center", va="center", fontsize=7.5,
                    color=vcol(oodv), zorder=3)
            # glyph: radius 2.1 pt, expressed in data units (x and y differ; draw as ellipse-free circle on x)
            marker(ax, c - 0.315, i - 0.165, theory(a, t, f), SER[f]["color"])
    ax.axvline(3.5, color="white", lw=3.0, zorder=1)
    for y in (0.5, 1.5, 3.5, 4.5):
        ax.axhline(y, color="white", lw=1.2, zorder=1)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(FM * 2, fontsize=8)
    for k, lab in enumerate(ax.get_xticklabels()):
        lab.set_color(SER[FM[k % 4]]["color"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([rf"\shortstack[r]{{{n}\\[-1pt]\footnotesize\color[HTML]{{5A6270}}{lg}}}" for _, n, lg in ARMS],
                       fontsize=8)
    ax.tick_params(length=0, pad=2.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, ncol - 0.5); ax.set_ylim(nrow - 0.5, -0.5)
    for k, (_, lab) in enumerate(TASKS):
        ax.text(1.5 + 4 * k, -1.30, lab, ha="center", va="bottom", fontsize=8.5, color=INK)
        h = 0.06
        ax.plot([4 * k - 0.5 + (h if k else 0), 4 * k + 3.5 - (0 if k else h)], [-1.20, -1.20], color=QUIET, lw=1.0,
                clip_on=False, solid_capstyle="butt")


def panel_b(ax):
    ax.set_facecolor(PANEL_BG)
    xs = list(range(len(KS)))
    ax.axhline(0.5, color=QUIET, lw=1.2, ls=(0, (4, 3)), zorder=1)
    styles = {"1-local": dict(ls="-", lw=1.6, ms=4.2), "1-local+ALiBi": dict(ls=(0, (4, 1.6)), lw=1.4, ms=3.6),
              "1-local+hard-since": dict(ls=(0, (1, 1.3)), lw=1.6, ms=0), "RoPE": dict(ls=(0, (5, 1.5, 1, 1.5)), lw=1.3, ms=0)}
    for f, dx in (("AL", -0.06), ("NC", 0.06)):
        st = SER[f]
        for arm, s in styles.items():
            ys = [K[f"{arm}|{f}|{k}"]["mean"] for k in KS]
            es = [K[f"{arm}|{f}|{k}"]["sd"] for k in KS]
            ax.errorbar([x + dx for x in xs], ys, yerr=es, color=st["color"], ls=s["ls"], lw=s["lw"],
                        marker=st["marker"] if s["ms"] else None, ms=s["ms"], mfc=st["color"], mec="white", mew=0.6,
                        elinewidth=0.9, capsize=1.4, zorder=3)
    # frontier k* of 1-local heads: boundary between the last solved and the first failed stratum
    for f, y in (("AL", 0.585), ("NC", 0.585)):
        ks = KSTAR[f]
        x = ks - 1 + 0.5
        ax.plot([x, x], [0.44, 1.01], color=SER[f]["color"], lw=0.9, ls=(0, (2, 1.5)), zorder=2, clip_on=False)
        ax.text(x, 1.015, rf"$k^*_{{\mathrm{{{f}}}}}$", ha="center", va="bottom",
                fontsize=7.5, color=SER[f]["color"], zorder=4)
    # inline format labels in empty spots (left of each format's drop, clear of every line), on a panel-colored box
    for f, (x, y) in (("NC", (0.12, 0.60)), ("AL", (1.10, 0.68))):
        ax.text(x, y, f, ha="left", va="center", fontsize=8, color=SER[f]["color"], zorder=6,
                bbox=dict(boxstyle="square,pad=0.08", fc=PANEL_BG, ec="none"))
    ax.set_xticks(xs); ax.set_xticklabels(KLAB, fontsize=8)
    ax.set_xlim(-0.4, len(KS) - 0.6); ax.set_ylim(0.44, 1.01)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8.5, labelpad=2)
    ax.set_xlabel(r"\textbf{Position} $\boldsymbol{k_{\min}}$ \textbf{of $\boldsymbol{v}$ in red neighbors' lists}",
                  fontsize=8.5, labelpad=2)
    style_axes(ax)
    return styles


def main(out="fig3.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    TOP, BOT = 1.50, 0.07          # heat map rows; (b) plot area shares TOP
    BOTB = 0.36                    # (b) plot-area bottom (room for its x label)
    M, GAP = 0.07, 0.24            # outer margins, ink gap between panels
    xa, wa = 0.95, 2.80
    ax_a = fig.add_axes([xa / W, BOT / H, wa / W, (TOP - BOT) / H])
    ax_b = fig.add_axes([4.5 / W, BOTB / H, 1.5 / W, (TOP - BOTB) / H])
    panel_a(ax_a)
    styles = panel_b(ax_b)
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    inch = lambda bb: (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
    fig.canvas.draw()
    ia = inch(drawn_extent(ax_a, r))
    xa += M - ia[0]
    ax_a.set_position([xa / W, BOT / H, wa / W, (TOP - BOT) / H])
    fig.canvas.draw()
    ia = inch(drawn_extent(ax_a, r))
    xb, wb = ia[2] + GAP + 0.35, 1.5
    for _ in range(4):
        ax_b.set_position([xb / W, BOTB / H, wb / W, (TOP - BOTB) / H])
        fig.canvas.draw()
        ib = inch(drawn_extent(ax_b, r))
        xb += (ia[2] + GAP) - ib[0]
        wb += (W - M) - ib[2] - ((ia[2] + GAP) - ib[0])
    ax_b.set_position([xb / W, BOTB / H, wb / W, (TOP - BOTB) / H])
    fig.canvas.draw()
    ia, ib = inch(drawn_extent(ax_a, r)), inch(drawn_extent(ax_b, r))
    # (b) legends: format color above the plot; mechanism line styles inside the empty band of the plot
    names = {"1-local": "1-local", "1-local+ALiBi": r"1-local+ALiBi", "1-local+hard-since": r"1-local+hard since",
             "RoPE": "RoPE"}
    hs2 = [Line2D([], [], color=QUIET, ls=s_["ls"], lw=s_["lw"]) for s_ in styles.values()]
    leg2 = ax_b.legend(hs2, list(names.values()), loc="lower right", bbox_to_anchor=(1.0, 0.17), ncol=1,
                       frameon=False, fontsize=7.5, handlelength=2.4, handletextpad=0.4, labelspacing=0.22,
                       borderaxespad=0.0, borderpad=0.1)
    leg1 = leg2
    # color key of the cell shade in (a): in-distribution balanced accuracy, 0.5 (light) to 1.0 (dark)
    cax = fig.add_axes([(ia[0] + 0.05) / W, (TOP + 0.085) / H, 0.50 / W, 0.06 / H])
    import numpy as _np
    cax.imshow(_np.linspace(0.5, 1.0, 64)[None, :], cmap=seq_cmap(), vmin=0.5, vmax=1.0, aspect="auto")
    cax.set_xticks([0, 63]); cax.set_xticklabels([".5", "1"], fontsize=6.5); cax.set_yticks([])
    cax.tick_params(length=1.5, pad=1)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
    fig.text((ia[0] + 0.05) / W, (TOP + 0.20) / H, "shade: upper value", ha="left", va="center",
             fontsize=6.2, color=INK)
    fig.text(ia[0] / W, (H - 0.02) / H, r"\textbf{(a)}", ha="left", va="top", fontsize=10, color=INK)
    fig.text(ib[0] / W, (H - 0.02) / H, r"\textbf{(b)}", ha="left", va="top", fontsize=10, color=INK)
    fig.canvas.draw()
    ia, ib = inch(drawn_extent(ax_a, r)), inch(drawn_extent(ax_b, r))
    l1 = inch(leg1.get_window_extent(r))
    print(f"(a) ink x {ia[0]:.3f}-{ia[2]:.3f} y {ia[1]:.3f}-{ia[3]:.3f}; (b) ink x {ib[0]:.3f}-{ib[2]:.3f} "
          f"y {ib[1]:.3f}-{ib[3]:.3f}; gap {ib[0] - ia[2]:.3f}; right margin {W - ib[2]:.3f}")
    print(f"legend1 x {l1[0]:.2f}-{l1[2]:.2f} y {l1[1]:.3f}-{l1[3]:.3f}")
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
