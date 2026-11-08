"""Figure 4: LLMs (Section 5.3).

(a) Route dissection by partial masking (E7) and the NC variant without head colors (NC-H, E4): balanced
    accuracy of 'v has a red neighbor' (D1a, also for NC-H) and of 'v is red' (Q0), mean over the
    four LLMs (bars, with the 95% interval of that mean from a joint pair bootstrap: one resample of pairs per
    question applied to every model, as in Figure 1b) and each model (dots). Glyph above a bar =
    theory (Prop. new:prop-mask): filled = a route survives, hollow = no model can decide it.
(b) Depth-one questions (D1a, D1c positives) by the position k_min of v in its red neighbors' lists, for
    EL, AL, NC and ECL; line = mean over models, band = range over the four models.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, style_axes

W, H = TEXTWIDTH_IN, 1.90
ROOT = Path(__file__).resolve().parent.parent
import sys as _s; _s.path.insert(0, str(ROOT))
from load_data import load_merged  # noqa: E402
D = load_merged()
from load_data import is_measured, sub_measured  # noqa: E402
MODELS = list(D["llm_main"]["models"])
TOP, BOT, M, GAP = 1.40, 0.50, 0.07, 0.26
# condition key, format color, label (two lines), theory (D1a, Q0): 'y' route survives, 'n' no model decides
COND = [("AL-none", "AL", r"full", ("y", "y")), ("AL-head", "AL", r"line heads\\[-1pt]masked", ("y", "n")),
        ("AL-entry", "AL", r"entries\\[-1pt]masked", ("n", "y")), ("NC-none", "NC", r"full", ("y", "y")),
        ("NC-head", "NC", r"line heads\\[-1pt]masked", ("y", "y")), ("NC-entry", "NC", r"entries\\[-1pt]masked", ("y", "y")),
        ("NC-H", "NC", r"no head\\[-1pt]colors", ("y", None))]
KS = ["1", "2", "3-4", ">=5"]
KLAB = ["1", "2", "3--4", r"$\geq$5"]


def val(m, key, q):
    if key == "NC-H":
        return D["llm_formats"][m]["NC-H"]["D1a"] if q == "D1a" else None
    return D["llm_partial"][m][key][q]


def panel_a(ax):
    ax.set_facecolor(PANEL_BG)
    ax.axhline(0.5, color=QUIET, lw=1.2, ls=(0, (4, 3)), zorder=1)
    bw = 0.36
    for i, (key, f, lab, th) in enumerate(COND):
        col = SER[f]["color"]
        for j, q in enumerate(("D1a", "Q0")):
            cs = [val(m, key, q) for m in MODELS]
            if cs[0] is None:
                continue
            x = i + (j - 0.5) * (bw + 0.04)
            J = D["llm_main"]["partial_mean_ci"][key][q]   # four-model mean, joint pair-bootstrap 95% interval
            mu = J["ba"]
            hw = [[mu - J["lo"]], [J["hi"] - mu]]
            ax.bar(x, mu - 0.4, bottom=0.4, width=bw, color=col if q == "D1a" else "white", ec=col, lw=1.1,
                   hatch=None if q == "D1a" else "////", zorder=2)
            ax.errorbar([x], [mu], yerr=hw, color=INK, lw=0, elinewidth=0.9, capsize=1.6, zorder=4)
            for k, c in enumerate(cs):
                ax.plot([x + (k - 1.5) * 0.06], [c["ba"]], ls="none", marker="o", ms=2.2, mfc="white", mec=INK,
                        mew=0.5, zorder=5)
            t = th[j]
            ax.plot([x], [1.05], ls="none", marker="o", ms=4.4, mew=1.0,          # same glyph as the legend
                    mfc=col if t == "y" else "white", mec=col, zorder=5, clip_on=False)
    ax.axvline(2.5, color="white", lw=2.5, zorder=1)
    ax.set_xticks(range(len(COND)))
    ax.set_xticklabels([rf"\shortstack{{{lab}}}" for _, _, lab, _ in COND], fontsize=7.5)
    for k, t in enumerate(ax.get_xticklabels()):
        t.set_color(SER[COND[k][1]]["color"])
    ax.set_xlim(-0.55, len(COND) - 0.45); ax.set_ylim(0.40, 1.01)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8.5, labelpad=2)
    ax.tick_params(axis="x", length=0, pad=2)
    style_axes(ax)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for f, (x0, x1) in (("AL", (-0.42, 2.42)), ("NC", (2.58, 6.42))):
        ax.plot([x0, x1], [-0.335, -0.335], transform=tr, color=SER[f]["color"], lw=1.0, clip_on=False,
                solid_capstyle="butt")
        ax.text((x0 + x1) / 2, -0.36, rf"\textbf{{{f}}}", transform=tr, ha="center", va="top", fontsize=8.5,
                color=SER[f]["color"], clip_on=False)


def panel_b(ax):
    ax.set_facecolor(PANEL_BG)
    xs = list(range(len(KS)))
    ext = []
    for f in ("EL", "AL", "NC", "ECL"):
        st = SER[f]
        # change from k_min = 1 (points), per model; the absolute values are in Table T_kstrata
        per = [[100 * (D["llm_main"]["strata"][m][f]["kmin"][k] - D["llm_main"]["strata"][m][f]["kmin"]["1"])
                for k in KS] for m in MODELS]
        mu = [sum(p[i] for p in per) / len(per) for i in xs]
        lo = [min(p[i] for p in per) for i in xs]
        hi = [max(p[i] for p in per) for i in xs]
        ext += lo + hi
        dx = {"EL": -0.09, "AL": -0.03, "NC": 0.03, "ECL": 0.09}[f]       # side by side, so no line hides another
        xo = [x + dx for x in xs]
        ax.fill_between(xo, lo, hi, color=st["color"], alpha=0.13, lw=0, zorder=2)
        ax.plot(xo, mu, color=st["color"], lw=1.7, ls=(0, (4, 1.6)) if f == "ECL" else st.get("ls", "-"), marker=st["marker"],
                ms=4.6 if f != "ECL" else 4.0, mfc=st["color"] if f != "ECL" else "white", mec="white" if f != "ECL" else st["color"],
                mew=0.8 if f != "ECL" else 1.2, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(KLAB, fontsize=8)
    ax.set_xlim(-0.25, len(KS) - 0.75)
    lo_all, hi_all = min(ext), max(ext)
    import math as _m
    ax.set_ylim(min(-6, 2 * _m.floor(lo_all / 2) - 1), max(2, 2 * _m.ceil(hi_all / 2) + 1))
    ax.axhline(0, color=QUIET, lw=0.8, zorder=1)
    ax.set_ylabel(r"\textbf{Change from} $\boldsymbol{k_{\min}=1}$ \textbf{(points)}", fontsize=8, labelpad=1)
    ax.set_xlabel(r"\textbf{Position} $\boldsymbol{k_{\min}}$ \textbf{of} $\boldsymbol{v}$ \textbf{(}$\boldsymbol{\Diamond\mathit{red}}$\textbf{)}",
                  fontsize=8.5, labelpad=2)
    style_axes(ax)


def main(out="fig4.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    ax_a = fig.add_axes([0.1, BOT / H, 0.5, (TOP - BOT) / H])
    ax_b = fig.add_axes([0.7, BOT / H, 0.25, (TOP - BOT) / H])
    panel_a(ax_a); panel_b(ax_b)
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    inch = lambda bb: (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
    xa, wa, wb = 0.4, 3.6, 1.8
    for _ in range(5):
        xb = xa + wa + GAP + 0.3
        ax_a.set_position([xa / W, BOT / H, wa / W, (TOP - BOT) / H])
        ax_b.set_position([xb / W, BOT / H, wb / W, (TOP - BOT) / H])
        fig.canvas.draw()
        ia, ib = inch(drawn_extent(ax_a, r)), inch(drawn_extent(ax_b, r))
        la, ra, lb, rb = xa - ia[0], ia[2] - (xa + wa), xb - ib[0], ib[2] - (xb + wb)
        avail = W - 2 * M - la - ra - GAP - lb - rb
        wa = 0.63 * avail; wb = avail - wa
        xa = M + la
        xb = xa + wa + ra + GAP + lb
        ax_a.set_position([xa / W, BOT / H, wa / W, (TOP - BOT) / H])
        ax_b.set_position([xb / W, BOT / H, wb / W, (TOP - BOT) / H])
    fig.canvas.draw()
    ia, ib = inch(drawn_extent(ax_a, r)), inch(drawn_extent(ax_b, r))
    ly = (TOP + 0.255) / H
    ha = [Patch(fc=QUIET, ec=QUIET, lw=1.0), Patch(fc="white", ec=QUIET, lw=1.0, hatch="////"),
          Line2D([], [], ls="none", marker="o", ms=4.4, mfc=INK, mec=INK, mew=1.0),
          Line2D([], [], ls="none", marker="o", ms=4.4, mfc="white", mec=INK, mew=1.0),
          Line2D([], [], ls="none", marker="o", ms=2.2, mfc="white", mec=INK, mew=0.5)]
    leg_a = fig.legend(ha, [r"$\Diamond\mathit{red}$", r"$\mathit{red}$", "route survives", "no model decides",
                            "single model"], loc="center", ncol=5, frameon=False, fontsize=7.5,
                       bbox_to_anchor=((ia[0] + ia[2]) / 2 / W, ly), handletextpad=0.35, columnspacing=0.9,
                       handlelength=1.3, handleheight=0.9)
    hb = [Line2D([], [], color=SER[s]["color"], lw=1.7, ls=SER[s].get("ls", "-"), marker=SER[s]["marker"],
                 ms=4.6 if s != "ECL" else 4.0, mfc=SER[s]["color"] if s != "ECL" else "white",
                 mec="white" if s != "ECL" else SER[s]["color"], mew=0.8 if s != "ECL" else 1.2)
          for s in ("EL", "AL", "NC", "ECL")]
    leg_b = fig.legend(hb, ["EL", "AL", "NC", "ECL"], loc="center", ncol=4, frameon=False, fontsize=8,
                       bbox_to_anchor=((ib[0] + ib[2]) / 2 / W, ly), handletextpad=0.3, columnspacing=0.7,
                       handlelength=1.9)
    top = (H - 0.02) / H
    fig.text(ia[0] / W, top, r"\textbf{(a)}", ha="left", va="top", fontsize=10, color=INK)
    fig.text(ib[0] / W, top, r"\textbf{(b)}", ha="left", va="top", fontsize=10, color=INK)
    fig.canvas.draw()
    la_, lb_ = inch(leg_a.get_window_extent(r)), inch(leg_b.get_window_extent(r))
    print(f"margins L {ia[0]:.3f} R {W - ib[2]:.3f}; gap {ib[0] - ia[2]:.3f}; bottoms a {ia[1]:.3f} b {ib[1]:.3f}; "
          f"tops a {ia[3]:.3f} b {ib[3]:.3f}")
    print(f"legend a x {la_[0]:.2f}-{la_[2]:.2f} (ink a {ia[0]:.2f}-{ia[2]:.2f}); legend b x {lb_[0]:.2f}-{lb_[2]:.2f} "
          f"(ink b {ib[0]:.2f}-{ib[2]:.2f}); legend y {la_[1]:.3f}-{la_[3]:.3f}")
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
