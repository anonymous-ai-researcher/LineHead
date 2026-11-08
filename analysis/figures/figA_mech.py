"""Appendix figure: mechanism evidence on Qwen2.5-7B (E9, 28 layers).
(a) Linear-probe accuracy for the line head's color, read at entry-name tokens, by layer; one line per
    token-distance bin between the entry and its head color (AL dashed, NC solid; darker = nearer).
(b) Activation patching: share of the clean-minus-corrupted logit difference restored by patching the
    head-color token's residual stream at each layer (AL, NC).
(c) EL: probe for the second endpoint's own color at its position (inner match), by layer; the vertical
    line marks the layer where the since probe (NC, nearest bin) peaks.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, style_axes

W, H = TEXTWIDTH_IN, 1.80
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import load_merged  # noqa: E402
D = load_merged()["mechanism"]
BINS = ["1-8", "9-24", ">24"]
BLAB = ["1--8", "9--24", r"$>$24"]
ALPHA = [1.0, 0.65, 0.35]
L = list(range(29))


def main(out="figA_mech.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    bot, top, M, gap = 0.36, 1.50, 0.07, 0.30
    lefts = [0.48]
    widths = [2.05, 1.55, 1.55]
    xs = [0.48, 0.48 + 2.05 + gap + 0.08, 0.48 + 2.05 + 1.55 + 2 * gap + 0.16]
    axs = [fig.add_axes([x / W, bot / H, w / W, (top - bot) / H]) for x, w in zip(xs, widths)]
    a, b, c = axs
    for ax in axs:
        ax.set_facecolor(PANEL_BG)
        ax.axhline(0.5 if ax is not b else 0.0, color=QUIET, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for f, ls in (("AL", (0, (4, 1.6))), ("NC", "-")):
        for k, (bn, al) in enumerate(zip(BINS, ALPHA)):
            if f"{f}|{bn}" not in D["probe"]:
                continue
            a.plot(L, D["probe"][f"{f}|{bn}"], color=SER[f]["color"], alpha=al, lw=1.5, ls=ls, zorder=3)
    a.set_ylim(0.45, 1.0); a.set_xlim(0, 28)
    a.set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8.5, labelpad=2)
    a.set_title(r"\textbf{Head color at entries}", fontsize=8.5, pad=3)
    for f, ls in (("AL", (0, (4, 1.6))), ("NC", "-")):
        b.plot(L, D["patch"][f], color=SER[f]["color"], lw=1.7, ls=ls, marker=SER[f]["marker"], ms=3.0,
               mfc=SER[f]["color"], mec="white", mew=0.5, zorder=3)
    b.set_ylim(-0.05, 1.05); b.set_xlim(0, 28)
    hb = [Line2D([], [], color=SER[f]["color"], lw=1.7, ls=ls, marker=SER[f]["marker"], ms=3.0, mfc=SER[f]["color"],
                 mec="white", mew=0.5) for f, ls in (("AL", (0, (4, 1.6))), ("NC", "-"))]
    b.legend(hb, ["AL", "NC"], loc="lower left", frameon=False, fontsize=7.5, handlelength=1.8, handletextpad=0.3,
             borderaxespad=0.3)
    b.set_ylabel(r"\textbf{Logit diff.\ restored}", fontsize=8.5, labelpad=2)
    b.set_title(r"\textbf{Patching the head color}", fontsize=8.5, pad=3)
    since_peak = max(L, key=lambda i: D["probe"]["NC|1-8"][i])
    inner_peak = max(L, key=lambda i: D["el_inner"][i])
    c.plot(L, D["el_inner"], color=SER["EL"]["color"], lw=1.7, marker=SER["EL"]["marker"], ms=3.0,
           mfc=SER["EL"]["color"], mec="white", mew=0.5, zorder=3)
    c.axvline(since_peak, color=SER["NC"]["color"], lw=1.0, ls=(0, (2, 1.5)), zorder=2)
    c.text(since_peak + 0.5, 0.99, r"since peak", ha="left", va="top", fontsize=7.5, color=SER["NC"]["color"])
    c.axvline(inner_peak, color=SER["EL"]["color"], lw=1.0, ls=(0, (2, 1.5)), zorder=2)
    c.text(inner_peak - 0.5, 0.99, r"inner peak", ha="right", va="top", fontsize=7.5, color=SER["EL"]["color"])
    c.set_ylim(0.45, 1.0); c.set_xlim(0, 28)
    c.set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8.5, labelpad=2)
    c.set_title(r"\textbf{EL: own color (inner match)}", fontsize=8.5, pad=3)
    for ax in axs:
        ax.set_xticks([0, 7, 14, 21, 28])
        ax.set_xlabel(r"\textbf{Layer}", fontsize=8.5, labelpad=1)
        style_axes(ax)
    # legend for (a): format line styles and distance shades, above the panels
    hs = [Line2D([], [], color=SER["AL"]["color"], lw=1.5, ls=(0, (4, 1.6))), Line2D([], [], color=SER["NC"]["color"], lw=1.5)]
    hs += [Line2D([], [], color=INK, alpha=al, lw=2.0) for al in ALPHA]
    # compact, in the empty upper-left corner (layers 0-6, where every curve is still at 0.5): no overlap
    leg = a.legend(hs, ["AL", "NC"] + list(BLAB), loc="upper left", ncol=1, frameon=False,
                   fontsize=7.5, handlelength=1.2, handletextpad=0.3, columnspacing=0.6, labelspacing=0.15,
                   borderaxespad=0.2)
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    G = 0.22                                          # ink gap between panels
    ratio = [1.3, 1.0, 1.0]
    for _ in range(4):
        fig.canvas.draw()
        ink = [drawn_extent(ax, r) for ax in axs]
        pos = [ax.get_position() for ax in axs]
        le = [p.x0 * W - i.x0 / dpi for p, i in zip(pos, ink)]       # ink left of axes
        re = [i.x1 / dpi - p.x1 * W for p, i in zip(pos, ink)]       # ink right of axes
        avail = W - 2 * M - 2 * G - sum(le) - sum(re)
        ws = [avail * q / sum(ratio) for q in ratio]
        x = M
        for k, ax in enumerate(axs):
            x += le[k]
            ax.set_position([x / W, bot / H, ws[k] / W, (top - bot) / H])
            x += ws[k] + re[k] + G
    fig.canvas.draw()
    ink = [drawn_extent(ax, r) for ax in axs]
    for k, ax in enumerate(axs):
        fig.text(ink[k].x0 / dpi / W, (H - 0.02) / H, rf"\textbf{{({'abc'[k]})}}", ha="left", va="top",
                 fontsize=10, color=INK)
    fig.canvas.draw()
    lb = leg.get_window_extent(r)
    print("ink", [(round(b_.x0 / dpi, 3), round(b_.x1 / dpi, 3)) for b_ in ink], "legend",
          round(lb.x0 / dpi, 2), round(lb.x1 / dpi, 2), round(lb.y0 / dpi, 3), round(lb.y1 / dpi, 3))
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
