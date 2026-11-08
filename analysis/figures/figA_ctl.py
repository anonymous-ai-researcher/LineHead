"""Figure (appendix, ablation controls): what the 128 heads of each ablation cost, in points of balanced
accuracy (exact-answer accuracy for GraphQA degree), mean over Qwen2.5-7B, Llama-3.1-8B and Gemma-2-9B (bars) and
the three models (dots).
(a) 'v has a red neighbor' (D1a) under the top-128 since heads at every position (E17), the same heads at list-entry
    names only (E18), the top-128 heads ranked on EL prompts (edge heads, E18) and 128 random heads in their layers.
(b) The top-128 since heads on 'v has a red neighbor' and on GraphQA node degree, which the list formats read off v's
    own line without a since step (E18, E13).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, style_axes

W, H = TEXTWIDTH_IN, 2.1
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from load_data import load_merged  # noqa: E402
D = load_merged()
F4 = ["EL", "AL", "NC", "ECL"]
ABL = {"Qwen2.5-7B": "e11a", "Llama-3.1-8B": "e11a_llama", "Gemma-2-9B": "e11a_gemma"}
TOP, BOT, M, GAP = 1.66, 0.55, 0.07, 0.34


def none(m, f):
    return D[ABL[m]]["none"][f"{f}|D1a"]["ba"]


def loss_a(cond, m, f):
    E8 = D["e18"][m]
    v = {"since": D["e17"][m]["top"][f"128|{f}|D1a"]["ba"], "entries": E8["entry"][f"128|{f}|D1a"]["ba"],
         "edge": E8["edge"][f"128|{f}|D1a"]["ba"], "random": E8["edge_random"][f"128|{f}|D1a"]["mean"]}[cond]
    return 100 * (none(m, f) - v)


def loss_b(task, m, f):
    if task == "dia":
        return loss_a("since", m, f)
    return 100 * (D["e13"][m][f"degree|{f}"]["ba"] - D["e18"][m]["degree"][f"128|{f}"]["ba"])


def bars(ax, groups, fn, labels):
    ax.set_facecolor(PANEL_BG)
    bw, step = 0.18, 0.20
    for gi, g in enumerate(groups):
        for fi, f in enumerate(F4):
            x = gi + (fi - 1.5) * step
            per = [fn(g, m, f) for m in ABL]
            mu = sum(per) / len(per)
            st = SER[f]
            ax.bar(x, mu, width=bw, color=st["color"], edgecolor="white", linewidth=0.8, zorder=3,
                   hatch="////" if f == "ECL" else None)
            ax.plot([x] * len(per), per, ls="none", marker="o", ms=2.6, mfc="white", mec=INK, mew=0.6, zorder=4)
    ax.axhline(0, color=QUIET, lw=0.9, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylabel(r"\textbf{Loss (points)}", fontsize=8, labelpad=1)
    style_axes(ax)


def main(out="figA_ctl.pdf"):
    plt.rcParams["hatch.color"] = "white"
    plt.rcParams["hatch.linewidth"] = 0.6
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    ax_a = fig.add_axes([0.1, BOT / H, 0.5, (TOP - BOT) / H])
    ax_b = fig.add_axes([0.7, BOT / H, 0.25, (TOP - BOT) / H])
    bars(ax_a, ["since", "entries", "edge", "random"], loss_a,
         [r"since heads", r"since heads," "\n" r"entries only", r"edge heads", r"random heads," "\n" r"edge layers"])
    bars(ax_b, ["dia", "deg"], loss_b, [r"$\Diamond$red", r"node degree" "\n" r"(GraphQA)"])
    ax_a.set_xlabel(r"\textbf{128 heads ablated} \textbf{(}$\boldsymbol{\Diamond}$\textbf{red)}", fontsize=8.5, labelpad=2)
    ax_b.set_xlabel(r"\textbf{Top 128 since heads}", fontsize=8.5, labelpad=2)
    ya = max(max(loss_a(c, m, f) for c in ("since", "entries", "edge", "random") for m in ABL for f in F4), 1)
    yb = max(max(loss_b(t, m, f) for t in ("dia", "deg") for m in ABL for f in F4), 1)
    lo = min(min(loss_a(c, m, f) for c in ("since", "entries", "edge", "random") for m in ABL for f in F4),
             min(loss_b(t, m, f) for t in ("dia", "deg") for m in ABL for f in F4), 0)
    for ax, y in ((ax_a, ya), (ax_b, yb)):
        ax.set_ylim(min(lo - 1.0, -1.5), 1.08 * y)
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
        wa = 0.64 * avail; wb = avail - wa
        xa = M + la
        xb = xa + wa + ra + GAP + lb
        ax_a.set_position([xa / W, BOT / H, wa / W, (TOP - BOT) / H])
        ax_b.set_position([xb / W, BOT / H, wb / W, (TOP - BOT) / H])
    fig.canvas.draw()
    ia, ib = inch(drawn_extent(ax_a, r)), inch(drawn_extent(ax_b, r))
    ly = (TOP + 0.20) / H
    hs = [Patch(facecolor=SER[f]["color"], edgecolor="white", hatch="////" if f == "ECL" else None) for f in F4]
    hs.append(plt.Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=INK, mew=0.6))
    fig.legend(hs, F4 + ["models"], loc="center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=((ia[0] + ib[2]) / 2 / W, ly), handletextpad=0.35, columnspacing=1.0, handlelength=1.3)
    top = (H - 0.02) / H
    fig.text(ia[0] / W, top, r"\textbf{(a)}", ha="left", va="top", fontsize=10, color=INK)
    fig.text(ib[0] / W, top, r"\textbf{(b)}", ha="left", va="top", fontsize=10, color=INK)
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print(f"margins L {ia[0]:.3f} R {W - ib[2]:.3f}; gap {ib[0] - ia[2]:.3f}")
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
