"""Figure (appendix, Section 5.4): causal and degree tests in LLMs.

(a) E11a and E17: balanced accuracy on 'v has a red neighbor' (D1a) by the number h of since heads
    mean-ablated (0 = none; 4 to 16: E11a; 32 to 128: E17), mean over Qwen2.5-7B, Llama-3.1-8B and Gemma-2-9B (lines)
    and range over the models (bands), per format; hollow markers at 128: 128 random heads of the same layers (mean
    over models of the 5-set means).
(b) E11b: depth one (D1a, D1c) on hub graphs by maximum degree Delta, mean over the four LLMs (lines) and range over
    models (bands), per format.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, style_axes

W, H = TEXTWIDTH_IN, 1.78
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from load_data import load_merged  # noqa: E402
D = load_merged()
MODELS = list(D["llm_main"]["models"])
F4 = ["EL", "AL", "NC", "ECL"]
DEG = [6, 12, 24, 48]
TOP, BOT, M, GAP = 1.30, 0.42, 0.07, 0.30
SHADES = {4: 0.35, 8: 0.65, 16: 1.0}


HS = [0, 4, 8, 16, 32, 64, 128]
ABL = {"Qwen2.5-7B": "e11a", "Llama-3.1-8B": "e11a_llama", "Gemma-2-9B": "e11a_gemma"}


def dose(m, f, h):
    A = D[ABL[m]]
    if h == 0:
        return A["none"][f"{f}|D1a"]["ba"]
    if h <= 16:
        return A["top"][f"{h}|{f}|D1a"]["ba"]
    return D["e17"][m]["top"][f"{h}|{f}|D1a"]["ba"]


def panel_a(ax):
    ax.set_facecolor(PANEL_BG)
    xs = list(range(len(HS)))
    for f in F4:
        st = SER[f]
        per = [[dose(m, f, h) for h in HS] for m in ABL]
        mu = [sum(p[i] for p in per) / len(per) for i in xs]
        lo = [min(p[i] for p in per) for i in xs]
        hi = [max(p[i] for p in per) for i in xs]
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.12, lw=0, zorder=2)
        ax.plot(xs, mu, color=st["color"], lw=1.7, ls=st.get("ls", "-"), marker=st["marker"],
                ms=4.6 if f != "ECL" else 4.0, mfc=st["color"] if f != "ECL" else "white",
                mec="white" if f != "ECL" else st["color"], mew=0.8 if f != "ECL" else 1.2, zorder=3)
        rnd = sum(D["e17"][m]["random"][f"128|{f}|D1a"]["mean"] for m in ABL) / len(ABL)
        ax.plot([xs[-1] + 0.32], [rnd], ls="none", marker=st["marker"], ms=4.4, mfc="white", mec=st["color"],
                mew=1.1, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(h) for h in HS], fontsize=8)
    ax.set_xlim(-0.25, len(HS) - 0.45)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8, labelpad=1)
    ax.set_xlabel(r"\textbf{Since heads ablated} $\boldsymbol{h}$", fontsize=8.5, labelpad=2)
    style_axes(ax)


def panel_b(ax):
    B = D["e11b"]
    ax.set_facecolor(PANEL_BG)
    xs = list(range(len(DEG)))
    for f in F4:
        st = SER[f]
        per = [[B[m][f"{f}|{d}"]["ba"] for d in DEG] for m in MODELS]
        mu = [sum(p[i] for p in per) / len(per) for i in xs]
        lo = [min(p[i] for p in per) for i in xs]
        hi = [max(p[i] for p in per) for i in xs]
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.12, lw=0, zorder=2)
        ax.plot(xs, mu, color=st["color"], lw=1.7, ls=st.get("ls", "-"), marker=st["marker"],
                ms=4.6 if f != "ECL" else 4.0, mfc=st["color"] if f != "ECL" else "white",
                mec="white" if f != "ECL" else st["color"], mew=0.8 if f != "ECL" else 1.2, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in DEG], fontsize=8)
    ax.set_xlim(-0.25, len(DEG) - 0.75)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8, labelpad=1)
    ax.set_xlabel(r"\textbf{Maximum degree} $\boldsymbol{\Delta}$ \textbf{(hub graphs)}", fontsize=8.5, labelpad=2)
    style_axes(ax)


def main(out="fig5.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    ax_a = fig.add_axes([0.1, BOT / H, 0.4, (TOP - BOT) / H])
    ax_b = fig.add_axes([0.6, BOT / H, 0.35, (TOP - BOT) / H])
    panel_a(ax_a); panel_b(ax_b)
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    inch = lambda bb: (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
    xa, wa, wb = 0.4, 2.8, 2.6
    for _ in range(5):
        xb = xa + wa + GAP + 0.3
        ax_a.set_position([xa / W, BOT / H, wa / W, (TOP - BOT) / H])
        ax_b.set_position([xb / W, BOT / H, wb / W, (TOP - BOT) / H])
        fig.canvas.draw()
        ia, ib = inch(drawn_extent(ax_a, r)), inch(drawn_extent(ax_b, r))
        la, ra, lb, rb = xa - ia[0], ia[2] - (xa + wa), xb - ib[0], ib[2] - (xb + wb)
        avail = W - 2 * M - la - ra - GAP - lb - rb
        wa = 0.5 * avail; wb = avail - wa
        xa = M + la
        xb = xa + wa + ra + GAP + lb
        ax_a.set_position([xa / W, BOT / H, wa / W, (TOP - BOT) / H])
        ax_b.set_position([xb / W, BOT / H, wb / W, (TOP - BOT) / H])
    fig.canvas.draw()
    ia, ib = inch(drawn_extent(ax_a, r)), inch(drawn_extent(ax_b, r))
    ly = (TOP + 0.22) / H
    ha = [Line2D([], [], ls="none", marker="o", ms=4.4, mfc="white", mec=INK, mew=1.1)]
    fig.legend(ha, ["128 random heads (same layers)"], loc="center", ncol=1, frameon=False,
               fontsize=7.5, bbox_to_anchor=((ia[0] + ia[2]) / 2 / W, ly), handletextpad=0.35, handlelength=1.2)
    hb = [Line2D([], [], color=SER[s]["color"], lw=1.7, ls=SER[s].get("ls", "-"), marker=SER[s]["marker"],
                 ms=4.6 if s != "ECL" else 4.0, mfc=SER[s]["color"] if s != "ECL" else "white",
                 mec="white" if s != "ECL" else SER[s]["color"], mew=0.8 if s != "ECL" else 1.2) for s in F4]
    fig.legend(hb, F4, loc="center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=((ib[0] + ib[2]) / 2 / W, ly),
               handletextpad=0.3, columnspacing=0.7, handlelength=1.9)
    top = (H - 0.02) / H
    fig.text(ia[0] / W, top, r"\textbf{(a)}", ha="left", va="top", fontsize=10, color=INK)
    fig.text(ib[0] / W, top, r"\textbf{(b)}", ha="left", va="top", fontsize=10, color=INK)
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print(f"margins L {ia[0]:.3f} R {W - ib[2]:.3f}; gap {ib[0] - ia[2]:.3f}")
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
