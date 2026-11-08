"""Appendix figure: real topologies (E5).
(a) Distribution of k_min (position of the query vertex in its red neighbors' lists, positives) for molecules,
    citation subgraphs and the random graphs of E1 (Qwen2.5-7B order draws).
(b) Depth-one balanced accuracy (D1a) per serialization on molecules and citation subgraphs, Qwen2.5-7B (solid)
    and Llama-3.1-8B (light); the short black tick on each bar is the prediction from the random-graph k_min
    strata weighted by the source's k_min distribution (validation rule C11), drawn where the strata exist.
"""
import json
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from fmt import fstr, lr_round  # noqa: E402
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, style_axes

W, H = TEXTWIDTH_IN, 1.82
from load_data import load_merged  # noqa: E402
P = load_merged()
RG, MAIN, GML = P["real_graphs"], P["llm_main"], P["graphml"]
KS = ["1", "2", "3-4", ">=5"]
KLAB = ["1", "2", "3--4", r"$\geq$5"]
SHADE = ["#1F3A5F", "#4E6A8F", "#A9B8CC", "#E3E8EF"]
FM = ["EL", "AL", "NC", "ECL", "GraphML"]
GCOL = "#8A939F"
MODELS = ["Qwen2.5-7B", "Llama-3.1-8B"]
SRC = [("molecules", "Molecules"), ("citation", "Citation")]


def color(f):
    return SER[f]["color"] if f in SER else GCOL


def pred(m, src, f):
    sh = RG["stats"][src]["kmin_share"]
    if f == "GraphML":                 # GraphML: its own random-graph strata, same model (Table T_real)
        ref = GML["by_model"][m]["strata_d1a"]["kmin"]
    else:
        ref = MAIN["strata_d1a"][m][f]["kmin"]
    ks = [k for k in sh if ref.get(k) is not None]
    return sum(sh[k] * ref[k] for k in ks) / sum(sh[k] for k in ks)


def main(out="figA_real.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    bot, top, M, G = 0.32, 1.35, 0.07, 0.30
    a = fig.add_axes([0.6 / W, bot / H, 1.2 / W, (top - bot) / H])
    bs = [fig.add_axes([(2.3 + i * 2.0) / W, bot / H, 1.8 / W, (top - bot) / H]) for i in range(2)]
    rows = [("molecules", "Molecules", RG["stats"]["molecules"]["kmin_share"]),
            ("citation", "Citation", RG["stats"]["citation"]["kmin_share"]),
            ("random", r"Random (E1)", MAIN["strata"]["Qwen2.5-7B"]["NC"]["kmin_share"])]
    for i, (_, lab, sh) in enumerate(rows):
        left = 0.0
        shown = lr_round([sh[kk] for kk in KS], 3)     # three decimals, sum 1.000 (same rule as Tables T_real, T_kstrata)
        for k, kk in enumerate(KS):
            v = sh[kk]
            a.barh(i, v, left=left, height=0.62, color=SHADE[k], ec="white", lw=0.8, zorder=2)
            if v >= 0.17:                    # wide segment: label inside
                a.text(left + v / 2, i, fstr(shown[k], 3)[1:], ha="center", va="center", fontsize=6.3,
                       color="white" if k < 2 else INK, zorder=3)
            elif v >= 0.02:                  # narrow segment: label just above the bar, never past its edges
                a.text(min(left + v / 2, 1 - 0.045), i - 0.33, fstr(shown[k], 3)[1:], ha="center", va="bottom",
                       fontsize=6.0, color=INK, zorder=3)
            left += v
    a.set_yticks(range(3)); a.set_yticklabels([lab for _, lab, _ in rows], fontsize=8)
    a.invert_yaxis(); a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1.0]); a.set_xticklabels(["0", ".5", "1"], fontsize=8)
    a.set_xlabel(r"\textbf{Share of positives}", fontsize=8.5, labelpad=1)
    a.tick_params(axis="y", length=0)
    style_axes(a)
    for ax, (src, lab) in zip(bs, SRC):
        ax.set_facecolor(PANEL_BG)
        ax.axhline(0.5, color=QUIET, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for j, f in enumerate(FM):
            for k, m in enumerate(MODELS):
                c = RG["results"][m][src][f]
                x = j + (k - 0.5) * 0.38
                ax.bar(x, c["ba"] - 0.4, bottom=0.4, width=0.34, color=color(f), alpha=1.0 if k == 0 else 0.45,
                       ec=color(f), lw=0.8, zorder=2)
                ax.errorbar([x], [c["ba"]], yerr=[(c["hi"] - c["lo"]) / 2], color=QUIET, lw=0, elinewidth=0.8,
                            capsize=0, zorder=4)
                p = pred(m, src, f)
                if p is not None:
                    ax.plot([x - 0.19, x + 0.19], [p, p], color=INK, lw=1.6, solid_capstyle="butt", zorder=5)
        ax.set_xticks(range(len(FM))); ax.set_xticklabels(["EL", "AL", "NC", "ECL", "Graph-\nML"], fontsize=7.5, linespacing=0.9)
        for t, f in zip(ax.get_xticklabels(), FM):
            t.set_color(color(f))
        ax.set_xlim(-0.6, len(FM) - 0.4); ax.set_ylim(0.4, 1.0)
        ax.set_title(rf"\textbf{{{lab}}}", fontsize=8.5, pad=3)
        ax.tick_params(axis="x", length=0)
        style_axes(ax)
    bs[0].set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8.5, labelpad=2)
    bs[1].tick_params(labelleft=False)
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    axs = [a] + bs
    gaps = [0.30, 0.12]
    ratio = [1.0, 1.35, 1.35]
    for _ in range(4):
        fig.canvas.draw()
        ink = [drawn_extent(ax, r) for ax in axs]
        pos = [ax.get_position() for ax in axs]
        le = [p.x0 * W - i.x0 / dpi for p, i in zip(pos, ink)]
        re = [i.x1 / dpi - p.x1 * W for p, i in zip(pos, ink)]
        avail = W - 2 * M - sum(gaps) - sum(le) - sum(re)
        ws = [avail * q / sum(ratio) for q in ratio]
        x = M
        for k, ax in enumerate(axs):
            x += le[k]
            ax.set_position([x / W, bot / H, ws[k] / W, (top - bot) / H])
            x += ws[k] + re[k] + (gaps[k] if k < 2 else 0)
    fig.canvas.draw()
    ink = [drawn_extent(ax, r) for ax in axs]
    hs = [Patch(fc=s, ec="white") for s in SHADE]
    fig.legend(hs, [rf"$k_{{\min}}$ {l}" for l in KLAB], loc="center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=((ink[0].x0 + ink[0].x1) / 2 / dpi / W, (top + 0.22) / H), handlelength=1.0,
               handletextpad=0.3, columnspacing=0.5)
    hb = [Patch(fc=QUIET, ec=QUIET), Patch(fc=QUIET, ec=QUIET, alpha=0.45), plt.Line2D([], [], color=INK, lw=1.4)]
    fig.legend(hb, ["Qwen2.5-7B", "Llama-3.1-8B", "prediction"], loc="center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=((ink[1].x0 + ink[2].x1) / 2 / dpi / W, (top + 0.30) / H), handlelength=1.4,
               handletextpad=0.3, columnspacing=0.8)
    fig.text(ink[0].x0 / dpi / W, (H - 0.02) / H, r"\textbf{(a)}", ha="left", va="top", fontsize=10, color=INK)
    fig.text(ink[1].x0 / dpi / W, (H - 0.02) / H, r"\textbf{(b)}", ha="left", va="top", fontsize=10, color=INK)
    fig.canvas.draw()
    print("ink", [(round(i.x0 / dpi, 3), round(i.x1 / dpi, 3)) for i in ink])
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
