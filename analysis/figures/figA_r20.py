"""Figure (appendix): nesting versus distance. (a, b) E23 and E33, balanced accuracy on diamond red in
distribution by the number of training examples under 1-local+hard since and RoPE (mean of 5 seeds, bars: s.d.),
for EL, ELB (EL's two nested matches, colors near the edges), ECL and the better list format; (c) E33c, depth one
in the four LLMs (mean of D1a to D1c; the E1 protocol, 400 pairs per question, three templates) under EL, ELB and ECL.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, style_axes

W, H = TEXTWIDTH_IN, 2.45
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from load_data import load_merged  # noqa: E402
D = load_merged()
NS = (25000, 50000, 100000, 200000, 500000)
M4 = ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B", "Qwen2.5-72B")
SHORT = {"Qwen2.5-7B": "Qwen\n7B", "Llama-3.1-8B": "Llama\n8B", "Gemma-2-9B": "Gemma\n9B", "Qwen2.5-72B": "Qwen\n72B"}
TOP, BOT, M, GAP = 1.78, 0.52, 0.07, 0.34


def curve(arm, f):
    if f == "ELB":
        c = [D["e33"]["cells"][f"{arm}|ELB|{N}|id"] for N in NS]
    elif f == "list":
        best = max(("AL", "NC"), key=lambda g: D["e23"]["n95"][f"{arm}|{g}"] and -D["e23"]["n95"][f"{arm}|{g}"])
        c = [D["e23"]["cells"][f"{arm}|{best}|{N}|id"] for N in NS]
    else:
        c = [D["e23"]["cells"][f"{arm}|{f}|{N}|id"] for N in NS]
    return [x["mean"] for x in c], [x["sd"] for x in c]


def panel(ax, arm, title):
    ax.set_facecolor(PANEL_BG)
    for f, lab in (("list", "AL"), ("ECL", "ECL"), ("ELB", "ELB"), ("EL", "EL")):
        mu, sd = curve(arm, f)
        st = SER[lab]
        ax.errorbar(NS, mu, yerr=sd, color=st["color"], marker=st["marker"], ms=3.6, lw=1.4, ls=st.get("ls", "-"),
                    capsize=1.6, elinewidth=0.8, zorder=3 if f != "ELB" else 4)
    ax.axhline(0.95, color=QUIET, lw=0.8, ls=(0, (2, 2)), zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels(["25k", "50k", "100k", "200k", "500k"], fontsize=8)
    ax.minorticks_off()
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel(rf"\textbf{{{title}}}", fontsize=8.5, labelpad=2)
    style_axes(ax)


def llm(ax):
    ax.set_facecolor(PANEL_BG)
    pq = D["llm_main"]["per_question"]
    for gi, m in enumerate(M4):
        vals = {"EL": sum(pq[m][q]["EL"] for q in ("D1a", "D1b", "D1c")) / 3,
                "ELB": sum(D["e33"]["llm"][m][q]["ba"] for q in ("D1a", "D1b", "D1c")) / 3,
                "ECL": D["llm_formats"][m]["ECL"]["depth1"]["ba"]}
        for fi, f in enumerate(("EL", "ELB", "ECL")):
            ax.bar(gi + (fi - 1) * 0.25, vals[f] - 0.5, bottom=0.5, width=0.23, color=SER[f]["color"], edgecolor="white",
                   linewidth=0.8, hatch="////" if f == "ECL" else ("...." if f == "ELB" else None), zorder=3)
    ax.set_xticks(range(len(M4)))
    ax.set_xticklabels([SHORT[m] for m in M4], fontsize=8, linespacing=0.9)
    ax.set_xlim(-0.55, len(M4) - 0.45)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel(r"\textbf{E33c: LLMs, depth one}", fontsize=8.5, labelpad=2)
    style_axes(ax)


def main(out="figA_r20.pdf"):
    plt.rcParams["hatch.color"] = "white"
    plt.rcParams["hatch.linewidth"] = 0.6
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    axs = [fig.add_axes([0.1, BOT / H, 0.2, (TOP - BOT) / H]) for _ in range(3)]
    panel(axs[0], "1-local+hard-since", "E23, E33: 1-local+hard since")
    panel(axs[1], "RoPE", "E23, E33: RoPE")
    llm(axs[2])
    axs[0].set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8.5, labelpad=1)
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    inch = lambda bb: (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
    share = (0.34, 0.34, 0.32)
    ws, xs = [1.8, 1.8, 1.5], [0.4, 2.6, 4.6]
    for _ in range(5):
        for ax, x, w in zip(axs, xs, ws):
            ax.set_position([x / W, BOT / H, w / W, (TOP - BOT) / H])
        fig.canvas.draw()
        ext = [inch(drawn_extent(ax, r)) for ax in axs]
        pads = [(x - e[0], e[2] - (x + w)) for ax, x, w, e in zip(axs, xs, ws, ext)]
        avail = W - 2 * M - sum(l + rr for l, rr in pads) - 2 * GAP
        ws = [s * avail for s in share]
        xs, cur = [], M
        for (l, rr), w in zip(pads, ws):
            xs.append(cur + l)
            cur += l + w + rr + GAP
    for ax, x, w in zip(axs, xs, ws):
        ax.set_position([x / W, BOT / H, w / W, (TOP - BOT) / H])
    fig.canvas.draw()
    ext = [inch(drawn_extent(ax, r)) for ax in axs]
    hs = [Line2D([], [], color=SER[f]["color"], marker=SER[f]["marker"], ms=3.6, lw=1.4, ls=SER[f].get("ls", "-"))
          for f in ("EL", "ELB", "ECL", "AL")]
    hs += [Patch(facecolor=SER["ELB"]["color"], edgecolor="white", hatch="....")]
    labs = ["EL: two nested matches, colors far", "ELB: two nested matches, colors near edges", "ECL: one match",
            "better list format", "ELB bars"]
    fig.legend(hs[:4], labs[:4], loc="center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=((ext[0][0] + ext[2][2]) / 2 / W, (TOP + 0.30) / H), handletextpad=0.35,
               columnspacing=0.9, handlelength=1.7)
    top = (H - 0.02) / H
    for ax_e, lab in zip(ext, ("(a)", "(b)", "(c)")):
        fig.text(ax_e[0] / W, top, rf"\textbf{{{lab}}}", ha="left", va="top", fontsize=10, color=INK)
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()
