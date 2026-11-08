"""Figure (appendix): (a) E19, loss in points of balanced accuracy on 'v has a red neighbor' (D1a) under the
knockout of the entries' attention to their own line head (every head), the control knockout (previous line head),
recolor patching of the top-128 since heads at the entries, and their ablation at the entries (E18); (b) E20 with
E14, the cost of moving the needed tokens apart at equal prompt length: filler between line head and entries minus
filler after the entries (AL, NC), and filler between color line and edges minus filler before the color line (EL,
ECL); (c) E21, relational tables, balanced accuracy by format on the depth-one and the global question. Bars: mean
over the models (three in a and b, four in c); dots: models.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch

from style import INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, drawn_extent, style_axes

W, H = TEXTWIDTH_IN, 2.25
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from load_data import load_merged  # noqa: E402
D = load_merged()
M3 = {"Qwen2.5-7B": "e11a", "Llama-3.1-8B": "e11a_llama", "Gemma-2-9B": "e11a_gemma"}
M4 = [m for m in ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B", "Qwen2.5-72B") if m in D["e21"]]
TOP, BOT, M, GAP = 1.72, 0.58, 0.07, 0.30
TAB = {"EL": "rows", "AL": "person", "NC": "project", "ECL": "inline"}


def none(m, f, q="D1a"):
    return D[M3[m]]["none"][f"{f}|{q}"]["ba"]


def loss_a(c, m, f):
    v = {"own": D["e19"][m]["ko"][f"own|{f}|D1a"]["ba"], "other": D["e19"][m]["ko"][f"other|{f}|D1a"]["ba"],
         "patch": D["e19"][m]["patch"][f"recolor128|{f}|D1a"]["ba"], "abl": D["e18"][m]["entry"][f"128|{f}|D1a"]["ba"]}[c]
    return 100 * (none(m, f) - v)


def cost_b(c, m, f):
    if f in ("AL", "NC"):
        return -100 * D["e20"][m]["contrast"][f"{f}|{c}"]["diff"]
    return -100 * D["e14"][m][f"{f}|contrast|{c}"]


def bars(ax, groups, fmts, fn, models, labels, base=0.0, step=0.2, bw=0.18):
    ax.set_facecolor(PANEL_BG)
    for gi, (g, fs) in enumerate(zip(groups, fmts)):
        for fi, f in enumerate(fs):
            x = gi + (fi - (len(fs) - 1) / 2) * step
            per = [fn(g, m, f) for m in models]
            mu = sum(per) / len(per)
            st = SER[f]
            ax.bar(x, mu - base, bottom=base, width=bw, color=st["color"], edgecolor="white", linewidth=0.8, zorder=3,
                   hatch="////" if f == "ECL" else None)
            ax.plot([x] * len(per), per, ls="none", marker="o", ms=2.6, mfc="white", mec=INK, mew=0.6, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8, linespacing=0.95)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    style_axes(ax)


def main(out="figA_r19.pdf"):
    plt.rcParams["hatch.color"] = "white"
    plt.rcParams["hatch.linewidth"] = 0.6
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    axs = [fig.add_axes([0.1, BOT / H, 0.2, (TOP - BOT) / H]) for _ in range(3)]
    A = ["own", "other", "patch", "abl"]
    bars(axs[0], A, [("AL", "NC")] * 4, loss_a, list(M3), [r"own" "\n" r"head", r"previous" "\n" r"head",
                                                             r"patch" "\n" r"entries", r"ablate" "\n" r"entries"], step=0.22, bw=0.2)
    axs[0].axhline(0, color=QUIET, lw=0.9, zorder=2)
    axs[0].set_ylabel(r"\textbf{Loss (points)}", fontsize=8, labelpad=1)
    axs[0].set_xlabel(r"\textbf{E19: knockout and entry interventions}", fontsize=8.5, labelpad=2)
    Bg = [8, 24, 64, 256]
    bars(axs[1], Bg, [("AL", "NC"), ("AL", "NC"), ("EL", "ECL"), ("EL", "ECL")], cost_b, list(M3),
         [r"lists" "\n" r"$n_{\mathrm f}{=}8$", r"lists" "\n" r"$n_{\mathrm f}{=}24$", r"edges" "\n" r"$n_{\mathrm f}{=}64$", r"edges" "\n" r"$n_{\mathrm f}{=}256$"], step=0.22, bw=0.2)
    axs[1].axhline(0, color=QUIET, lw=0.9, zorder=2)
    axs[1].set_ylabel(r"\textbf{Cost (points)}", fontsize=8, labelpad=1)
    axs[1].set_xlabel(r"\textbf{E20 (lists) and E14 (edges)}", fontsize=8.5, labelpad=2)
    fn_c = lambda q, m, f: D["e21"][m][f"{TAB[f]}|{q}"]["ba"]
    bars(axs[2], ["T1", "G1"], [("EL", "AL", "NC", "ECL")] * 2, fn_c, M4, [r"depth one", r"global"], base=0.5, step=0.19, bw=0.17)
    axs[2].set_ylim(0.5, 1.0)
    axs[2].set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8, labelpad=1)
    axs[2].set_xlabel(r"\textbf{E21, relational tables}", fontsize=8.5, labelpad=2)
    for ax, fn, keys in ((axs[0], loss_a, [(c, f) for c in A for f in ("AL", "NC")]),
                         (axs[1], cost_b, [(c, f) for c in (8, 24) for f in ("AL", "NC")] + [(c, f) for c in (64, 256) for f in ("EL", "ECL")])):
        vals = [fn(c, m, f) for c, f in keys for m in M3]
        ax.set_ylim(min(min(vals) - 1.0, -1.5), 1.1 * max(vals))
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    inch = lambda bb: (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
    share = (0.38, 0.32, 0.30)
    ws = [1.8, 1.5, 1.8]
    xs = [0.4, 2.6, 4.5]
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
    ly = (TOP + 0.22) / H
    F4 = ["EL", "AL", "NC", "ECL"]
    hs = [Patch(facecolor=SER[f]["color"], edgecolor="white", hatch="////" if f == "ECL" else None) for f in F4]
    hs.append(plt.Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=INK, mew=0.6))
    labs = [rf"{f} / {TAB[f]}" for f in F4] + ["models"]
    fig.legend(hs, labs, loc="center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=((ext[0][0] + ext[2][2]) / 2 / W, ly), handletextpad=0.35, columnspacing=1.0, handlelength=1.3)
    top = (H - 0.02) / H
    for ax_e, lab in zip(ext, ("(a)", "(b)", "(c)")):
        fig.text(ax_e[0] / W, top, rf"\textbf{{{lab}}}", ha="left", va="top", fontsize=10, color=INK)
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print(f"margins L {ext[0][0]:.3f} R {W - ext[2][2]:.3f}")
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
