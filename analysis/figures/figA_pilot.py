"""Appendix figure: pilot scaling (E0b). X1 ('some red vertex has a red neighbor') under NC, name-blind,
balanced accuracy by model width, for 3 and 6 layers (facets), 4k or 16k training pairs (line style) and 8k or
32k steps (marker fill); mean and s.d. over 3 seeds. The base configuration (width 64, 3 layers, 4k pairs, 8k
steps) is the measured pilot cell of Table 1 (hollow triangle labeled "measured")."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D

from style import COLUMN_IN, INK, PANEL_BG, ACCENT, QUIET, SER, drawn_extent, style_axes

W, H = COLUMN_IN, 1.85
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import load_merged  # noqa: E402
D = load_merged()["pilot_scaling"]
WID = [64, 128, 256]
C = SER["NC"]["color"]
LS = {"4k": (0, (4, 1.6)), "16k": "-"}


def main(out="figA_pilot.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    M, G, bot, top = 0.04, 0.10, 0.36, 1.42
    axs = [fig.add_axes([0.4 / W, bot / H, 1.2 / W, (top - bot) / H]), fig.add_axes([1.7 / W, bot / H, 1.2 / W, (top - bot) / H])]
    for ax, L in zip(axs, (3, 6)):
        ax.set_facecolor(PANEL_BG)
        ax.axhline(0.5, color=QUIET, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for dk in ("4k", "16k"):
            for sk in ("8k", "32k"):
                cs = [D[f"{w}|{L}|{dk}|{sk}"] for w in WID]
                off = {("4k", "8k"): -0.09, ("4k", "32k"): -0.03, ("16k", "8k"): 0.03, ("16k", "32k"): 0.09}[(dk, sk)]
                ax.errorbar([x + off for x in range(3)], [c["mean"] for c in cs], yerr=[c["sd"] for c in cs], color=C, ls=LS[dk], lw=1.4,
                            marker="^", ms=4.2, mfc=C if sk == "32k" else "white", mec=C, mew=1.0, elinewidth=1.0,
                            ecolor=INK, capsize=2.4, capthick=1.0, zorder=3)
        if L == 3:
            b = D["64|3|4k|8k"]
            # measured base cell: 8k steps, so hollow like its legend entry, drawn in black
            ax.plot([-0.09], [b["mean"]], ls="none", marker="^", ms=5.4, mfc="white", mec=INK, mew=1.3, zorder=5)
            ax.annotate(r"base", xy=(-0.09, b["mean"]), xytext=(0.32, 0.60), fontsize=7.5, color=INK, ha="left",
                        va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=1, shrinkB=3.5))
        ax.set_xticks(range(3)); ax.set_xticklabels([str(w) for w in WID], fontsize=8)
        ax.set_xlim(-0.3, 2.3); ax.set_ylim(0.45, 1.0)
        ax.set_title(rf"\textbf{{{L} layers}}", fontsize=8.5, pad=3)
        style_axes(ax)
    axs[0].set_ylabel(r"\textbf{Balanced accuracy}", fontsize=8.5, labelpad=2)
    axs[1].tick_params(labelleft=False)
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    for _ in range(3):
        fig.canvas.draw()
        i0, i1 = drawn_extent(axs[0], r), drawn_extent(axs[1], r)
        p0, p1 = axs[0].get_position(), axs[1].get_position()
        le0 = p0.x0 * W - i0.x0 / dpi; re0 = i0.x1 / dpi - p0.x1 * W
        le1 = p1.x0 * W - i1.x0 / dpi; re1 = i1.x1 / dpi - p1.x1 * W
        w = (W - 2 * M - G - le0 - re0 - le1 - re1) / 2
        axs[0].set_position([(M + le0) / W, bot / H, w / W, (top - bot) / H])
        axs[1].set_position([(M + le0 + w + re0 + G + le1) / W, bot / H, w / W, (top - bot) / H])
    fig.text((axs[0].get_position().x0 + axs[1].get_position().x1) / 2, 0.03 / H,
             r"\textbf{Width} $\boldsymbol{d_{\mathrm{model}}}$", ha="center", va="bottom", fontsize=8.5)
    hs = [Line2D([], [], color=C, ls=LS["4k"], lw=1.4), Line2D([], [], color=C, ls=LS["16k"], lw=1.4),
          Line2D([], [], ls="none", marker="^", ms=4.2, mfc="white", mec=C, mew=1.0),
          Line2D([], [], ls="none", marker="^", ms=4.2, mfc=C, mec=C)]
    fig.legend(hs, ["4k pairs", "16k pairs", "8k steps", "32k steps"], loc="center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, (H - 0.10) / H), handlelength=1.6, handletextpad=0.25,
               columnspacing=0.55)
    fig.canvas.draw()
    print("ink", [(round(drawn_extent(a, r).x0 / dpi, 3), round(drawn_extent(a, r).x1 / dpi, 3)) for a in axs])
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out, f"{W:.3f} x {H:.3f} in")


if __name__ == "__main__":
    main()
