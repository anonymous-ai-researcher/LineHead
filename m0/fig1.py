"""Figure 1: (a) one graph, three serializations, three logics; (b) M0 accuracy.

Fonts: Times text (`times`, as in the ACL template), Times-matched math
(`newtxmath`, also loaded in the paper preamble), Inconsolata for code; set
through LaTeX so the figure matches the paper exactly.
Size is the full text width of the ACL two-column layout (16.0 cm).

Usage:
  python fig1.py --summary results/summary.json --out figures/fig1.pdf
If the summary comes from the mock model, panel (b) is stamped as a layout
preview so it cannot be mistaken for results.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

TEXTWIDTH_IN = 16.0 / 2.54

plt.rcParams.update({
    "text.usetex": True,
    # Text: Times via `times` (Nimbus Roman, identical to the ACL body text).
    # Math: newtxmath, whose glyphs are designed to match Times; Computer Modern
    # math is lighter and narrower and looks mismatched next to Times.
    # Code: Inconsolata, as \texttt in the ACL template. (newtxmath supplies the
    # amssymb symbols, so amssymb must not be loaded as well.)
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}"
                           r"\usepackage{newtxmath}\usepackage{inconsolata}",
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42,
})

# Serialization palette: muted, colorblind-safe, and distinct from the task's own
# red/blue node colors. Each serialization also has its own marker shape.
SER = {
    "EL": {"color": "#6C4FA3", "marker": "o", "name": "Edge list"},
    "AL": {"color": "#1F8A84", "marker": "s", "name": "Adjacency list"},
    "NC": {"color": "#D98B1E", "marker": "^", "name": "Node-centric"},
}
NODE = {"red": "#D1495B", "blue": "#3D6FB6"}
INK, QUIET, PANEL_BG = "#1F2328", "#5A6270", "#F6F7F9"
CLASS_LABEL = {"global": r"Global\\[-1pt]\footnotesize(PL\,+\,GC)",
               "depth1": r"Depth 1\\[-1pt]\footnotesize($\Diamond\varphi$)",
               "depth2": r"Depth 2\\[-1pt]\footnotesize($\Diamond\Diamond\varphi$)",
               "count": r"Counting\\[-1pt]\footnotesize($\Diamond_{\geq k}\varphi$)"}


def arrow(ax, p, q, color=QUIET, lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def box(ax, x, y, w, h, edge, face="white", lw=1.2, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=face, ec=edge, lw=lw, zorder=1))


PA_W, PA_H = 3.80, 2.85          # panel (a) size in inches; drawn in inch units
# Panel (a) geometry, in panel-(a) inches. Panel (b) is aligned to these lines in main().
A_TITLE_Y = 2.66                 # column-title row (center)
A_TOP, A_BH, A_GAP = 2.50, 0.60, 0.13   # top of first box, box height, vertical gap
A_LX, A_LW = 2.90, 0.88          # logic boxes ("depth <= f_NC(L)" at 7.6 pt is ~0.76 in)
A_NOTE_Y = -0.06                 # footer note (center)
A_SX, A_SW = 1.24, 1.33          # serialization boxes (<= 23 chars of 7.3 pt mono = 1.17 in)
ARROW_GAP = 0.04                 # clearance between an arrow tip/tail and a box
ARROW_LEN = A_LX - (A_SX + A_SW) - 2 * ARROW_GAP   # every horizontal arrow has this length


def panel_a(ax):
    """Drawn in inch coordinates so circles stay round and text widths can be budgeted.
    Inconsolata at 6.5 pt is 3.25 pt per character: 29 characters fit in a 1.50 in box."""
    ax.set_xlim(0, PA_W); ax.set_ylim(0, PA_H); ax.set_aspect("equal"); ax.axis("off")
    title_y = A_TITLE_Y
    # column layout (inches)
    gx0, gx1 = 0.02, 1.00            # graph
    sx, sw = A_SX, A_SW
    lx, lw_ = A_LX, A_LW
    bh, gap, top = A_BH, A_GAP, A_TOP
    hh = 0.165                       # header bar height
    ys = {s: top - bh - i * (bh + gap) for i, s in enumerate(["EL", "AL", "NC"])}

    for x, t in [(0.53, "Same graph"), (sx + sw / 2, "Serialized as text"),
                 (lx + lw_ / 2, "Decidable logic")]:
        ax.text(x, title_y, rf"\textbf{{{t}}}", ha="center", va="center", fontsize=9, color=INK)

    # --- the graph ---------------------------------------------------------
    pos = {"ash": (0.49, 2.12), "elm": (0.18, 1.62), "fig": (0.80, 1.62),
           "oak": (0.18, 1.02), "bay": (0.80, 1.02), "yew": (0.49, 0.55)}
    color = {"ash": "blue", "elm": "blue", "fig": "blue", "oak": "red", "bay": "blue", "yew": "red"}
    for a, b in [("ash", "elm"), ("ash", "fig"), ("elm", "oak"), ("fig", "bay"),
                 ("bay", "yew"), ("oak", "yew")]:
        ax.plot(*zip(pos[a], pos[b]), color="#A3AAB5", lw=1.9, zorder=1, solid_capstyle="round")
    r = 0.112
    for n, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), r, fc=NODE[color[n]], ec="white", lw=1.3, zorder=3))
        ax.text(x, y - 0.005, rf"\textbf{{{n}}}", color="white", ha="center", va="center",
                fontsize=7.2, zorder=4)
    ax.add_patch(Circle(pos["ash"], r + 0.05, fc="none", ec=INK, lw=1.0,
                        ls=(0, (2.2, 1.6)), zorder=3))
    ax.text(0.49, 2.42, r"query node", ha="center", va="center", fontsize=7.6, color=QUIET)
    ax.text(0.49, 0.27, r"Q: $\Diamond\Diamond\mathit{red}$ at \texttt{ash}?",
            ha="center", va="center", fontsize=8, color=INK)
    # node-color key
    for i, (c, lab) in enumerate([("red", "red"), ("blue", "blue")]):
        ax.add_patch(Circle((0.22 + i * 0.46, 0.08), 0.04, fc=NODE[c], ec="none"))
        ax.text(0.29 + i * 0.46, 0.08, lab, ha="left", va="center", fontsize=7.6, color=QUIET)

    # --- serializations and logics ----------------------------------------
    snippets = {
        "EL": [r"Colors: ash blue, ...", r"Edges:", r"ash -{}- elm \ ..."],
        "AL": [r"ash (blue): elm, fig", r"elm (blue): ash, oak", r"oak (red): elm, yew ..."],
        "NC": [r"ash is blue. Neighbors:", r"elm (blue), fig (blue).", r"elm is blue. ..."],
    }
    # bracket connector: graph -> vertical spine -> one straight arrow per serialization
    spine_x = sx - ARROW_GAP - ARROW_LEN     # bracket arrows as long as the logic arrows
    mids = [ys[s] + bh / 2 for s in ys]
    ax.plot([0.93, spine_x], [1.40, 1.40], color=QUIET, lw=1.7, solid_capstyle="round")
    ax.plot([spine_x, spine_x], [min(mids), max(mids)], color=QUIET, lw=1.7, solid_capstyle="round")
    for s, y in ys.items():
        c = SER[s]["color"]
        box(ax, sx, y, sw, bh, edge=c, lw=1.3, r=0.05)
        ax.add_patch(FancyBboxPatch((sx, y + bh - hh), sw, hh,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=c, ec=c, lw=1.3, zorder=1))
        ax.add_patch(plt.Rectangle((sx, y + bh - hh), sw, hh / 2, fc=c, ec="none", zorder=1))
        ax.text(sx + 0.06, y + bh - hh / 2, rf"\textbf{{{SER[s]['name']} ({s})}}",
                color="white", ha="left", va="center", fontsize=7.6, zorder=2)
        body_top = y + bh - hh - 0.075
        for k, line in enumerate(snippets[s]):
            ax.text(sx + 0.07, body_top - k * 0.133, rf"\texttt{{{line}}}", ha="left",
                    va="center", fontsize=7.3, color=INK, zorder=2)
        box(ax, lx, y + 0.07, lw_, bh - 0.14, edge=c, face=c + "1F", lw=1.3, r=0.05)
        ax.text(lx + lw_ / 2, y + bh / 2 + 0.085, r"GML\,+\,GC", ha="center", va="center",
                fontsize=7.6, color=INK, zorder=2)
        ax.text(lx + lw_ / 2, y + bh / 2 - 0.085, rf"depth $\leq f_{{\mathrm{{{s}}}}}\mkern-1mu(L)$",
                ha="center", va="center", fontsize=7.6, color=INK, zorder=2)
        arrow(ax, (spine_x, y + bh / 2), (sx - ARROW_GAP, y + bh / 2))
        arrow(ax, (sx + sw + ARROW_GAP, y + bh / 2), (lx - ARROW_GAP, y + bh / 2), color=c, lw=1.9)
    ax.text(sx + (lx + lw_ - sx) / 2, A_NOTE_Y,
            r"\textit{ID-blind names: no query-node property is decidable}",
            ha="center", va="center", fontsize=7.8, color=QUIET)


def panel_b(ax, summary, model=None):
    model = model or next(iter(summary))
    s = summary[model]
    cells = {(c["cls"], c["ser"]): c for c in s["cells"]}
    classes = ["global", "depth1", "depth2", "count"]
    off = {"EL": -0.22, "AL": 0.0, "NC": 0.22}
    ax.set_facecolor(PANEL_BG)
    for i in range(len(classes)):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, color="white", zorder=0)
    ax.axhline(0.5, color=QUIET, lw=1.4, ls=(0, (4, 3)), zorder=1, label="chance")
    for ser, st in SER.items():
        xs, ys, lo, hi = [], [], [], []
        for i, cls in enumerate(classes):
            c = cells.get((cls, ser))
            if c:
                xs.append(i + off[ser]); ys.append(c["bal_acc"])
                lo.append(c["bal_acc"] - c["ci_low"]); hi.append(c["ci_high"] - c["bal_acc"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=st["marker"], ms=7.0, mfc=st["color"], mec="white",
                    mew=1.0, ecolor=st["color"], elinewidth=1.9, capsize=3.0, capthick=1.7,
                    label=ser, zorder=3)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([rf"\shortstack{{{CLASS_LABEL[c]}}}" for c in classes], fontsize=8)
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel(r"\textbf{Balanced accuracy}", fontsize=9)
    ax.set_xlabel(r"\textbf{Logic the question requires}", fontsize=9)
    ax.xaxis.label.set_va("center")
    ax.tick_params(axis="y", labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if s.get("mock"):
        ax.text(0.5, 0.80, r"\shortstack{\textbf{LAYOUT PREVIEW}\\[1pt]\scriptsize random-guess mock model, not results}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#B42318", alpha=0.85, zorder=5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#B42318", lw=1.0, alpha=0.9))


def legend_handles():
    """One-row legend: one marker per serialization (named in full in panel (a)), then chance."""
    from matplotlib.lines import Line2D
    hs = []
    hs += [Line2D([], [], ls="none", marker=st["marker"], ms=7.0, mfc=st["color"], mec="white",
                  mew=1.0) for st in SER.values()]
    hs.append(Line2D([], [], color=QUIET, lw=1.4, ls=(0, (4, 3))))
    labels = list(SER) + ["chance"]
    return hs, labels


def drawn_extent(ax, r):
    """Union of what is actually drawn (text, patches, lines, tick labels, axis labels, spines);
    unlike get_tightbbox it ignores the invisible axes rectangle of panel (a)."""
    from matplotlib.transforms import Bbox
    arts = list(ax.texts) + list(ax.patches) + list(ax.lines) + list(ax.collections)
    if ax.axison:
        arts += [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        arts += [sp for sp in ax.spines.values() if sp.get_visible()]
    bbs = [a.get_window_extent(r) for a in arts if a.get_visible()]
    bbs = [b for b in bbs if b.width > 0 or b.height > 0]
    return Bbox.union(bbs)


def layout(fig, ax_a, ax_b, H, a_off, gap_ab=0.30, iters=4):
    """Place panel (b) by rule, not by eye:
      plot-area top    = top of the EL box in (a)
      plot-area bottom = bottom of the NC box in (a)
      x-label center   = center of the footer note in (a)
      legend center    = center of the column-title row in (a)
      left gap         = gap_ab inches between (a)'s logic boxes and (b)'s ink
      right margin     = (a)'s left margin (symmetric page margins)
    All quantities are figure inches. Returns a dict of measured alignment errors."""
    W, dpi = fig.get_figwidth(), fig.dpi
    r = fig.canvas.get_renderer()
    ink = lambda bb: (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
    top_line = a_off + A_TOP
    bot_line = a_off + A_TOP - 3 * A_BH - 2 * A_GAP
    head_c, foot_c = a_off + A_TITLE_Y, a_off + A_NOTE_Y
    a_right = A_LX + A_LW                       # panel (a) axes start at x = 0
    fig.canvas.draw()
    mL = ink(drawn_extent(ax_a, r))[0]
    x0, w = 4.4, 1.8
    for _ in range(iters):
        ax_b.set_position([x0 / W, bot_line / H, w / W, (top_line - bot_line) / H])
        ax_b.xaxis.set_label_coords(0.5, (foot_c - bot_line) / (top_line - bot_line))
        fig.canvas.draw()
        tb = ink(drawn_extent(ax_b, r))
        dl, dr = x0 - tb[0], tb[2] - (x0 + w)
        x0 = a_right + gap_ab + dl
        w = (W - mL - dr) - x0
    hs, labels = legend_handles()
    tb = ink(drawn_extent(ax_b, r))
    cx = (tb[0] + tb[2]) / 2 / W               # centered on panel (b)'s full ink width
    leg = fig.legend(hs, labels, loc="center", bbox_to_anchor=(cx, head_c / H), ncol=4,
                     frameon=False, fontsize=7.8, handletextpad=0.35, columnspacing=0.9,
                     handlelength=1.6, borderaxespad=0.0)
    fig.canvas.draw()
    lb = ink(leg.get_window_extent(r))
    tb = ink(drawn_extent(ax_b, r))
    xl = ink(ax_b.xaxis.label.get_window_extent(r))
    ab = ink(ax_b.get_window_extent(r))
    return {"axes": (x0, w), "ink_b": tb, "legend": lb, "margin_left_a": mL,
            "err_top": ab[3] - top_line, "err_bottom": ab[1] - bot_line,
            "err_footer": (xl[1] + xl[3]) / 2 - foot_c, "err_header": (lb[1] + lb[3]) / 2 - head_c,
            "gap_ab": tb[0] - a_right, "margin_right_b": W - tb[2],
            "legend_inside_b": lb[0] >= tb[0] - 1e-3 and lb[2] <= tb[2] + 1e-3,
            "legend_w": lb[2] - lb[0], "ink_w_b": tb[2] - tb[0]}


def main():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="figures/fig1.pdf")
    args = ap.parse_args()
    summary = json.load(open(args.summary))
    H, a_off = 3.12, 0.21
    fig = plt.figure(figsize=(TEXTWIDTH_IN, H))
    FigureCanvasAgg(fig)                        # raster canvas for exact text measurement
    ax_a = fig.add_axes([0.0, a_off / H, PA_W / TEXTWIDTH_IN, PA_H / H])
    ax_b = fig.add_axes([0.7, 0.2, 0.3, 0.6])
    panel_a(ax_a)
    panel_b(ax_b, summary, args.model)
    m = layout(fig, ax_a, ax_b, H, a_off)
    label_top = (H - 0.02) / H
    fig.text(m["margin_left_a"] / TEXTWIDTH_IN, label_top, r"\textbf{(a)}", ha="left", va="top",
             fontsize=10, color=INK)
    fig.text(m["ink_b"][0] / TEXTWIDTH_IN, label_top, r"\textbf{(b)}", ha="left", va="top",
             fontsize=10, color=INK)
    print("alignment check (inches):")
    print(f"  arrow length {ARROW_LEN:.3f} (all horizontal arrows), clearance to boxes {ARROW_GAP:.3f}")
    for k in ("err_header", "err_top", "err_bottom", "err_footer"):
        print(f"  {k:12s} {m[k]:+.4f}")
    print(f"  left margin (a) {m['margin_left_a']:.3f} | right margin (b) {m['margin_right_b']:.3f}"
          f" | gap a-b {m['gap_ab']:.3f} | legend {m['legend_w']:.3f} vs (b) ink {m['ink_w_b']:.3f},"
          f" within: {m['legend_inside_b']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
