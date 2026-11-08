"""Figure fig:ladder (Section 4): what each positional capability makes definable, per serialization (Sections 3 and 4).

Rows: capabilities (cumulative). Columns: serializations. Each cell shows two properties:
'v is red' (Q0) and 'v has a red neighbor' (D1a). Filled marker = definable (proved),
hollow = not definable (proved); a dagger marks "definable for bounded degree" (Proposition prop:shift, part c).
Result numbers follow the paper's numbering.
Marker shape and color follow the serialization (color system).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import SER3, INK, PANEL_BG, ACCENT, QUIET, SER, TEXTWIDTH_IN, COLUMN_IN, drawn_extent

W, H = TEXTWIDTH_IN, 2.56
ROWS = [  # label, operators
    (r"Names indistinguishable", r"(name-blind)"),
    (r"Names matchable", r"$\mathsf{P},\ \mathsf{M}$"),
    (r"$+$ local shift", r"$\mathsf{P},\ \mathsf{M},\ \mathsf{Y}$"),
    (r"$+$ since (line heads)", r"$\mathsf{P},\ \mathsf{M},\ \mathsf{Y},\ \mathsf{S}$"),
]
# status per row, serialization: (Q0, D1a); values: "yes", "no", "no+" (not definable for unbounded
# degree, proved, Theorem thm:lower; definable for bounded degree, Proposition prop:shift (c))
STATUS = [
    {"EL": ("no", "no"), "AL": ("no", "no"), "NC": ("no", "no")},
    {"EL": ("no", "no"), "AL": ("no", "no"), "NC": ("no", "no")},
    {"EL": ("yes", "yes"), "AL": ("yes", "no+"), "NC": ("yes", "no+")},
    {"EL": ("yes", "yes"), "AL": ("yes", "yes"), "NC": ("yes", "yes")},
]
# numbers of the supporting results in the paper
NUM = {'thm:blind': '1', 'thm:noshift': '3', 'prop:shift': '4', 'thm:lower': '7', 'prop:since': '5'}


def _num(label):
    return NUM[label]


T1, T3, P4, T6, P5 = (_num(x) for x in ("thm:blind", "thm:noshift", "prop:shift", "thm:lower", "prop:since"))
REF = [  # result supporting each row, per serialization
    {"EL": f"Thm {T1}", "AL": f"Thm {T1}", "NC": f"Thm {T1}"},
    {"EL": f"Thm {T3}", "AL": f"Thm {T3}", "NC": f"Thm {T3}"},
    {"EL": f"Prop {P4}", "AL": f"Prop {P4}, Thm {T6}", "NC": f"Prop {P4}, Thm {T6}"},
    {"EL": f"Prop {P4}", "AL": f"Prop {P5}", "NC": f"Prop {P5}"},
]

X_LAB, X0, CW = 0.26, 1.98, 1.40      # label column left, first data column left, column width
Y_HEAD, RH = 2.38, 0.47              # header row center, row height
ROW_Y = [Y_HEAD - 0.42 - i * RH for i in range(4)]


def pill(ax, cx, cy, label, ser, status):
    st = SER[ser]
    w, h = 0.56, 0.24
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.11",
                                fc="white", ec="#C9CED6", lw=0.9, zorder=2))
    filled = status == "yes"
    ax.plot([cx - w / 2 + 0.12], [cy], marker=st["marker"], ms=7.2, mew=1.6,
            mfc=st["color"] if filled else "white", mec=st["color"], zorder=3)
    if status == "no+":
        label = label[:-1] + r"^{\dagger}$"
    ax.text(cx - w / 2 + 0.22, cy - 0.005, label, ha="left", va="center", fontsize=8, color=INK, zorder=3)


def main(out="fig2.pdf"):
    fig = plt.figure(figsize=(W, H)); FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # banded rows
    for i, y in enumerate(ROW_Y):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((X_LAB - 0.08, y - RH / 2), X0 + 3 * CW - X_LAB + 0.02, RH,
                                       fc=PANEL_BG, ec="none", zorder=0))
    # column headers: serialization chips (same style as Figure 1)
    for j, s in enumerate(SER3):
        cx = X0 + (j + 0.5) * CW
        ax.add_patch(FancyBboxPatch((cx - 0.52, Y_HEAD - 0.12), 1.04, 0.24,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=SER[s]["color"],
                                    ec=SER[s]["color"], zorder=1))
        ax.text(cx, Y_HEAD - 0.005, rf"\textbf{{{SER[s]['name']} ({s})}}", ha="center", va="center",
                fontsize=8, color="white", zorder=2)
    ax.text(X_LAB + 0.22, Y_HEAD, r"\textbf{What the model can do}", ha="left", va="center", fontsize=8.5, color=INK)

    # capability axis arrow
    # rows 2-4 add capabilities; row 1 (name-blind) is not nested with them, so the arrow starts at row 2
    arrow = FancyArrowPatch((X_LAB + 0.02, ROW_Y[1] + RH / 2 - 0.04), (X_LAB + 0.02, ROW_Y[-1] - RH / 2 + 0.04),
                            arrowstyle="-|>", mutation_scale=12, lw=1.7, color=QUIET, shrinkA=0, shrinkB=0,
                            capstyle="butt", joinstyle="miter")  # butt/miter: drawn ends coincide with the given points
    ax.add_patch(arrow)
    for i, (lab, ops) in enumerate(ROWS):
        y = ROW_Y[i]
        ax.text(X_LAB + 0.22, y + 0.075, lab, ha="left", va="center", fontsize=8, color=INK)
        ax.text(X_LAB + 0.22, y - 0.105, ops, ha="left", va="center", fontsize=7.5, color=QUIET)
        for j, s in enumerate(SER3):
            cx = X0 + (j + 0.5) * CW
            q0, d1 = STATUS[i][s]
            pill(ax, cx - 0.31, y + 0.055, r"$\mathit{red}$", s, q0)
            pill(ax, cx + 0.31, y + 0.055, r"$\Diamond\mathit{red}$", s, d1)
            ax.text(cx, y - 0.155, REF[i][s], ha="center", va="center", fontsize=7.5, color=QUIET)

    # legend row, below the table, outside all cells
    ly = ROW_Y[-1] - RH / 2 - 0.17
    lx = X_LAB + 0.75
    items = [("yes", r"definable (proved)"), ("no", r"not definable (proved)"),
             ("dag", r"$\dagger$\ definable for bounded degree (Prop %sc)" % P4)]
    xs = [lx, lx + 1.30, lx + 2.70]
    for x, (stt, txt) in zip(xs, items):
        if stt != "dag":
            ax.plot([x], [ly], marker="o", ms=6.0, mew=1.5, mfc=INK if stt == "yes" else "white", mec=INK)
            ax.text(x + 0.16, ly - 0.004, txt, ha="left", va="center", fontsize=8, color=INK)
        else:
            ax.text(x - 0.06, ly - 0.004, txt, ha="left", va="center", fontsize=8, color=INK)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = drawn_extent(ax, r)
    print(f"ink x {bb.x0/fig.dpi:.3f}-{bb.x1/fig.dpi:.3f} in of {W:.3f}; y {bb.y0/fig.dpi:.3f}-{bb.y1/fig.dpi:.3f} of {H:.3f}")
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out)


def main_compact(out="fig2c.pdf"):
    """Column-width version for Section 4 (Figure fig:ladder): per cell the two markers (``v is red'', left;
    ``v has a red neighbor'', right) and the supporting result; same statuses and numbers as the full figure."""
    Wc, Hc = COLUMN_IN, 1.62
    xl, x0, cw = 0.04, 1.02, (COLUMN_IN - 1.06) / 3
    yh, rh = Hc - 0.13, 0.30
    ry = [yh - 0.39 - i * rh for i in range(4)]
    fig = plt.figure(figsize=(Wc, Hc)); FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wc); ax.set_ylim(0, Hc); ax.set_aspect("equal"); ax.axis("off")
    for i, y in enumerate(ry):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((xl, y - rh / 2), Wc - xl - 0.02, rh, fc=PANEL_BG, ec="none", zorder=0))
    for j, s_ in enumerate(SER3):
        cx = x0 + (j + 0.5) * cw
        ax.add_patch(FancyBboxPatch((cx - 0.30, yh - 0.09), 0.60, 0.18, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=SER[s_]["color"], ec=SER[s_]["color"], zorder=1))
        ax.text(cx, yh - 0.004, rf"\textbf{{{s_}}}", ha="center", va="center", fontsize=7.5, color="white", zorder=2)
        ax.text(cx - 0.17, yh - 0.20, r"$\mathit{red}$", ha="center", va="center", fontsize=6.5, color=QUIET)
        ax.text(cx + 0.17, yh - 0.20, r"$\Diamond\mathit{red}$", ha="center", va="center", fontsize=6.5, color=QUIET)
    labs = [(r"Name-blind", r""), (r"Match", r"$\mathsf{P},\mathsf{M}$"), (r"$+$ shift", r"$\mathsf{Y}$"),
            (r"$+$ since", r"$\mathsf{S}$")]
    for i, (lab, ops) in enumerate(labs):
        y = ry[i]
        ax.text(xl + 0.05, y + 0.02, lab + (r"\ \ " + r"{\color[HTML]{5A6270}" + ops + "}" if ops else ""),
                ha="left", va="center", fontsize=7.5, color=INK)
        for j, s_ in enumerate(SER3):
            cx = x0 + (j + 0.5) * cw
            for dx, stt in ((-0.17, STATUS[i][s_][0]), (0.17, STATUS[i][s_][1])):
                st = SER[s_]
                ax.plot([cx + dx], [y + 0.05], marker=st["marker"], ms=5.6, mew=1.3,
                        mfc=st["color"] if stt == "yes" else "white", mec=st["color"], zorder=3)
                if stt == "no+":
                    ax.text(cx + dx + 0.075, y + 0.10, r"$\dagger$", ha="left", va="center", fontsize=6.5, color=INK)
            ax.text(cx, y - 0.10, REF[i][s_], ha="center", va="center", fontsize=6, color=QUIET)
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main_compact()      # the paper uses the column-width version (Section 4); main() draws the earlier full-width one
