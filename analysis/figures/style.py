"""Shared figure style for the paper.

Fonts: Times text (`times`), Times-matched math (`newtxmath`), Inconsolata code; rendered by LaTeX.
Type scale (figures are printed at 100%, so these are printed point sizes; captions are 10 pt):
  10   bold     panel labels (a), (b)
  8.5  bold     titles that name a region: column and row headers, axis titles, facet titles
  8    regular  content: tick labels, legend entries, cell values, text inside boxes
  7.5  regular  secondary labels in gray (operator lists, theorem references, keys, "query vertex");
                italic for notes (the takeaway line)
  7.3  regular  Inconsolata, only for verbatim serialized text
  White text on a colored fill (header bars, node names) is bold for contrast.
  Serialization abbreviations EL/AL/NC are never bold outside header bars; color carries identity.
Color roles:
  serialization (the only categorical hues): EL purple / circle, AL teal / square, NC amber / triangle,
    ECL (theory-derived) dark slate / diamond / dashed line
  node colors in drawn graphs only: red, blue
  theory prediction: filled = decidable, hollow = not decidable (no extra hue)
  structure: grays; sequential values: light gray -> deep slate; highlight: weight, never a new hue
"""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 16.0 / 2.54     # ACL two columns + gap
COLUMN_IN = 7.7 / 2.54          # one ACL column

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}\usepackage{amsthm}"
                           r"\usepackage{newtxmath}\usepackage{inconsolata}\usepackage{xcolor}",
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 1.1, "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42,
})

SER = {
    "EL": {"color": "#6C4FA3", "marker": "o", "name": "Edge list"},
    "AL": {"color": "#1F8A84", "marker": "s", "name": "Adjacency list"},
    "NC": {"color": "#D98B1E", "marker": "^", "name": "Node-centric"},
    "ECL": {"color": "#34495E", "marker": "D", "name": "Edge list, inline colors", "ls": (0, (4, 2))},
    "ELB": {"color": "#A98BD6", "marker": "v", "name": "Edge list, block colors", "ls": (0, (1.5, 1.5))},   # , E33
}
SER3 = ["EL", "AL", "NC"]          # the three studied serializations (ECL is the derived fourth)
NODE = {"red": "#D1495B", "blue": "#3D6FB6"}
INK, QUIET, GRID, PANEL_BG = "#1F2328", "#5A6270", "#D5D9E0", "#F6F7F9"
SEQ_LO, SEQ_HI = "#EEF1F5", "#1F3A5F"
ACCENT = "#C0392B"               # accent for marked cells

def vcol(v):
    """Color of a value label."""
    return INK


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("slate", [SEQ_LO, "#A9B8CC", "#4E6A8F", SEQ_HI])


def text_on(value, vmin=0.5, vmax=1.0):
    """Ink color readable on the sequential colormap at this value."""
    t = (value - vmin) / (vmax - vmin)
    return "white" if t > 0.55 else INK


def style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)


def drawn_extent(ax, r):
    from matplotlib.transforms import Bbox
    arts = list(ax.texts) + list(ax.patches) + list(ax.lines) + list(ax.collections) + list(ax.images)
    if ax.axison:
        arts += [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        arts += [sp for sp in ax.spines.values() if sp.get_visible()]
    bbs = [a.get_window_extent(r) for a in arts if a.get_visible()]
    return Bbox.union([b for b in bbs if b.width > 0 or b.height > 0])
