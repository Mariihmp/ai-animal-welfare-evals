"""Shared matplotlib style for all projects (light-mode PNGs for markdown reports).

Palette from the dataviz reference instance: categorical slots in fixed order, a
blue<->red diverging pair with a gray midpoint, recessive hairline grid, and text
in ink tokens (never series colours).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"   # red = penalised, blue = favoured
DIVERGING = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.labelcolor": INK2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.markersize": 7,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def header(fig, title, sub=None):
    """Figure-level title + subtitle placed above the top of the figure (bbox='tight' keeps them)."""
    from matplotlib.transforms import ScaledTranslation
    base = fig.transFigure
    fig.text(0.01, 1.0, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom",
             transform=base + ScaledTranslation(0, 0.30 if sub else 0.08, fig.dpi_scale_trans))
    if sub:
        fig.text(0.01, 1.0, sub, fontsize=9, color=INK2, ha="left", va="bottom",
                 transform=base + ScaledTranslation(0, 0.08, fig.dpi_scale_trans))


def subtitle(ax, text, y=1.02):
    ax.text(0, y, text, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("figure:", path)
