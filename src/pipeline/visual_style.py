# src/pipeline/visual_style.py
from __future__ import annotations
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pathlib
import numpy as np
# Apply global look (calm, readable)
def apply_style():
    mpl.rcParams.update({


            # Output
            "figure.dpi": 180, "savefig.dpi": 300,

            # Typography (smaller + consistent)
            "font.family": "Times New Roman",
            "font.size": 10,  # base text
            "axes.titlesize": 12,  # was larger
            "axes.titleweight": "medium",
            "axes.titlepad": 6,  # spacing above axes
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10, "ytick.labelsize": 10,

            # Clean frame + grid
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.45,

            # Legends
            "legend.frameon": True, "legend.fancybox": True,
            "legend.edgecolor": "0.8", "legend.framealpha": 0.9,
        })



# Nice ticks + unit box for PD
def format_pd_axes(ax, mode: str = "birth-death", min_persistence: float = 0.05):
    if mode == "birth-death":
        lo, hi = 0.0, 1.0
        # diagonal + light persistence band

        xx = np.linspace(lo, hi, 256)
        ax.fill_between(xx, xx, xx + min_persistence, color="0.93", zorder=0, label=f"p<{min_persistence}")
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.45")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Birth"); ax.set_ylabel("Death")
    else:
        ax.axhspan(0, min_persistence, color="0.93", zorder=0, label=f"p<{min_persistence}")
        ax.set_xlabel("Birth"); ax.set_ylabel("Persistence (death − birth)")
    # tidy ticks
    ax.xaxis.set_major_locator(MultipleLocator(0.2)); ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2)); ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f")); ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

# Save helper: always export PNG + SVG, tight bounds
def savefig(fig, path_no_ext):

    p = pathlib.Path(path_no_ext)
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
