# src/pipeline/visual_style.py
from __future__ import annotations
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Apply global look (calm, readable)
def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "medium",
        "axes.labelsize": 12,
        "axes.labelweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
    })

# Nice ticks + unit box for PD
def format_pd_axes(ax, mode: str = "birth-death", min_persistence: float = 0.05):
    if mode == "birth-death":
        lo, hi = 0.0, 1.0
        # diagonal + light persistence band
        import numpy as np
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
    import pathlib
    p = pathlib.Path(path_no_ext)
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
