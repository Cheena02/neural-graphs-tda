from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
import matplotlib.patheffects as PathEffects
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from src.pipeline.visual_style import apply_style, format_pd_axes, savefig
apply_style()


def _clean(D: np.ndarray) -> np.ndarray:
    if D is None or D.size == 0:
        return np.empty((0, 2), np.float32)
    D = np.asarray(D, np.float32)
    if D.ndim != 2 or D.shape[1] != 2:
        return np.empty((0, 2), np.float32)
    mask = np.isfinite(D).all(axis=1)
    return D[mask]

def _filter(D: np.ndarray, eps: float) -> np.ndarray:
    if D.size == 0:
        return D
    pers = D[:, 1] - D[:, 0]
    return D[pers >= eps]

def _plot_pd(
    H0: np.ndarray,
    H1: np.ndarray,
    out_path: Path,
    title: str,
    min_persistence: float = 0.05,
    mode: str = "birth-death",
    annotate_top_h1: int = 3,
    annotate_top_h0: int = 0,        # 0 avoids clutter from H0
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- clean + threshold (unchanged from before) --------------------------
    def _clean(D):
        D = np.asarray(D, np.float32)
        return D[np.isfinite(D).all(axis=1)] if D.ndim == 2 and D.shape[1] == 2 else np.empty((0, 2), np.float32)
    H0 = _clean(H0); H1 = _clean(H1)
    p0 = (H0[:, 1] - H0[:, 0]) if H0.size else np.empty((0,), np.float32)
    p1 = (H1[:, 1] - H1[:, 0]) if H1.size else np.empty((0,), np.float32)
    if H0.size:
        keep = p0 >= float(min_persistence); H0, p0 = H0[keep], p0[keep]
    if H1.size:
        keep = p1 >= float(min_persistence); H1, p1 = H1[keep], p1[keep]

    def xy(D):
        return (D[:, 0], D[:, 1]) if mode == "birth-death" else (D[:, 0], D[:, 1] - D[:, 0])

    x0, y0 = xy(H0); x1, y1 = xy(H1)

    # ---- figure & axes (square; controlled margins) -------------------------
    fig, ax = plt.subplots(figsize=(6.0, 6.0))    # a touch bigger than before
    format_pd_axes(ax, mode=mode, min_persistence=min_persistence)
    ax.margins(x=0.02, y=0.02)                    # small breathing room

    # ---- color scale (unchanged idea) ---------------------------------------
    cmap = get_cmap("viridis")
    vmax = float(max(p0.max() if p0.size else 0.0, p1.max() if p1.size else 0.0, min_persistence * 1.5))
    norm = Normalize(vmin=min_persistence, vmax=vmax)

    # Dynamic marker sizes by persistence = visual hierarchy
    if x0.size:
        s0 = 14 + 18 * (norm(p0) - norm(min_persistence))
        ax.scatter(x0, y0, s=s0, marker="o", c=norm(p0), cmap=cmap,
                   edgecolors="none", alpha=0.9, label=f"H0 (n={len(x0)})")
    if x1.size:
        s1 = 16 + 22 * (norm(p1) - norm(min_persistence))
        ax.scatter(x1, y1, s=s1, marker="^", c=norm(p1), cmap=cmap,
                   edgecolors="none", alpha=0.95, label=f"H1 (n={len(x1)})")

    # Slim colorbar with small ticks
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Persistence")
    cb.ax.tick_params(labelsize=9)

    # Legend INSIDE top-left so it never collides with colorbar
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)
    for txt in leg.get_texts():
        txt.set_fontsize(10)

    # Minimal title (smaller + tighter)
    ax.set_title(title, fontsize=14, weight="medium", pad=4)

    # ---- small, non-overlapping annotations ---------------------------------
    def _annotate_top(D, P, tag, k):
        if not D.size or k <= 0: return
        xx, yy = xy(D)
        idx = np.argsort(P)[::-1][:k]
        for i, j in enumerate(idx):
            # alternate nudges to avoid stacking
            dx, dy = ((6, 6), (6, -6), (-6, 6))[i % 3]
            t = ax.annotate(f"{tag}:{P[j]:.2f}", (xx[j], yy[j]),
                            xytext=(dx, dy), textcoords="offset points",
                            fontsize=9, color="black",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
            # subtle outline improves readability on busy backgrounds
            t.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

    _annotate_top(H1, p1, "H1", annotate_top_h1)
    _annotate_top(H0, p0, "H0", annotate_top_h0)

    savefig(fig, out_path.with_suffix(""))
    plt.close(fig)



def _plot_barcode(
    D: np.ndarray,
    out_path: Path,
    title: str,
    min_persistence: float = 0.05,
    annotate_top: int = 4,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(D):
        D = np.asarray(D, np.float32)
        return D[np.isfinite(D).all(axis=1)] if D.ndim == 2 and D.shape[1] == 2 else np.empty((0, 2), np.float32)
    D = _clean(D)
    if D.size == 0:
        fig, ax = plt.subplots(figsize=(10.5, 3.6), constrained_layout=True)
        ax.text(0.5, 0.5, "No intervals ≥ threshold", ha="center", va="center")
        ax.set_axis_off(); savefig(fig, out_path.with_suffix("")); plt.close(fig); return

    P = D[:, 1] - D[:, 0]
    keep = P >= float(min_persistence); D, P = D[keep], P[keep]
    order = np.argsort(P)[::-1]; D, P = D[order], P[order]

    # Figure with breathing room; smaller title
    fig, ax = plt.subplots(figsize=(10.5, 3.6), constrained_layout=True)
    ax.axvspan(0, min_persistence, color="0.93", zorder=0)  # noise band
    cmap = get_cmap("viridis"); norm = Normalize(vmin=min_persistence, vmax=float(max(P.max(), min_persistence * 1.05)))

    y = np.arange(len(D))
    for i, (b, d) in enumerate(D):
        ax.hlines(y[i], b, d, lw=3.0, color=cmap(norm(P[i])), alpha=0.95)

    # Label only top few bars; offset alternates to prevent overlap
    for i in range(min(annotate_top, len(D))):
        b, d = D[i]
        yoff = 0.35 if (i % 2) else -0.35
        t = ax.text(d, i + yoff, f"{(d-b):.2f}", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.85", alpha=0.85))
        t.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

    ax.set_ylim(-1, len(D)); ax.set_yticks([])
    ax.margins(x=0.02)  # stops right-edge numbers from being cut
    ax.set_xlim(left=0, right=max(1.0, float(D[:, 1].max()) * 1.03))
    ax.set_yticks([])
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Persistence")
    ax.set_title(f"{title}  •  n={len(D)}  •  min_pers={min_persistence}", pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, axis="x", which="both", linestyle=":")

    savefig(fig, out_path.with_suffix(""))
    plt.close(fig)



def plot_dataset(diagrams_dir: Path, out_dir: Path,
                 min_persistence: float = 0.04,
                 mode: str = "birth-persistence") -> None:
    diagrams_dir = Path(diagrams_dir); out_dir = Path(out_dir)
    ph_dir   = out_dir / "ph_diagrams"
    bc_h0    = out_dir / "barcodes" / "H0"
    bc_h1    = out_dir / "barcodes" / "H1"
    for d in (ph_dir, bc_h0, bc_h1): d.mkdir(parents=True, exist_ok=True)

    bases = {Path(p).name.replace("_H0.npy", "") for p in glob(str(diagrams_dir / "*_H0.npy"))}
    for base in sorted(bases):
        H0 = np.load(diagrams_dir / f"{base}_H0.npy")
        H1 = np.load(diagrams_dir / f"{base}_H1.npy")
        _plot_pd(H0, H1, ph_dir / f"{base}_ph.png", title=base,
                 min_persistence=min_persistence, mode=mode, annotate_top_h1=5,annotate_top_h0 = 1)
        _plot_barcode(H0, bc_h0 / f"{base}_H0_barcode.png",
                      title=f"{base} • H0",
                      min_persistence=min_persistence, annotate_top=0)

        _plot_barcode(H1, bc_h1 / f"{base}_H1_barcode.png",
                      title=f"{base} • H1",
                      min_persistence=min_persistence, annotate_top=0)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagrams_dir", required=True, help="folder containing *_H0.npy and *_H1.npy")
    ap.add_argument("--out_dir", required=True, help="where to save plots")
    ap.add_argument("--min_persistence", type=float, default=0.04)
    ap.add_argument("--mode", choices=["birth-persistence", "birth-death"], default="birth-persistence")
    args = ap.parse_args()

    plot_dataset(Path(args.diagrams_dir), Path(args.out_dir),
                 min_persistence=args.min_persistence, mode=args.mode)

if __name__ == "__main__":
        if len(sys.argv) == 1:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]

            datasets = [
                ("mouse",
                PROJECT_ROOT / "my_results/results_raw/nucmm/mouse/raw/diagrams",
                PROJECT_ROOT / "my_results/results_raw/nucmm/mouse/raw/plots"),
                ("zebrafish",
                 PROJECT_ROOT / "my_results/results_raw/nucmm/zebrafish/raw/diagrams",
                 PROJECT_ROOT / "my_results/results_raw/nucmm/zebrafish/raw/plots"),
            ]
            for name, diag_dir, plot_dir in datasets:
                print(f"[plot] {name}: {diag_dir} -> {plot_dir}")
                plot_dataset(diag_dir, plot_dir, min_persistence=0.04, mode="birth-death")
            print("[plot] DONE")
        else:
            main()

