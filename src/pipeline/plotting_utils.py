from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
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
    mode: str = "birth-death",     # classic PD (diagonal visible)
    annotate_top_h1: int = 3,      # few labels keeps it clean
    annotate_top_h0: int = 1
) -> None:
    """
    Pretty birth–death PD with diagonal + light persistence band,
    square axes, calm markers, and a slim colorbar.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- clean input & threshold by persistence
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

    # --- figure
    fig, ax = plt.subplots(figsize=(5.4, 5.4))

    # axes formatting (draws diagonal + shaded band for birth–death)
    format_pd_axes(ax, mode=mode, min_persistence=min_persistence)

    # common colormap scaled by persistence
    cmap = get_cmap("viridis")
    vmax = float(max(
        (np.max(p0) if p0.size else 0.0),
        (np.max(p1) if p1.size else 0.0),
        min_persistence * 1.5
    ))
    norm = Normalize(vmin=min_persistence, vmax=vmax)

    # plots (small, calm markers)
    if x0.size:
        ax.scatter(x0, y0, s=18, marker="o", c=norm(p0), cmap=cmap,
                   edgecolors="none", alpha=0.9, label=f"H0 (n={len(x0)})")
    if x1.size:
        ax.scatter(x1, y1, s=22, marker="^", c=norm(p1), cmap=cmap,
                   edgecolors="none", alpha=0.95, label=f"H1 (n={len(x1)})")

    # slim colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Persistence")

    # minimal annotations (top-k by persistence)
    def _annotate_top(D, P, tag, k):
        if not D.size or k <= 0: return
        xx, yy = xy(D)
        for i in np.argsort(P)[::-1][:k]:
            ax.annotate(f"{tag}:{P[i]:.2f}", (xx[i], yy[i]),
                        xytext=(4, 4), textcoords="offset points", fontsize=9)

    _annotate_top(H1, p1, "H1", annotate_top_h1)
    _annotate_top(H0, p0, "H0", annotate_top_h0)

    # tidy legend + title
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.set_title(title, pad=6)

    # save PNG + SVG
    savefig(fig, out_path.with_suffix(""))  # writes .png and .svg
    plt.close(fig)



def _plot_barcode(D: np.ndarray, out_png: Path, title: str,
                  min_persistence: float = 0.04, annotate_top: int = 5) -> None:
    def _clean(D):
        if D is None or D.size == 0: return np.empty((0, 2), np.float32)
        D = np.asarray(D, np.float32)
        return D[np.isfinite(D).all(axis=1)] if (D.ndim == 2 and D.shape[1] == 2) else np.empty((0, 2), np.float32)

    D = _clean(D)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if D.size:
        pers = D[:, 1] - D[:, 0]
        keep = pers >= min_persistence
        D, pers = D[keep], pers[keep]
        order = np.argsort(pers)[::-1]
        D, pers = D[order], pers[order]

    fig = plt.figure(figsize=(10.0, 4.2)); ax = plt.gca()
    ax.axvspan(0, min_persistence, color="0.95", zorder=0)

    if D.size == 0:
        ax.text(0.5, 0.5, "No intervals ≥ threshold", ha="center", va="center"); ax.set_axis_off()
    else:
        y = np.arange(len(D))
        cmap = get_cmap("viridis"); norm = Normalize(vmin=min_persistence, vmax=float(pers.max()))
        for i, (b, d) in enumerate(D):
            ax.hlines(y[i], b, d, lw=3.0, color=cmap(norm(pers[i])), alpha=0.95)
        for i in range(min(annotate_top, len(D))):
            b, d = D[i]; ax.text(d, i, f"  {d-b:.2f}", va="center", fontsize=9)

        ax.set_ylim(-1, len(D)); ax.set_yticks([])
        ax.set_xlabel("Filtration value")
        ax.set_title(f"{title}  •  n={len(D)}  •  min_pers={min_persistence}")

    ax.grid(True, axis="x", ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close(fig)

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
                 min_persistence=min_persistence, mode=mode, annotate_top=5)
        _plot_barcode(H0, bc_h0 / f"{base}_H0_barcode.png",
                      title=f"{base} • H0", min_persistence=min_persistence, annotate_top=5)
        _plot_barcode(H1, bc_h1 / f"{base}_H1_barcode.png",
                      title=f"{base} • H1", min_persistence=min_persistence, annotate_top=5)




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

