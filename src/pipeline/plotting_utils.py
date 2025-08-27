from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys

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

def _plot_pd(H0: np.ndarray, H1: np.ndarray, out_png: Path, title: str,
             min_persistence: float = 0.04, mode: str = "birth-persistence") -> None:
    H0 = _filter(_clean(H0), min_persistence)
    H1 = _filter(_clean(H1), min_persistence)

    if mode == "birth-persistence":
        def xy(D):
            if D.size == 0:
                return np.empty((0, 2)), np.empty((0,))
            x = D[:, 0]; y = D[:, 1] - D[:, 0]
            return np.column_stack([x, y]), y
        xlabel, ylabel, draw_diag = "Birth", "Persistence (death − birth)", False
    else:  # "birth-death"
        def xy(D):
            if D.size == 0:
                return np.empty((0, 2)), np.empty((0,))
            x = D[:, 0]; y = D[:, 1]
            return np.column_stack([x, y]), (y - x)
        xlabel, ylabel, draw_diag = "Birth", "Death", True

    H0xy, _ = xy(H0); H1xy, _ = xy(H1)

    lim = 1.0
    for A in (H0xy, H1xy):
        if A.size:
            lim = max(lim, float(A.max()))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 5.2))
    if H0xy.size:
        plt.scatter(H0xy[:, 0], H0xy[:, 1], s=16, alpha=0.7, label=f"H0 (n={len(H0xy)})")
    if H1xy.size:
        plt.scatter(H1xy[:, 0], H1xy[:, 1], s=16, alpha=0.7, marker="^", label=f"H1 (n={len(H1xy)})")
    if draw_diag:
        plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)

    plt.xlim(0, lim)
    if mode == "birth-death":
        plt.ylim(0, lim)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.axis("equal"); plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def _plot_barcode(D: np.ndarray, out_png: Path, title: str,
                  min_persistence: float = 0.04) -> None:
    """
    Plot a barcode from [birth, death] rows (after cleaning/filtering).
    Saves to out_png.
    """
    D = _filter(_clean(D), min_persistence)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if D.size == 0:
        plt.figure(figsize=(7, 3))
        plt.title(f"{title}\n(empty)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()
        return

    # Sort by interval length (longer on top)
    pers = D[:, 1] - D[:, 0]
    order = np.argsort(pers)[::-1]
    D = D[order]

    plt.figure(figsize=(7.2, 3.6))
    for i, (b, d) in enumerate(D):
        plt.hlines(y=i, xmin=b, xmax=d, linewidth=2)
        plt.plot([b, d], [i, i], ".", ms=5)

    plt.gca().invert_yaxis()
    plt.xlabel("Filtration value")
    plt.ylabel("Interval index (sorted by length)")
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_dataset(diagrams_dir: Path, out_dir: Path,
                 min_persistence: float = 0.04, mode: str = "birth-persistence") -> None:
    """Plot all *_H0.npy/*_H1.npy pairs in diagrams_dir into out_dir."""
    diagrams_dir = Path(diagrams_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    bases = {Path(p).name.replace("_H0.npy", "") for p in glob(str(diagrams_dir / "*_H0.npy"))}
    for base in sorted(bases):
        H0 = np.load(diagrams_dir / f"{base}_H0.npy")
        H1 = np.load(diagrams_dir / f"{base}_H1.npy")

        # 1) Persistence diagram
        _plot_pd(H0, H1, out_dir / f"{base}.png", title=base,
                 min_persistence=min_persistence, mode=mode)

        # 2) Barcodes (H0 and H1)
        barcode_dir = out_dir / "barcodes"
        barcode_dir.mkdir(parents=True, exist_ok=True)
        _plot_barcode(H0, barcode_dir / f"{base}_H0_barcode.png",
                      title=f"{base} • H0", min_persistence=min_persistence)
        _plot_barcode(H1, barcode_dir / f"{base}_H1_barcode.png",
                      title=f"{base} • H1", min_persistence=min_persistence)


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
                PROJECT_ROOT / "results_raw/nucmm/mouse/raw/diagrams",
                PROJECT_ROOT / "results_raw/nucmm/mouse/raw/plots"),
                ("zebrafish",
                 PROJECT_ROOT / "results_raw/nucmm/zebrafish/raw/diagrams",
                 PROJECT_ROOT / "results_raw/nucmm/zebrafish/raw/plots"),
            ]
            for name, diag_dir, plot_dir in datasets:
                print(f"[plot] {name}: {diag_dir} -> {plot_dir}")
                plot_dataset(diag_dir, plot_dir, min_persistence=0.04, mode="birth-death")
            print("[plot] DONE")
        else:
            main()

