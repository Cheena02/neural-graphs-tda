from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from gudhi import CubicalComplex


# ---------------- Utilities ----------------
def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mn, mx = float(img.min()), float(img.max())
    return (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img, np.float32)


def load_grayscale(fp: Path) -> np.ndarray:
    img = Image.open(fp).convert("L")  # grayscale
    return np.array(img, dtype=np.float32)


def preproc_for_cubical(
    img: np.ndarray,
    denoise: bool = True,
    invert: bool = False,
    clahe: bool = False,
) -> np.ndarray:
    """
    Light, optional preprocessing intended to stabilize PH a bit
    without binarizing (we keep grayscale for cubical filtration).
    """
    x = img.copy()
    x = normalize01(x)

    if denoise:
        # small Gaussian blur smooths sensor noise, not edges too much
        x = cv2.GaussianBlur(x, (3, 3), 0.6)

    if clahe:
        # Optional local contrast; can help separate textures
        _u8 = (x * 255).astype(np.uint8)
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = _clahe.apply(_u8).astype(np.float32) / 255.0

    if invert:
        x = 1.0 - x

    return normalize01(x)


def build_cubical(img01: np.ndarray) -> CubicalComplex:
    """
    GUDHI expects a 2D array of function values (float OK).
    Filtration is *sublevel* (lower-star): we 'flood' from low to high values.
    """
    return CubicalComplex(top_dimensional_cells=img01)


# ---------------- Plotting ----------------
def plot_persistence_diagram(H0: np.ndarray, H1: np.ndarray, title: str, out: Path | None):
    # gather points
    d0 = H0 if H0.size else np.empty((0, 2))
    d1 = H1 if H1.size else np.empty((0, 2))

    # choose square bounds
    allv = np.concatenate([d0, d1], axis=0) if d0.size or d1.size else np.array([[0, 0]])
    mn = float(allv.min()) if allv.size else 0.0
    mx = float(allv.max()) if allv.size else 1.0

    plt.figure(figsize=(6, 6))
    if d0.size:
        plt.scatter(d0[:, 0], d0[:, 1], s=12, label="H0 (components)")
    if d1.size:
        plt.scatter(d1[:, 0], d1[:, 1], s=12, label="H1 (loops)")
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(title)
    plt.legend()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()


def plot_barcode(H0: np.ndarray, H1: np.ndarray, title: str, out: Path | None, max_bars: int = 200):
    """
    Simple barcode plot (no external helpers). Limits to max_bars for readability.
    """
    def _plot_family(ax, D: np.ndarray, y0: int, label: str):
        for i, (b, d) in enumerate(D[:max_bars]):
            y = y0 + i
            ax.plot([b, d], [y, y], lw=2)
        ax.text(0.01, y0 + 0.2, label, va="bottom", ha="left")

    # Sort by persistence (longer bars first)
    H0s = H0[np.argsort((H0[:, 1] - H0[:, 0]))[::-1]] if H0.size else H0
    H1s = H1[np.argsort((H1[:, 1] - H1[:, 0]))[::-1]] if H1.size else H1

    fig, ax = plt.subplots(figsize=(10, 5))
    if H0s.size:
        _plot_family(ax, H0s, 0, "H0")
    if H1s.size:
        y0 = (H0s.shape[0] + 2) if H0s.size else 0
        _plot_family(ax, H1s, y0, "H1")
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Bar index")
    ax.set_title(title)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

def plot_sorted_lengths(H0: np.ndarray, H1: np.ndarray, title: str, out: Path | None):
    """Plot sorted persistence lengths for H0 and H1."""
    def lengths(D):
        return (D[:, 1] - D[:, 0]) if D.size else np.array([])

    H0_lengths = lengths(H0)
    H1_lengths = lengths(H1)

    plt.figure(figsize=(8, 5))
    if H0_lengths.size:
        plt.plot(sorted(H0_lengths, reverse=True), label="H0 (components)")
    if H1_lengths.size:
        plt.plot(sorted(H1_lengths, reverse=True), label="H1 (loops)")

    plt.yscale("log")  # log scale helps show noise vs signal
    plt.xlabel("Feature index (sorted)")
    plt.ylabel("Persistence length (death - birth)")
    plt.title(title)
    plt.legend()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="PH on grayscale image via Cubical Complex (GUDHI).")
    ap.add_argument("image", type=Path, help="Path to image (e.g., .png/.jpg/.tif)")
    ap.add_argument("--invert", action="store_true", help="Invert grayscale (use if loops should be bright).")
    ap.add_argument("--no-denoise", action="store_true", help="Disable light Gaussian denoise.")
    ap.add_argument("--clahe", action="store_true", help="Apply local contrast (CLAHE).")
    ap.add_argument("--min-persistence", type=float, default=0.02, help="Filter tiny features (0–1).")
    ap.add_argument("--outdir", type=Path, default=Path("ph_results"))
    args = ap.parse_args()

    img = load_grayscale(args.image)
    img01 = preproc_for_cubical(
        img,
        denoise=not args.no_denoise,
        invert=args.invert,
        clahe=args.clahe,
    )

    cc = build_cubical(img01)
    # Compute persistence; 2 = Z2 field (default, robust). min_persistence filters noise.
    cc.persistence(homology_coeff_field=2, min_persistence=args.min_persistence)

    H0 = cc.persistence_intervals_in_dimension(0)  # components
    H1 = cc.persistence_intervals_in_dimension(1)  # loops (holes)

    # Quick stats
    nH0 = H0.shape[0]
    nH1 = H1.shape[0]
    print(f"Found {nH0} H0 intervals (components), {nH1} H1 intervals (loops) "
          f"with min_persistence >= {args.min_persistence}")

    # Save/plot
    stem = args.image.stem
    plot_persistence_diagram(H0, H1, f"Persistence Diagram — {stem}", args.outdir / f"{stem}_diagram.png")
    plot_barcode(H0, H1, f"Persistence Barcode — {stem}", args.outdir / f"{stem}_barcode.png")
    plot_sorted_lengths(H0, H1, f"Sorted Persistence Lengths — {stem}", args.outdir / f"{stem}_lengths.png")

    # Also dump raw intervals for your report/code
    np.save(args.outdir / f"{stem}_H0.npy", H0)
    np.save(args.outdir / f"{stem}_H1.npy", H1)


if __name__ == "__main__":
    main()
