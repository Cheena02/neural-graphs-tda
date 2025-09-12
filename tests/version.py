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
    GUDHI expects an array of function values (2D or 3D). Filtration is sublevel.
    """
    return CubicalComplex(top_dimensional_cells=img01)

# --- NEW: 3D helpers ---
def is_volume_path(p: Path) -> bool:
    return p.suffix.lower() == ".npy"

def load_volume_npy(fp: Path) -> np.ndarray:
    vol = np.load(fp).astype(np.float32)
    # normalize to [0,1]
    vmin, vmax = float(vol.min()), float(vol.max())
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return vol

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
    plt.xlabel("Birth"); plt.ylabel("Death")
    plt.title(title); plt.legend()
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
    ax.set_xlabel("Filtration value"); ax.set_ylabel("Bar index")
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

# --- NEW: multi-dim plots (for 3D: H2) ---
def plot_persistence_diagram_multi(diagrams: dict[int, np.ndarray], title: str, out: Path | None):
    """
    diagrams: {0: H0_array, 1: H1_array, 2: H2_array (optional)}
    """
    colors = {0: "blue", 1: "orange", 2: "green"}
    labels = {0: "H0 (components)", 1: "H1 (loops)", 2: "H2 (voids)"}
    pts = [D.reshape(-1, 2) for D in diagrams.values() if D.size]
    if pts:
        allv = np.concatenate(pts, axis=0)
        mn, mx = float(allv.min()), float(allv.max())
    else:
        mn, mx = 0.0, 1.0

    plt.figure(figsize=(6, 6))
    for dim, D in diagrams.items():
        if D.size:
            plt.scatter(D[:, 0], D[:, 1], s=12, label=labels.get(dim, f"H{dim}"),
                        c=colors.get(dim, None))
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("Birth"); plt.ylabel("Death")
    plt.title(title); plt.legend()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

def plot_barcode_multi(diagrams: dict[int, np.ndarray], title: str, out: Path | None, max_bars: int = 200):
    """
    Simple barcode for H0/H1/H2 if present. Sort each by persistence (desc).
    """
    labels = {0: "H0", 1: "H1", 2: "H2"}
    # sort bars in each dim by length
    sortedD = {}
    for dim, D in diagrams.items():
        if D.size:
            lens = (D[:, 1] - D[:, 0])
            idx = np.argsort(lens)[::-1]
            sortedD[dim] = D[idx][:max_bars]
    # lay them in blocks
    fig, ax = plt.subplots(figsize=(10, 6))
    y = 0
    for dim in sorted(sortedD.keys()):
        D = sortedD[dim]
        start_y = y
        for i, (b, d) in enumerate(D):
            ax.plot([b, d], [y, y], lw=2)
            y += 1
        ax.text(0.01, start_y + 0.2, labels.get(dim, f"H{dim}"), va="bottom")
        y += 2  # gap between blocks
    ax.set_xlabel("Filtration value"); ax.set_ylabel("Bar index")
    ax.set_title(title)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="PH on grayscale 2D image or 3D volume via Cubical Complex (GUDHI).")
    ap.add_argument("image", type=Path, help="Path to 2D image (.png/.jpg/.tif) or 3D volume (.npy)")
    ap.add_argument("--invert", action="store_true", help="(2D only) Invert grayscale (use if loops should be bright).")
    ap.add_argument("--no-denoise", action="store_true", help="(2D only) Disable light Gaussian denoise.")
    ap.add_argument("--clahe", action="store_true", help="(2D only) Apply local contrast (CLAHE).")
    ap.add_argument("--min-persistence", type=float, default=0.02, help="Filter tiny features (0–1).")
    ap.add_argument("--outdir", type=Path, default=Path("ph_results"))
    args = ap.parse_args()

    stem = args.image.stem
    is_vol = is_volume_path(args.image)

    if is_vol:
        data01 = load_volume_npy(args.image)  # 3D array [0,1]
    else:
        img = load_grayscale(args.image)      # 2D array
        data01 = preproc_for_cubical(
            img,
            denoise=not args.no_denoise,
            invert=args.invert,
            clahe=args.clahe,
        )

    # Build Cubical Complex on 2D or 3D; GUDHI handles N-D
    cc = build_cubical(data01)
    cc.persistence(homology_coeff_field=2, min_persistence=args.min_persistence)

    # Collect intervals
    H0 = cc.persistence_intervals_in_dimension(0)
    H1 = cc.persistence_intervals_in_dimension(1)
    diagrams = {0: H0, 1: H1}
    if is_vol:
        H2 = cc.persistence_intervals_in_dimension(2)
        diagrams[2] = H2
        nH2 = H2.shape[0]
    else:
        nH2 = 0

    # Quick stats
    nH0, nH1 = H0.shape[0], H1.shape[0]
    if is_vol:
        print(f"Found {nH0} H0 (components), {nH1} H1 (loops), {nH2} H2 (voids) "
              f"with min_persistence >= {args.min_persistence}")
    else:
        print(f"Found {nH0} H0 (components), {nH1} H1 (loops) "
              f"with min_persistence >= {args.min_persistence}")

    # Save/plot
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / f"{stem}_H0.npy", H0)
    np.save(args.outdir / f"{stem}_H1.npy", H1)
    if is_vol:
        np.save(args.outdir / f"{stem}_H2.npy", diagrams[2])

    # Plots
    if is_vol:
        plot_persistence_diagram_multi(diagrams, f"Persistence Diagram — {stem}",
                                       args.outdir / f"{stem}_diagram.png")
        plot_barcode_multi(diagrams, f"Persistence Barcode — {stem}",
                           args.outdir / f"{stem}_barcode.png")
        # (optional) you can add a sorted-lengths plot for H2 similarly if you want
        plot_sorted_lengths(H0, H1, f"Sorted Persistence Lengths — {stem}",
                            args.outdir / f"{stem}_lengths.png")
    else:
        plot_persistence_diagram(H0, H1, f"Persistence Diagram — {stem}",
                                 args.outdir / f"{stem}_diagram.png")
        plot_barcode(H0, H1, f"Persistence Barcode — {stem}",
                     args.outdir / f"{stem}_barcode.png")
        plot_sorted_lengths(H0, H1, f"Sorted Persistence Lengths — {stem}",
                            args.outdir / f"{stem}_lengths.png")
def run_batch(indir: Path, outdir: Path, min_persistence: float = 0.02,
              invert: bool = False, no_denoise: bool = False, clahe: bool = False):
    """
    Run PH on all supported files in a directory (2D: png/jpg/tif, 3D: npy).
    """
    exts2d = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    exts3d = {".npy"}
    files = [f for f in indir.iterdir() if f.suffix.lower() in (exts2d | exts3d)]
    if not files:
        print(f"No supported images/volumes found in {indir}")
        return

    for f in sorted(files):
        print("="*60)
        print(f"Processing {f.name}")
        stem = f.stem
        is_vol = is_volume_path(f)

        if is_vol:
            data01 = load_volume_npy(f)
        else:
            img = load_grayscale(f)
            data01 = preproc_for_cubical(
                img,
                denoise=not no_denoise,
                invert=invert,
                clahe=clahe,
            )

        cc = build_cubical(data01)
        cc.persistence(homology_coeff_field=2, min_persistence=min_persistence)

        H0 = cc.persistence_intervals_in_dimension(0)
        H1 = cc.persistence_intervals_in_dimension(1)
        diagrams = {0: H0, 1: H1}
        if is_vol:
            H2 = cc.persistence_intervals_in_dimension(2)
            diagrams[2] = H2
            nH2 = H2.shape[0]
        else:
            nH2 = 0

        nH0, nH1 = H0.shape[0], H1.shape[0]
        if is_vol:
            print(f"  H0={nH0}, H1={nH1}, H2={nH2}")
        else:
            print(f"  H0={nH0}, H1={nH1}")

        # Save intervals
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / f"{stem}_H0.npy", H0)
        np.save(outdir / f"{stem}_H1.npy", H1)
        if is_vol:
            np.save(outdir / f"{stem}_H2.npy", diagrams[2])

        # Plots
        if is_vol:
            plot_persistence_diagram_multi(diagrams, f"PD — {stem}", outdir / f"{stem}_diagram.png")
            plot_barcode_multi(diagrams, f"Barcode — {stem}", outdir / f"{stem}_barcode.png")
            plot_sorted_lengths(H0, H1, f"Sorted Lengths (H0/H1) — {stem}", outdir / f"{stem}_lengths.png")
        else:
            plot_persistence_diagram(H0, H1, f"PD — {stem}", outdir / f"{stem}_diagram.png")
            plot_barcode(H0, H1, f"Barcode — {stem}", outdir / f"{stem}_barcode.png")
            plot_sorted_lengths(H0, H1, f"Sorted Lengths — {stem}", outdir / f"{stem}_lengths.png")

if __name__ == "__main__":
    if __name__ == "__main__":
        ap = argparse.ArgumentParser(description="Persistent Homology runner")
        ap.add_argument("path", type=Path, help="Path to a single image/volume OR a folder")
        ap.add_argument("--invert", action="store_true", help="(2D only) Invert grayscale")
        ap.add_argument("--no-denoise", action="store_true", help="(2D only) Disable Gaussian denoise")
        ap.add_argument("--clahe", action="store_true", help="(2D only) Apply CLAHE contrast")
        ap.add_argument("--min-persistence", type=float, default=0.02)
        ap.add_argument("--outdir", type=Path, default=Path("ph_results"))
        args = ap.parse_args()

        if args.path.is_dir():
            run_batch(args.path, args.outdir,
                      min_persistence=args.min_persistence,
                      invert=args.invert,
                      no_denoise=args.no_denoise,
                      clahe=args.clahe)
        else:
            # run the single-file mode (your old main logic)
            main()

