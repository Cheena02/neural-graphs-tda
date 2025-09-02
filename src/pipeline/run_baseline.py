# src/pipeline/run_baseline.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from src.io.loader import load_config, list_images, load_image
from src.tda.cubical import cubical_diagrams
from src.pipeline.plotting_utils import plot_dataset

# -----------------------------------------------------------------------------
# CONFIG: add/adjust dataset YAMLs here
# -----------------------------------------------------------------------------
DATASETS = [
    ("mouse",     "config/datasets/nucmm_mouse.yaml"),
    ("zebrafish", "config/datasets/nucmm_zebrafish.yaml"),
]

# Always resolve results under the project root (…/Dataset Analysis)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "my_results"/"results_raw"   #

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    m = float(img.max()) if img.size else 0.0
    return img / m if m > 0 else img

def _diag_stats(D: np.ndarray) -> dict:
    if D.size == 0:
        return dict(n=0, total=0.0, max=0.0, median=0.0)
    pers = D[:, 1] - D[:, 0]
    return dict(
        n=int(len(pers)),
        total=float(pers.sum()),
        max=float(pers.max()),
        median=float(np.median(pers)),
    )

def _plot_persistence_diagram(diags: dict[str, np.ndarray], out_png: Path, title: str = "Persistence Diagram") -> None:
    """
    Combined H0 (blue) + H1 (red) diagram with y=x diagonal, equal axes, legend.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))

    max_val = 1.0
    if diags.get("H0") is not None and diags["H0"].size:
        plt.scatter(diags["H0"][:, 0], diags["H0"][:, 1], s=18, label="H0")
        max_val = max(max_val, float(diags["H0"].max()))
    if diags.get("H1") is not None and diags["H1"].size:
        plt.scatter(diags["H1"][:, 0], diags["H1"][:, 1], s=18, label="H1")
        max_val = max(max_val, float(diags["H1"].max()))

    # dashed diagonal y=x
    plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)

    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# Choose a single filtration (no lower/upper):
SINGLE_SUPERLEVEL = False   # False=sublevel (recommended). Set True only if you want superlevel.

def process_dataset(ds_yaml: str | os.PathLike, out_dir: str | os.PathLike,
                    downsample: int = 768, coeff: int = 2) -> None:
    cfg = load_config(ds_yaml)
    out_dir = Path(out_dir)
    diag_dir = out_dir / "diagrams"
    plots_dir = out_dir / "plots"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(list_images(cfg)):
        img = load_image(p, cfg.get("color_mode", "grayscale"))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = _normalize01(img)

        if downsample:
            h, w = img.shape
            m = max(h, w)
            if m > downsample:
                s = downsample / m
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        # --- SINGLE filtration only ---
        diags = cubical_diagrams(img, superlevel=SINGLE_SUPERLEVEL, coeff=coeff)  # {"H0": Nx2, "H1": Mx2}
        h0 = diags.get("H0", np.empty((0, 2), dtype=np.float32))
        h1 = diags.get("H1", np.empty((0, 2), dtype=np.float32))

        stem = Path(p).stem
        # Save raw arrays (NO lower/upper suffixes)
        np.save(diag_dir / f"{stem}_H0.npy", h0)
        np.save(diag_dir / f"{stem}_H1.npy", h1)

        # Aggregate stats (one row per image)
        s0 = _diag_stats(h0); s1 = _diag_stats(h1)
        rows.append({
            "image": p, "filtration": "single",
            "H0_n": s0["n"], "H0_total": s0["total"], "H0_max": s0["max"], "H0_median": s0["median"],
            "H1_n": s1["n"], "H1_total": s1["total"], "H1_max": s1["max"], "H1_median": s1["median"],
        })

    # Write metrics
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    print(f"[raw] wrote {out_dir / 'metrics.csv'}  (rows={len(rows)})")
    print(f"[raw] diagrams: {diag_dir}")

    # --- NEW: auto-generate plots via plotting_utils into plots/{pd,barcodes} ---
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_dataset(
        diagrams_dir=out_dir / "diagrams",
        out_dir=plots_dir,
        min_persistence=0.05,  # tweak if needed
        mode="birth-death",  # or "birth-death"
    )
    print(f"[raw] plots: {plots_dir}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for split, yml in DATASETS:
        out = RESULTS_ROOT / "nucmm" / split / "raw"
        process_dataset(yml, out)
