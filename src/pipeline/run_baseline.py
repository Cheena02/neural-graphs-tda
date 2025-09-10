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
from src.tda.thresholds import auto_min_persistence

# -----------------------------------------------------------------------------
# CONFIG: add/adjust dataset YAMLs here
# -----------------------------------------------------------------------------
DATASETS = [
    ("mouse", "config/datasets/nucmm_mouse.yaml"),
    ("zebrafish", "config/datasets/nucmm_zebrafish.yaml"),

    ('mousebirn', 'config/datasets/mousebirn.yaml'),
]

# Always resolve results under the project root (…/Dataset Analysis)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "my_results"/"results_raw"#

# ======= Stage controls (baseline stays original) ============================
# Defaults keep baseline exactly as before:
STAGE = "raw"                 # choices: "raw", "preproc", "noise"
GAUSSIAN_SIGMA = 0.0          # <-- 0 means NO denoise in baseline
BORDER_CLAMP_PX = 0          # <-- 0 means NO clamp in baseline

# Noise controls (used only when STAGE="noise")
NOISE_KIND  = "none"          # "none" | "gauss" | "sp" | "blur"
NOISE_LEVEL = 0.0             # e.g., gauss sigma in [0,1]; sp prob in [0,1]; blur sigma>0
NOISE_REPS  = 1               # save independent noisy realizations with suffix _r{1..R}
# ============================================================================

# Choose a single filtration (no lower/upper):
SINGLE_SUPERLEVEL = False   # False=sublevel (recommended). Set True only if you want superlevel.

#  ----------------------------------------------------------------------------
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
        img = load_image(p, cfg.get("color_mode"))
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
        # --- hygiene: gentle denoise & border clamp (recommended) ---

        if GAUSSIAN_SIGMA and GAUSSIAN_SIGMA > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(GAUSSIAN_SIGMA))

        # --- OPTIONAL border clamp (off when BORDER_CLAMP_PX == 0) ---
        b = int(BORDER_CLAMP_PX or 0)
        if b > 0:
            img[:b, :] = 0
            img[-b:, :] = 0
            img[:, :b] = 0
            img[:, -b:] = 0

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
        min_persistence=0.00,  # tweak if needed
        mode="birth-death",  # or "birth-death"
    )
    print(f"[raw] plots: {plots_dir}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for split, yml in DATASETS:
        cfg = load_config(yml)
        # Prefer a YAML name if present; else use the last folder name
        ds_key = cfg.get("name") or Path(cfg["path"]).name or split
        out = RESULTS_ROOT / ds_key / STAGE
        # downsample=0 for true baseline
        process_dataset(yml, out, downsample=0)
