# src/pipeline/compute_ph.py
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import cv2

from src.io.loader import load_config, list_images, load_image
from src.tda.cubical import cubical_diagrams


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


def compute_ph_dataset(config_yaml: str, out_dir: Path, downsample: int = 768, coeff: int = 2) -> int:
    """
    Compute cubical PH for a dataset defined by YAML.
    Saves raw diagrams (.npy) and metrics.csv in out_dir.
    Returns number of rows written to metrics (2 per image: lower & upper).
    """
    cfg = load_config(config_yaml)
    out_dir = Path(out_dir)
    (out_dir / "diagrams").mkdir(parents=True, exist_ok=True)

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

        stem = Path(p).stem
        for tag, superlevel in [("lower", False), ("upper", True)]:
            diags = cubical_diagrams(img, superlevel=superlevel, coeff=coeff)
            h0 = diags.get("H0", np.empty((0, 2), dtype=np.float32))
            h1 = diags.get("H1", np.empty((0, 2), dtype=np.float32))

            # Save raw diagrams
            np.save(out_dir / "diagrams" / f"{stem}_{tag}_H0.npy", h0)
            np.save(out_dir / "diagrams" / f"{stem}_{tag}_H1.npy", h1)

            # Aggregate stats
            s0, s1 = _diag_stats(h0), _diag_stats(h1)
            rows.append({
                "image": p, "filtration": tag,
                "H0_n": s0["n"], "H0_total": s0["total"], "H0_max": s0["max"], "H0_median": s0["median"],
                "H1_n": s1["n"], "H1_total": s1["total"], "H1_max": s1["max"], "H1_median": s1["median"],
            })

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    print(f"[PH] wrote {out_dir / 'metrics.csv'}  (rows={len(rows)})")
    print(f"[PH] diagrams: {out_dir / 'diagrams'}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Compute cubical PH from images (no preprocessing, no plotting).")
    ap.add_argument("--config", required=True, help="dataset YAML (original OR preprocessed)")
    ap.add_argument("--out_dir", required=True, help="destination for metrics.csv + diagrams/*.npy")
    ap.add_argument("--downsample", type=int, default=768)
    ap.add_argument("--coeff", type=int, default=2)
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Resolve out_dir: if relative, anchor at project root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    compute_ph_dataset(args.config, out_dir, downsample=args.downsample, coeff=args.coeff)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if len(sys.argv) == 1:
        # No args → run BOTH preprocessed datasets into my_results/preproc_results/...
        defaults = [
            # (friendly name, config path, output base)
            ("mouse",     "config/datasets/nucmm_mouse_preproc.yaml",     PROJECT_ROOT / "my_results" / "preproc_results" / "nucmm" / "mouse" / "baseline"),
            ("zebrafish", "config/datasets/nucmm_zebrafish_preproc.yaml", PROJECT_ROOT / "my_results" / "preproc_results" / "nucmm" / "zebrafish" / "baseline"),
        ]
        total_rows = 0
        for name, cfg, outdir in defaults:
            try:
                print(f"[PH] Running {name}: cfg={cfg} → {outdir}")
                total_rows += compute_ph_dataset(cfg, outdir, downsample=768, coeff=2)
            except Exception as e:
                print(f"[PH][WARN] {name} failed: {e}")
        print(f"[PH] DONE. Total rows written: {total_rows}")
    else:
        main()
