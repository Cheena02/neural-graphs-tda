import os, yaml, pathlib

yamls = [
    "config/datasets/nucmm_mouse.yaml",
    "config/datasets/nucmm_zebrafish.yaml",
    "config/datasets/nucmm_mouse_preproc.yaml",
    "config/datasets/nucmm_zebrafish_preproc.yaml",
    "config/datasets/sample_dataset.yaml",
]

for yml in yamls:
    cfg = yaml.safe_load(open(yml))
    p = os.path.expandvars(cfg["path"])
    print(f"{yml}:")
    print(" ->", p)
    print(" -> Exists:", pathlib.Path(p).exists(), "\n")


