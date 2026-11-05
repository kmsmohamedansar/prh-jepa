"""Score every saved image encoder against the language model."""
import argparse, json, pathlib, platform, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from metrics import mutual_knn, linear_cka, shuffled_baseline, bootstrap

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "heldout"], required=True)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    feats = ROOT / "results" / "features"
    lm = np.load(feats / f"{args.split}-lm.npy")
    rows = {}
    for f in sorted(feats.glob(f"{args.split}-*.npy")):
        name = f.stem.split("-", 1)[1]
        if name == "lm":
            continue
        x = np.load(f)
        rows[name] = {
            "mutual_knn": mutual_knn(x, lm, args.k),
            "mutual_knn_ci95": bootstrap(x, lm, lambda a, b: mutual_knn(a, b, args.k)),
            "linear_cka": linear_cka(x, lm),
            "shuffled": shuffled_baseline(x, lm, args.k),
        }
    out = {"split": args.split, "k": args.k, "machine": platform.platform(), "rows": rows}
    (ROOT / "results" / f"{args.split}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
