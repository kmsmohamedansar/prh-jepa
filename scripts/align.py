"""Score every saved WIT-1024 text encoder against DINOv2-B and compare with the paper's numbers."""
import argparse, json, pathlib, platform, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from metrics import (
    bootstrap,
    mutual_knn,
    prh_prep,
    prh_score,
    shuffled_baseline,
    subsample_ci,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Numbers reported by Balestriero and LeCun (2026) in paper/data/llm_scale.csv and paper/data/wit_cka.csv
PAPER_REPORTED = {
    "qwen0.5b": {"checkpoint": "Qwen/Qwen2.5-0.5B", "mutual_knn": 0.102, "unbiased_cka": 0.318},
    "qwen1.5b": {"checkpoint": "Qwen/Qwen2.5-1.5B", "mutual_knn": 0.108, "unbiased_cka": 0.324},
    "qwen3b": {"checkpoint": "Qwen/Qwen2.5-3B", "mutual_knn": 0.125, "unbiased_cka": 0.363},
    "qwen7b": {"checkpoint": "Qwen/Qwen2.5-7B", "mutual_knn": 0.118, "unbiased_cka": 0.353},
    "bert": {"checkpoint": "google-bert/bert-base-uncased", "mutual_knn": 0.111, "unbiased_cka": 0.292},
    "mpnet": {"checkpoint": "sentence-transformers/all-mpnet-base-v2", "mutual_knn": 0.118, "unbiased_cka": 0.390},
    "clip": {"checkpoint": "openai/clip-vit-base-patch32", "mutual_knn": 0.124, "unbiased_cka": 0.339},
}


def load_layers(npz_path):
    with np.load(npz_path) as data:
        return [data[f"layer_{i}"] for i in range(len(data.files))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", default="dinov2", help="reference image encoder key")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--tolerance", type=float, default=0.01)
    args = p.parse_args()

    feats_dir = ROOT / "results" / "features"
    ref_path = feats_dir / f"wit1024-{args.ref}.npz"
    if not ref_path.exists():
        raise SystemExit(f"missing reference features {ref_path}; run scripts/extract.py --model {args.ref} first")
    ref_layers = load_layers(ref_path)

    # Optional cross-check with vendor/prh-proof's PyTorch implementation if present
    vendor_dir = ROOT / "vendor" / "prh-proof"
    has_vendor = vendor_dir.exists()
    if has_vendor:
        sys.path.insert(0, str(vendor_dir))
        import torch
        from prh import metrics as vendor_metrics

    rows = {}
    for npz_file in sorted(feats_dir.glob("wit1024-*.npz")):
        name = npz_file.stem.split("-", 1)[1]
        if name == args.ref:
            continue
        text_layers = load_layers(npz_file)
        mknn, i_best, j_best = prh_score(ref_layers, text_layers, k=args.k, metric="mknn")
        cka, i_cka, j_cka = prh_score(ref_layers, text_layers, k=args.k, metric="cka")

        best_a = prh_prep(ref_layers[i_best])
        best_b = prh_prep(text_layers[j_best])
        ci_boot = bootstrap(best_a, best_b, k=args.k, n=1000, seed=0)
        ci_sub = subsample_ci(best_a, best_b, k=args.k, n=200, size=800, seed=0)
        shuf = shuffled_baseline(best_a, best_b, k=args.k, seed=0)

        # Sensitivity checks: last layer only (with PRH clipping), and best layer without 0.95 clipping
        last_mknn = mutual_knn(prh_prep(ref_layers[-1]), prh_prep(text_layers[-1]), k=args.k)
        unclipped_mknn = max(
            mutual_knn(a, b, k=args.k) for a in ref_layers for b in text_layers
        )

        row = {
            "n_layers_vision": len(ref_layers),
            "n_layers_text": len(text_layers),
            "mutual_knn": round(mknn, 6),
            "mutual_knn_3dp": round(mknn, 3),
            "best_layers_mknn": [int(i_best), int(j_best)],
            "mutual_knn_ci95_bootstrap": [round(ci_boot[0], 4), round(ci_boot[1], 4)],
            "mutual_knn_ci95_subsample": [round(ci_sub[0], 4), round(ci_sub[1], 4)],
            "unbiased_cka": round(cka, 6),
            "unbiased_cka_3dp": round(cka, 3),
            "best_layers_cka": [int(i_cka), int(j_cka)],
            "last_layer_mutual_knn": round(last_mknn, 6),
            "unclipped_max_mutual_knn": round(unclipped_mknn, 6),
            "shuffled": round(shuf, 6),
        }

        if has_vendor:
            t_ref = [torch.from_numpy(x) for x in ref_layers]
            t_txt = [torch.from_numpy(x) for x in text_layers]
            v_mknn, vi, vj = vendor_metrics.prh_score(t_ref, t_txt, k=args.k, metric="mknn")
            v_cka, _, _ = vendor_metrics.prh_score(t_ref, t_txt, k=args.k, metric="cka")
            row["vendor_mutual_knn"] = round(float(v_mknn), 6)
            row["vendor_unbiased_cka"] = round(float(v_cka), 6)
            row["vendor_best_layers_mknn"] = [int(vi), int(vj)]
            row["abs_diff_vs_vendor_mknn"] = round(abs(mknn - float(v_mknn)), 8)
            row["abs_diff_vs_vendor_cka"] = round(abs(cka - float(v_cka)), 8)

        if name in PAPER_REPORTED:
            rep = PAPER_REPORTED[name]
            diff_mknn = mknn - rep["mutual_knn"]
            diff_cka = cka - rep["unbiased_cka"]
            row["paper_reported_mknn"] = rep["mutual_knn"]
            row["paper_reported_cka"] = rep["unbiased_cka"]
            row["diff_vs_paper_mknn"] = round(diff_mknn, 4)
            row["diff_vs_paper_cka"] = round(diff_cka, 4)
            row["reproduced_within_0_01"] = bool(abs(diff_mknn) <= args.tolerance)

        rows[name] = row

    out = {
        "benchmark": "WIT-1024",
        "reference": args.ref,
        "k": args.k,
        "tolerance": args.tolerance,
        "machine": platform.platform(),
        "rows": rows,
    }
    out_path = ROOT / "results" / "phase2_wit1024.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
