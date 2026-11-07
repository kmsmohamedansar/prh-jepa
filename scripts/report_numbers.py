"""Rebuild paper/numbers.json from results/phase1.json, results/phase2_wit1024.json,
results/phase3_views.json, and results/raw/extract-*.json.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "numbers.json"


def main():
    p1 = json.loads((RESULTS / "phase1.json").read_text())
    p2 = json.loads((RESULTS / "phase2_wit1024.json").read_text())
    p3_path = RESULTS / "phase3_views.json"
    p3 = json.loads(p3_path.read_text()) if p3_path.exists() else None

    extractions = {}
    for p in sorted((RESULTS / "raw").glob("extract-*.json")):
        meta = json.loads(p.read_text())
        extractions[meta["model_key"]] = {
            "checkpoint": meta["checkpoint"],
            "modality": meta["modality"],
            "device": meta["device"],
            "dtype": meta["dtype"],
            "batch_size": meta["batch_size"],
            "n_items": meta["n_items"],
            "n_layers": meta["n_layers"],
            "dim": meta["dim"],
            "seconds": meta["seconds"],
            "peak_rss_mb": meta.get("peak_rss_mb"),
            "cleaned_cache_mb": meta.get("cleaned_cache_mb", 0.0),
        }

    p2_rows = p2["rows"]
    diffs_mknn = [abs(r["diff_vs_paper_mknn"]) for r in p2_rows.values() if "diff_vs_paper_mknn" in r]
    diffs_cka = [abs(r["diff_vs_paper_cka"]) for r in p2_rows.values() if "diff_vs_paper_cka" in r]
    clip_drops = {
        k: round(r["mutual_knn"] - r["unclipped_max_mutual_knn"], 4)
        for k, r in p2_rows.items()
    }
    layer_drops = {
        k: round(r["mutual_knn"] - r["last_layer_mutual_knn"], 4)
        for k, r in p2_rows.items()
    }

    numbers = {
        "environment": {
            "machine": p1["machine"],
            "chip_and_ram_reported": "Apple M4, 16 GB unified memory",
            "vendor_repo": p1["vendor_repo"],
            "vendor_commit": p1["vendor_commit"],
            "stable_pretraining_commit": "ab836bf699a2be712dfcf980a5eb70d36391e876",
            "versions": next(iter((RESULTS / "raw").glob("extract-*.json")), None)
            and json.loads(next(iter((RESULTS / "raw").glob("extract-*.json"))).read_text())["versions"],
        },
        "phase1": p1,
        "phase2": {
            "benchmark": p2["benchmark"],
            "reference": p2["reference"],
            "k": p2["k"],
            "tolerance": p2["tolerance"],
            "n_models_tested": len(p2_rows),
            "n_models_reproduced": sum(1 for r in p2_rows.values() if r.get("reproduced_within_0_01")),
            "max_abs_diff_vs_paper_mknn": round(max(diffs_mknn), 4),
            "max_abs_diff_vs_paper_cka": round(max(diffs_cka), 4),
            "max_abs_diff_vs_vendor_mknn": round(max(r["abs_diff_vs_vendor_mknn"] for r in p2_rows.values()), 6),
            "max_abs_diff_vs_vendor_cka": round(max(r["abs_diff_vs_vendor_cka"] for r in p2_rows.values()), 8),
            "clip_benefit_mknn": clip_drops,
            "best_minus_last_layer_mknn": layer_drops,
            "extractions": extractions,
            "rows": p2_rows,
        },
        "phase3": p3,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(numbers, indent=2))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
