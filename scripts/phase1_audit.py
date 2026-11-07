"""Phase 1 audit of galilai-group/prh-proof (no training, no model downloads).

1. Checks paper/runs.csv run counts and states.
2. Rebuilds each headline number in the authors' README from paper/data/*.csv.
3. Cross-checks scripts/metrics.py against prh.metrics on shared feature arrays.
Writes results/phase1.json.
"""
import json, pathlib, platform, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "prh-proof"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(VENDOR))

import metrics as our_metrics


def audit_runs_csv():
    df = pd.read_csv(VENDOR / "paper" / "runs.csv")
    states = df["state"].value_counts().to_dict()
    return {
        "total_rows": int(len(df)),
        "states": {str(k): int(v) for k, v in states.items()},
        "unique_tags": int(df["tag"].nunique()),
    }


def audit_headline_numbers():
    data_dir = VENDOR / "paper" / "data"
    scaling = pd.read_csv(data_dir / "scaling.csv")
    phase_real = pd.read_csv(data_dir / "phase_real.csv")
    llm_scale = pd.read_csv(data_dir / "llm_scale.csv")
    wit_cka = pd.read_csv(data_dir / "wit_cka.csv")
    vision_fam = pd.read_csv(data_dir / "vision_families.csv")
    bench_coco = pd.read_csv(data_dir / "bench_coco.csv")
    phase_grid = pd.read_csv(data_dir / "phase_grid.csv")
    phase_pred = pd.read_csv(data_dir / "phase_law_predictions.csv")
    pairfree = pd.read_csv(data_dir / "pairfree.csv")
    video = pd.read_csv(data_dir / "video.csv")

    # 1. 42x sample efficiency claim
    good_2m = scaling.query("objective == 'good' and encoder == 'BERT-S' and data == 'recap' and images_M == 2.16")
    ntp_940m = scaling.query("objective == 'ntp' and encoder == 'BERT-B' and data == 'recap' and images_M == 940.0")
    pr_good = phase_real.query("model == 'BERT-S, 2.16M images' and objective == 'good'").iloc[0]
    pr_ntp = phase_real.query("model == 'BERT-B, 940M images' and objective == 'ntp'").iloc[0]
    sample_ratio = float(pr_ntp["samples"] / pr_good["samples"])
    token_ratio = float(pr_ntp["tokens"] / pr_good["tokens"])

    # 2. 109M text JEPA on WIT-1024 with DINOv2-B (.184) vs LLMs
    good_940m = scaling.query("objective == 'good' and encoder == 'BERT-B' and data == 'recap' and images_M == 940.0")
    wit_940m_seeds = [round(float(x), 4) for x in good_940m["wit"].tolist()]
    wit_940m_mean = round(float(good_940m["wit"].mean()), 4)

    llm_dinov2 = {
        row["model"]: float(row["WIT-1024_mutual-kNN_dinov2"])
        for _, row in llm_scale.iterrows()
        if not str(row["model"]).startswith("ours")
    }
    clip_text_wit = float(wit_cka.query("vision == 'dinov2' and text == 'clip'")["mutual_knn"].iloc[0])

    # 3. CLIP ViT-H on WIT-1024 (.230, range .227-.233)
    clip_h = vision_fam.query("vision == 'clip-h'")
    clip_h_good = clip_h[clip_h["text"].isin(["text-rtall-final", "text-rtall-b64-l01-s1", "text-rtall-b64-l01-s2"])]["wit"]
    clip_h_qwen72b = float(clip_h.query("text == 'qwen72b'")["wit"].iloc[0])
    clip_h_llama70b = float(clip_h.query("text == 'llama3-70b'")["wit"].iloc[0])

    # 4. COCO-5k mutual 10-NN with DINOv2-B (.161)
    coco_d2 = bench_coco.query("bench == 'coco5k' and metric == 'mutual-kNN' and vision == 'dinov2'")
    coco_good = coco_d2[coco_d2["text"].isin(["text-rtall-final", "text-rtall-b64-l01-s1", "text-rtall-b64-l01-s2"])]["score"]
    coco_refs = {
        t: float(coco_d2.query(f"text == {t!r}")["score"].iloc[0])
        for t in ("llama3-70b", "qwen72b", "clip", "mpnet")
    }

    # 5. Scaling law predictions and seed noise
    cells_s0 = (
        phase_grid[phase_grid["seed"] == 0]
        .dropna(subset=["r1"])
        .groupby(["images_M", "samples_M", "q", "design"])["r1"]
        .mean()
        .reset_index()
    )
    reg = phase_pred.set_index(["law", "samples_M", "q"])["predicted_r1"]
    held_v1 = cells_s0[~((cells_s0["design"] == "grid") & (cells_s0["samples_M"] <= 20))]
    pred_v1 = np.array([reg.get(("v1", r.samples_M, r.q), np.nan) for r in held_v1.itertuples()])
    ok_v1 = ~np.isnan(pred_v1)
    rmse_v1_heldout = float(np.sqrt(np.mean((held_v1["r1"].to_numpy()[ok_v1] - pred_v1[ok_v1]) ** 2)))

    held_v2 = cells_s0[cells_s0["design"] != "grid"]
    pred_v2 = np.array([reg.get(("v2", r.samples_M, r.q), np.nan) for r in held_v2.itertuples()])
    ok_v2 = ~np.isnan(pred_v2)
    rmse_v2_heldout = float(np.sqrt(np.mean((held_v2["r1"].to_numpy()[ok_v2] - pred_v2[ok_v2]) ** 2)))

    rand_v2 = cells_s0[cells_s0["design"] == "random"]
    pred_rand_v2 = np.array([reg.get(("v2", r.samples_M, r.q), np.nan) for r in rand_v2.itertuples()])
    rmse_v2_random_85 = float(np.sqrt(np.mean((rand_v2["r1"].to_numpy() - pred_rand_v2) ** 2)))

    # Seed noise across multi-seed grid cells (all 18 pairwise seed differences across 6 3-seed cells)
    multi = phase_grid.pivot_table(index=["files", "samples_M", "q"], columns="seed", values="r1").dropna()
    seed_diffs = np.concatenate([
        (multi[0] - multi[1]).to_numpy(),
        (multi[0] - multi[2]).to_numpy(),
        (multi[1] - multi[2]).to_numpy(),
    ])
    seed_rmse = float(np.sqrt(np.mean(seed_diffs**2)))

    # 6. Pair-free alignment on COCO 2014
    pf_row = pairfree.query(
        "bench == 'coco2014' and method == 'wp' and pairs == 0 and encoder == 'ours, good JEPA, BERT-B, 940M images'"
    ).iloc[0]

    # 7. Video and captioner tables
    vid_rot = video.query("clip == 'mean of 8 frames, --rotation' and decoding == 'consensus32'").iloc[0]
    vid_cap = video.query("clip == 'mean of 8 frames' and decoding == 'consensus32'").iloc[0]

    return {
        "sample_efficiency_42x": {
            "good_2_16m_wit_seeds": [round(float(x), 4) for x in good_2m["wit"].tolist()],
            "good_2_16m_wit_mean": round(float(good_2m["wit"].mean()), 4),
            "ntp_940m_wit_seeds": [round(float(x), 4) for x in ntp_940m["wit"].tolist()],
            "ntp_940m_wit_mean": round(float(ntp_940m["wit"].mean()), 4),
            "samples_good": float(pr_good["samples"]),
            "samples_ntp": float(pr_ntp["samples"]),
            "sample_ratio": round(sample_ratio, 2),
            "tokens_good": int(pr_good["tokens"]),
            "tokens_ntp": int(pr_ntp["tokens"]),
            "token_ratio": round(token_ratio, 2),
            "note": "Sample ratio is 41.67x (900M / 21.6M). Because good-view JEPA reads 3 captions per sample (128.4 vs 42.8 tokens), the token ratio is 13.89x.",
        },
        "wit1024_dinov2_b_headline": {
            "jepa_109m_seeds": wit_940m_seeds,
            "jepa_109m_mean": wit_940m_mean,
            "jepa_109m_mean_3dp": round(wit_940m_mean, 3),
            "llm_baselines": llm_dinov2,
            "clip_text": clip_text_wit,
            "note": "Three-seed mean is 0.1837 (reported as .184). The fourth high-SIGReg seed (.121) is noted in README text and scored against CLIP in vision_families.csv (.145/.150/.157), not listed in scaling.csv.",
        },
        "wit1024_clip_vit_h": {
            "jepa_109m_seeds": [round(float(x), 4) for x in clip_h_good.tolist()],
            "jepa_109m_mean": round(float(clip_h_good.mean()), 4),
            "qwen2_5_72b": clip_h_qwen72b,
            "llama3_70b": clip_h_llama70b,
        },
        "coco5k_dinov2_b": {
            "jepa_109m_seeds": [round(float(x), 4) for x in coco_good.tolist()],
            "jepa_109m_mean": round(float(coco_good.mean()), 4),
            "jepa_109m_mean_3dp": round(float(coco_good.mean()), 3),
            "baselines": coco_refs,
        },
        "scaling_law": {
            "v1_heldout_runs": int(ok_v1.sum()),
            "v1_heldout_rmse": round(rmse_v1_heldout, 4),
            "v2_random_runs": int(len(rand_v2)),
            "v2_random_rmse": round(rmse_v2_random_85, 4),
            "v2_all_heldout_runs": int(ok_v2.sum()),
            "v2_all_heldout_rmse": round(rmse_v2_heldout, 4),
            "seed_noise_rmse": round(seed_rmse, 4),
            "note": "README reports .0093 on the 85 random design runs (v2_random_rmse = 0.0093) and .0096 pairwise seed noise (seed_noise_rmse = 0.0096). Including the 3 320M-sample long extrapolation runs (88 held-out runs total, as tables.py prints) gives RMSE 0.0109.",
        },
        "pairfree_coco2014": {
            "runs": int(pf_row["runs"]),
            "cifar10_mean_pct": round(float(pf_row["CIFAR-10 mean"]) * 100, 2),
            "cifar100_mean_pct": round(float(pf_row["CIFAR-100 mean"]) * 100, 2),
            "imagenet100_mean_pct": round(float(pf_row["ImageNet-100 mean"]) * 100, 2),
            "foscttm_mean": round(float(pf_row["foscttm mean"]), 4),
        },
        "captioner": {
            "msrvtt_cider_rotation": round(float(vid_rot["CIDEr"]), 4),
            "msrvtt_cider_captioner_map": round(float(vid_cap["CIDEr"]), 4),
            "coco_cider_838_in_csv": False,
            "note": "MSR-VTT CIDEr .246 (0.2457) is in video.csv. COCO CIDEr .838 (tab:captioner_scale) is marked 'typed' in docs/reproduce.md and has no CSV in paper/data/.",
        },
    }


def cross_check_metrics():
    import torch
    from prh import metrics as vendor_metrics

    rng = np.random.default_rng(42)
    cases = []
    for idx, (n, da, db, outlier) in enumerate([
        (200, 32, 32, False),
        (256, 64, 48, False),
        (512, 128, 128, True),
        (1024, 64, 96, True),
    ]):
        a = rng.normal(size=(n, da)).astype(np.float32)
        b = rng.normal(size=(n, db)).astype(np.float32)
        # Inject shared signal into half the dimensions
        shared = rng.normal(size=(n, 16)).astype(np.float32)
        a[:, :16] += shared
        b[:, :16] += shared
        if outlier:
            a[0, 0] = 500.0
            b[1, 2] = -400.0

        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        our_mknn = our_metrics.mutual_knn(a, b, k=10)
        their_mknn = float(vendor_metrics.mutual_knn(ta, tb, k=10))
        our_ucka = our_metrics.unbiased_cka(a, b)
        their_ucka = float(vendor_metrics.unbiased_cka(ta, tb))

        # Multi-layer prh_score check (2 layers each)
        la = [a, rng.normal(size=(n, da)).astype(np.float32)]
        lb = [rng.normal(size=(n, db)).astype(np.float32), b]
        our_prh_mknn, oi, oj = our_metrics.prh_score(la, lb, k=10, metric="mknn")
        their_prh_mknn, ti, tj = vendor_metrics.prh_score([torch.from_numpy(x) for x in la], [torch.from_numpy(x) for x in lb], k=10, metric="mknn")
        our_prh_cka, _, _ = our_metrics.prh_score(la, lb, k=10, metric="cka")
        their_prh_cka, _, _ = vendor_metrics.prh_score([torch.from_numpy(x) for x in la], [torch.from_numpy(x) for x in lb], k=10, metric="cka")

        cases.append({
            "case": idx,
            "n": n,
            "outlier": outlier,
            "abs_diff_mutual_knn": abs(our_mknn - their_mknn),
            "abs_diff_unbiased_cka": abs(our_ucka - their_ucka),
            "abs_diff_prh_score_mknn": abs(our_prh_mknn - float(their_prh_mknn)),
            "abs_diff_prh_score_cka": abs(our_prh_cka - float(their_prh_cka)),
            "layers_match": (oi, oj) == (int(ti), int(tj)),
        })

    # Rotation sensitivity check of prh_prep (elementwise 0.95 quantile clipping)
    rng_rot = np.random.default_rng(0)
    a_rot = rng_rot.normal(size=(1024, 64)).astype(np.float32)
    q_rot, _ = np.linalg.qr(rng_rot.normal(size=(64, 64)))
    b_rot = (a_rot @ q_rot).astype(np.float32)
    raw_mknn_rot = our_metrics.mutual_knn(a_rot, b_rot, k=10)
    raw_ucka_rot = our_metrics.unbiased_cka(a_rot, b_rot)
    prh_mknn_rot, _, _ = our_metrics.prh_score([a_rot], [b_rot], k=10, metric="mknn")
    prh_cka_rot, _, _ = our_metrics.prh_score([a_rot], [b_rot], k=10, metric="cka")

    return {
        "n_cases": len(cases),
        "max_abs_diff_mutual_knn": float(max(c["abs_diff_mutual_knn"] for c in cases)),
        "max_abs_diff_unbiased_cka": float(max(c["abs_diff_unbiased_cka"] for c in cases)),
        "max_abs_diff_prh_score_mknn": float(max(c["abs_diff_prh_score_mknn"] for c in cases)),
        "max_abs_diff_prh_score_cka": float(max(c["abs_diff_prh_score_cka"] for c in cases)),
        "all_layers_match": all(c["layers_match"] for c in cases),
        "rotation_invariance_check_n1024_d64": {
            "raw_mutual_knn": round(float(raw_mknn_rot), 4),
            "raw_unbiased_cka": round(float(raw_ucka_rot), 4),
            "prh_score_mknn_after_quantile_clip": round(float(prh_mknn_rot), 4),
            "prh_score_cka_after_quantile_clip": round(float(prh_cka_rot), 4),
            "note": "Elementwise 0.95-quantile clipping in _prh_prep is coordinate-axis dependent and reduces mutual 10-NN on an orthogonal rotation of N=1024, d=64 Gaussian features from 1.0000 to 0.7746 (and unbiased CKA from 1.0000 to 0.9794).",
        },
        "cases": cases,
    }


def main():
    out = {
        "vendor_repo": "galilai-group/prh-proof",
        "vendor_commit": "a2b9949fa907d4fbe562742d4b1ffa7205ab7427",
        "machine": platform.platform(),
        "pytest_suite": {
            "total_collected": 139,
            "not_slow_passed": 133,
            "not_slow_skipped": 3,
            "not_slow_failed": 1,
            "not_slow_seconds": 51.80,
            "slow_passed": 2,
            "slow_seconds": 81.99,
            "skipped_reasons": [
                "tests/test_benchmarks.py::test_msrvtt_extract_frames_decodes_Real_mp4 (optional decord not installed)",
                "tests/test_captioner.py::test_coco_metrics_smoke (optional pycocoevalcap not installed)",
                "tests/test_pairfree.py::test_run_pairfree_dispatches_ot (optional unpaired_rosetta not installed)",
            ],
            "failure_detail": (
                "tests/test_metrics.py::test_unbiased_cka_matches_the_u_statistic_and_its_invariances fails under "
                "pytest 9.2.0 with TypeError: iteration over a 0-d tensor because pytest.approx(reference) iterates "
                "over expected in _contains_decimal; casting float(reference) matches unbiased_cka(a, b) to <1e-16."
            ),
        },
        "runs_csv": audit_runs_csv(),
        "headline_numbers": audit_headline_numbers(),
        "metrics_crosscheck": cross_check_metrics(),
    }
    out_path = ROOT / "results" / "phase1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
