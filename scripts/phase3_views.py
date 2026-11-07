"""Phase 3: Test good views against bad views in both exact finite worlds and small neural LeJEPA runs.

1. Finite-world spectral verification (using prh.theory):
   - Theorem 3 product worlds (n=16 states, m=6 nuisances, k=5, 20 seeds): compares good views vs bad views on
     nuisance directions, the Theorem 3 lower bound, and cross-modal error.
   - Theorem 4 sample-density curves (5 classes x 6 states, k=4, 10 seeds): compares good views vs bad views
     across samples per state (2..64) when classes are separated (gap=0.25) vs adjacent (gap=0.0).
2. Controlled neural LeJEPA check on multi-caption scenes (yerevann/coco-karpathy train split, 82,783 scenes x 5 captions):
   - Trains the paper's TextEncoder (arch='tiny', 12 layers, width 192, ~11M params) with LeJEPA (lamb=0.02)
     for 1,000 steps (batch_size=128, 2 global + 2 local views, max_len=48) across 3 seeds (0, 1, 2).
   - Everything is held fixed except the view constructor:
       * good views: each of the 4 views draws an independent human caption of the same scene
       * bad views: all 4 views draw masked spans from a single randomly chosen caption of the scene
   - Evaluates on WIT-1024 against DINOv2-B at steps 0 (untrained floor), 500, and 1,000 using scripts/metrics.py.
"""
import argparse, gc, json, math, pathlib, platform, resource, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "prh-proof"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(VENDOR))

from metrics import bootstrap, mutual_knn, prh_prep, prh_score


def run_finite_world_checks():
    from prh import theory

    # 1. Theorem 3 check: product world (state z, nuisance nu) across 20 seeds
    n_states, n_nuis, k = 16, 6, 5
    thm3_cases = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pi, p = theory.random_world(n_states, rng, blocks=4, leak=0.05)
        q1 = rng.dirichlet(np.ones(n_nuis))
        q2 = rng.dirichlet(np.ones(n_nuis))

        # Reference second modality with good views
        w_ref = theory.good_views(pi, q2)
        g_ref = theory.whiten(theory.spectral_solution(w_ref, k), np.kron(p, q2))

        # Modality 1 with good views
        w_good = theory.good_views(pi, q1)
        f_good = theory.whiten(theory.spectral_solution(w_good, k), np.kron(p, q1))
        bound_good = theory.nuisance_bound(f_good, p, q1)
        err_good = theory.cross_modal_error(f_good, g_ref, p, q1, q2)

        # Modality 1 with bad views: corruption keeps state with weight 0.85, nuisance with weight 0.92
        a_corrupt = 0.85 * np.eye(n_states) + 0.15 * np.broadcast_to(p, (n_states, n_states))
        b_corrupt = 0.92 * np.eye(n_nuis) + 0.08 * np.broadcast_to(q1, (n_nuis, n_nuis))
        w_bad = theory.bad_views(p, q1, a_corrupt, b_corrupt)
        f_bad = theory.whiten(theory.spectral_solution(w_bad, k), np.kron(p, q1))
        a_pairs = a_corrupt.T @ (p[:, None] * a_corrupt)
        b_pairs = b_corrupt.T @ (q1[:, None] * b_corrupt)
        r_nuis = theory.nuisance_directions(a_pairs, b_pairs, k)
        bound_bad = theory.nuisance_bound(f_bad, p, q1)
        err_bad = theory.cross_modal_error(f_bad, g_ref, p, q1, q2)

        thm3_cases.append({
            "seed": seed,
            "good_bound": round(bound_good, 8),
            "good_cross_error": round(err_good, 8),
            "bad_nuisance_dirs": int(r_nuis),
            "bad_bound": round(bound_bad, 6),
            "bad_cross_error": round(err_bad, 6),
            "bound_holds": bool(err_bad + 1e-9 >= bound_bad),
        })

    # 2. Theorem 4 / sample-density check (line_alignment, matching figures.py theory)
    densities = [1, 2, 4, 8, 16, 32, 64]
    line_curves = {}
    for label, gap, reach in [
        ("good_literal_gap050", 0.5, "good"),
        ("bad_r002_literal_gap050", 0.5, 0.02),
        ("bad_r002_mixed_gap000", 0.0, 0.02),
        ("bad_r012_mixed_gap000", 0.0, 0.12),
    ]:
        pts = {}
        for ps in densities:
            vals = [theory.line_alignment(ps, gap=gap, reach=reach, seed=s) for s in range(10)]
            pts[str(ps)] = {
                "mean": round(float(np.mean(vals)), 4),
                "sd": round(float(np.std(vals, ddof=1)), 4),
            }
        line_curves[label] = pts

    return {
        "theorem3_product_worlds": {
            "n_worlds": len(thm3_cases),
            "n_states": n_states,
            "n_nuisances": n_nuis,
            "k": k,
            "good_cross_error_max": float(max(c["good_cross_error"] for c in thm3_cases)),
            "bad_nuisance_dirs_mean": round(float(np.mean([c["bad_nuisance_dirs"] for c in thm3_cases])), 2),
            "bad_bound_mean": round(float(np.mean([c["bad_bound"] for c in thm3_cases])), 4),
            "bad_cross_error_mean": round(float(np.mean([c["bad_cross_error"] for c in thm3_cases])), 4),
            "bound_holds_all": all(c["bound_holds"] for c in thm3_cases),
        },
        "theorem4_density_curves": line_curves,
    }


def _eval_text_encoder_on_wit(enc, tok, wit_texts, ref_layers, max_len, device, batch_size=64):
    import torch
    from prh.views import TextView

    enc.eval()
    eval_view = TextView(tok, max_len=max_len)
    all_ids = torch.stack([
        eval_view({"text": torch.tensor(tok(t, add_special_tokens=False)["input_ids"], dtype=torch.long)})["text"]
        for t in wit_texts
    ])
    layer_chunks = []
    cls_chunks = []
    with torch.no_grad():
        for i in range(0, len(all_ids), batch_size):
            ids = all_ids[i : i + batch_size].to(device)
            mask = ids.ne(enc.pad_id)
            out = enc.bert(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            m = mask.unsqueeze(-1).float()
            hs_mean = [((h.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy().astype(np.float32) for h in out.hidden_states]
            layer_chunks.append(hs_mean)
            cls_chunks.append(out.last_hidden_state[:, 0].float().cpu().numpy().astype(np.float32))
            del out, ids, mask
    text_layers = [np.concatenate([c[i] for c in layer_chunks], axis=0) for i in range(len(layer_chunks[0]))]
    cls_final = np.concatenate(cls_chunks, axis=0)

    mknn, ib, jb = prh_score(ref_layers, text_layers, k=10, metric="mknn")
    cka, ic, jc = prh_score(ref_layers, text_layers, k=10, metric="cka")
    cls_mknn = mutual_knn(prh_prep(ref_layers[-1]), prh_prep(cls_final), k=10)
    ci = bootstrap(prh_prep(ref_layers[ib]), prh_prep(text_layers[jb]), k=10, n=1000, seed=0)
    return {
        "mutual_knn": round(float(mknn), 6),
        "mutual_knn_ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "best_layers_mknn": [int(ib), int(jb)],
        "unbiased_cka": round(float(cka), 6),
        "cls_last_layer_mknn": round(float(cls_mknn), 6),
    }


def _sample_batch_views(tokenized_scenes, batch_indices, rng, good_views, n_global, n_local, max_len, cls_id, sep_id, mask_id, pad_id, gscale=(0.5, 1.0), lscale=(0.15, 0.5), mask_ratio=0.15):
    import torch

    n_views = n_global + n_local
    scales = [gscale] * n_global + [lscale] * n_local
    bs = len(batch_indices)
    buf = np.full((n_views, bs, max_len), pad_id, dtype=np.int64)
    buf[:, :, 0] = cls_id

    for row_idx, scene_idx in enumerate(batch_indices):
        caps = tokenized_scenes[scene_idx]
        n_c = len(caps)
        if not good_views:
            chosen = caps[int(rng.integers(n_c))]
        for v_idx, (lo, hi) in enumerate(scales):
            ids = caps[int(rng.integers(n_c))] if good_views else chosen
            ids = ids[: max_len - 2]
            L = len(ids)
            if L > 0:
                keep = max(1, round(L * (lo + (hi - lo) * float(rng.random()))))
                start = int(rng.integers(L - keep + 1)) if L > keep else 0
                span = ids[start : start + keep].copy()
                span[rng.random(keep) < mask_ratio] = mask_id
                buf[v_idx, row_idx, 1 : 1 + keep] = span
                buf[v_idx, row_idx, 1 + keep] = sep_id
            else:
                buf[v_idx, row_idx, 1] = sep_id
    return [torch.from_numpy(buf[v]) for v in range(n_views)]


def run_neural_jepa_check(steps=400, batch_size=128, seeds=(0, 1, 2), max_len=36, arch="tiny", lr=5e-4, lamb=0.02, cleanup_cache=True):
    import datasets, torch
    from huggingface_hub import scan_cache_dir
    from stable_pretraining.methods.lejepa import LeJEPA
    from transformers import AutoTokenizer
    from prh.models import TOKENIZER, TextEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ref_npz = ROOT / "results" / "features" / "wit1024-dinov2.npz"
    if not ref_npz.exists():
        raise SystemExit(f"missing {ref_npz}; run Phase 2 dinov2 extraction first")
    with np.load(ref_npz) as data:
        ref_layers = [data[f"layer_{i}"] for i in range(len(data.files))]

    wit_ds = datasets.load_dataset("minhuh/prh", revision="wit_1024", split="train")
    wit_texts = [t[0] for t in wit_ds.select_columns(["text"])["text"]]
    del wit_ds

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    coco_ds = datasets.load_dataset("yerevann/coco-karpathy", split="train")
    raw_sentences = coco_ds.select_columns(["sentences"])["sentences"]
    del coco_ds

    # Pre-tokenize all 82,783 x 5 captions once into compact 1D NumPy arrays
    flat_texts = [c or "" for caps in raw_sentences for c in caps]
    lengths = [len(caps) for caps in raw_sentences]
    del raw_sentences
    flat_ids = tok(flat_texts, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    del flat_texts
    tokenized_scenes = []
    pos = 0
    for n_caps in lengths:
        scene_caps = [np.asarray(flat_ids[pos + k], dtype=np.int64) for k in range(n_caps) if len(flat_ids[pos + k]) > 0]
        if scene_caps:
            tokenized_scenes.append(scene_caps)
        pos += n_caps
    del flat_ids

    n_scenes = len(tokenized_scenes)
    n_global, n_local = 2, 2
    eval_steps = sorted({0, steps // 2, steps})
    runs = []

    for seed in seeds:
        # Evaluate untrained initialization once per seed (identical for good and bad arms)
        torch.manual_seed(seed)
        init_enc = TextEncoder(tok.pad_token_id, arch=arch).to(device)
        init_metrics = _eval_text_encoder_on_wit(init_enc, tok, wit_texts, ref_layers, max_len, device)
        n_params = sum(p.numel() for p in init_enc.parameters())
        del init_enc
        if device == "mps":
            torch.mps.empty_cache()

        for arm in ("good", "bad"):
            good_flag = arm == "good"
            torch.manual_seed(seed)
            jepa = LeJEPA(f"vit_{arch}_patch16_224", lamb=lamb)
            jepa.backbone = TextEncoder(tok.pad_token_id, arch=arch)
            # Keep SlicedEppsPulley on CPU: PyTorch 2.15's torch.Generator(device='mps') deadlocks inside SlicedEppsPulley.forward
            sigreg_cpu = jepa.sigreg.to("cpu")
            backbone = jepa.backbone.to(device)
            projector = jepa.projector.to(device)

            params = list(backbone.parameters()) + list(projector.parameters())
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05, betas=(0.9, 0.999))
            warmup = max(1, int(0.05 * steps))
            rng = np.random.default_rng(seed)

            t0 = time.perf_counter()
            checkpoints = {"0": init_metrics}
            losses = []

            for step in range(1, steps + 1):
                # Linear warmup + cosine decay matching train.py
                if step <= warmup:
                    cur_lr = lr * (0.01 + 0.99 * step / warmup)
                else:
                    prog = (step - warmup) / max(1, steps - warmup)
                    cur_lr = (lr / 1000) + 0.5 * (lr - lr / 1000) * (1.0 + math.cos(math.pi * prog))
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr

                backbone.train()
                projector.train()
                batch_idx = rng.integers(0, n_scenes, size=batch_size).tolist()
                views = _sample_batch_views(
                    tokenized_scenes,
                    batch_idx,
                    rng=rng,
                    good_views=good_flag,
                    n_global=n_global,
                    n_local=n_local,
                    max_len=max_len,
                    cls_id=tok.cls_token_id,
                    sep_id=tok.sep_token_id,
                    mask_id=tok.mask_token_id,
                    pad_id=tok.pad_token_id,
                )
                views_dev = [v.to(device) for v in views]
                opt.zero_grad(set_to_none=True)
                g_feat = backbone(torch.cat(views_dev[:n_global]))
                l_feat = backbone(torch.cat(views_dev[n_global:]))
                all_proj = projector(torch.cat([g_feat, l_feat])).view(n_global + n_local, batch_size, -1)
                centers = all_proj[:n_global].mean(0, keepdim=True)
                inv_loss = (centers - all_proj).square().mean()
                sig_loss = sigreg_cpu(all_proj.reshape(-1, all_proj.size(-1)).cpu()).to(device)
                loss = inv_loss + lamb * sig_loss
                loss.backward()
                opt.step()

                if step % 100 == 0 or step == 1:
                    l_val = round(float(loss.detach().cpu()), 5)
                    inv_val = round(float(inv_loss.detach().cpu()), 5)
                    sig_val = round(float(sig_loss.detach().cpu()), 5)
                    losses.append({
                        "step": step,
                        "loss": l_val,
                        "inv_loss": inv_val,
                        "sigreg_loss": sig_val,
                    })
                    print(f"  [{arm} seed={seed}] step {step}/{steps}: loss={l_val:.4f} (inv={inv_val:.4f}, sig={sig_val:.2f})", flush=True)
                del loss, inv_loss, sig_loss, all_proj, g_feat, l_feat, views_dev, views

                if step in eval_steps:
                    if device == "mps":
                        torch.mps.empty_cache()
                    checkpoints[str(step)] = _eval_text_encoder_on_wit(
                        backbone, tok, wit_texts, ref_layers, max_len, device
                    )

            elapsed = round(time.perf_counter() - t0, 2)
            run_rec = {
                "arm": arm,
                "seed": int(seed),
                "arch": arch,
                "params": int(n_params),
                "steps": int(steps),
                "batch_size": int(batch_size),
                "scenes_seen": int(steps * batch_size),
                "views_per_scene": n_global + n_local,
                "seconds": elapsed,
                "checkpoints": checkpoints,
                "loss_trace": losses,
            }
            runs.append(run_rec)
            print(
                f"[{arm} seed={seed}] step 0 mknn={checkpoints['0']['mutual_knn']:.4f} -> "
                f"step {steps} mknn={checkpoints[str(steps)]['mutual_knn']:.4f} "
                f"(cka={checkpoints[str(steps)]['unbiased_cka']:.4f}, {elapsed:.1f}s)",
                flush=True,
            )
            del backbone, projector, sigreg_cpu, jepa, opt
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

    if cleanup_cache:
        info = scan_cache_dir()
        revs = [rev.commit_hash for r in info.repos if r.repo_id == "yerevann/coco-karpathy" for rev in r.revisions]
        if revs:
            info.delete_revisions(*revs).execute()

    good_final = [r["checkpoints"][str(steps)]["mutual_knn"] for r in runs if r["arm"] == "good"]
    bad_final = [r["checkpoints"][str(steps)]["mutual_knn"] for r in runs if r["arm"] == "bad"]
    init_vals = [r["checkpoints"]["0"]["mutual_knn"] for r in runs if r["arm"] == "good"]
    good_cka = [r["checkpoints"][str(steps)]["unbiased_cka"] for r in runs if r["arm"] == "good"]
    bad_cka = [r["checkpoints"][str(steps)]["unbiased_cka"] for r in runs if r["arm"] == "bad"]

    return {
        "corpus": "yerevann/coco-karpathy (train split, 5 human captions per scene)",
        "n_scenes_corpus": n_scenes,
        "arch": arch,
        "params": runs[0]["params"],
        "steps": steps,
        "batch_size": batch_size,
        "views": [n_global, n_local],
        "max_len": max_len,
        "lamb": lamb,
        "lr": lr,
        "device": device,
        "summary": {
            "init_mknn_seeds": [round(x, 4) for x in init_vals],
            "init_mknn_mean": round(float(np.mean(init_vals)), 4),
            "bad_mknn_seeds": [round(x, 4) for x in bad_final],
            "bad_mknn_mean": round(float(np.mean(bad_final)), 4),
            "bad_mknn_sd": round(float(np.std(bad_final, ddof=1)), 4) if len(bad_final) > 1 else 0.0,
            "good_mknn_seeds": [round(x, 4) for x in good_final],
            "good_mknn_mean": round(float(np.mean(good_final)), 4),
            "good_mknn_sd": round(float(np.std(good_final, ddof=1)), 4) if len(good_final) > 1 else 0.0,
            "good_minus_bad_seeds": [round(g - b, 4) for g, b in zip(good_final, bad_final)],
            "good_minus_bad_mean": round(float(np.mean(np.array(good_final) - np.array(bad_final))), 4),
            "bad_cka_mean": round(float(np.mean(bad_cka)), 4),
            "good_cka_mean": round(float(np.mean(good_cka)), 4),
            "good_beats_bad_every_seed": all(g > b for g, b in zip(good_final, bad_final)),
        },
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--update-finite-world-only", action="store_true", help="recompute only finite_world in existing phase3_views.json")
    args = p.parse_args()

    out_path = ROOT / "results" / "phase3_views.json"
    if args.update_finite_world_only and out_path.exists():
        out = json.loads(out_path.read_text())
        out["finite_world"] = run_finite_world_checks()
        runs = out["neural_jepa"]["runs"]
        st = str(out["neural_jepa"]["steps"])
        init_cka = [r["checkpoints"]["0"]["unbiased_cka"] for r in runs if r["arm"] == "good"]
        good_cka = [r["checkpoints"][st]["unbiased_cka"] for r in runs if r["arm"] == "good"]
        bad_cka = [r["checkpoints"][st]["unbiased_cka"] for r in runs if r["arm"] == "bad"]
        out["neural_jepa"]["summary"]["init_cka_mean"] = round(float(np.mean(init_cka)), 4)
        out["neural_jepa"]["summary"]["good_cka_seeds"] = [round(x, 4) for x in good_cka]
        out["neural_jepa"]["summary"]["bad_cka_seeds"] = [round(x, 4) for x in bad_cka]
        out["neural_jepa"]["summary"]["good_beats_bad_cka_every_seed"] = all(g > b for g, b in zip(good_cka, bad_cka))
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out["finite_world"]["theorem4_density_curves"], indent=2))
        return

    t0 = time.perf_counter()
    finite_world = run_finite_world_checks()
    neural = run_neural_jepa_check(steps=args.steps, batch_size=args.batch_size, seeds=tuple(args.seeds))
    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)

    out = {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "total_seconds": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": peak_rss_mb,
        "finite_world": finite_world,
        "neural_jepa": neural,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["finite_world"]["theorem3_product_worlds"], indent=2))
    print(json.dumps(out["neural_jepa"]["summary"], indent=2))


if __name__ == "__main__":
    main()
