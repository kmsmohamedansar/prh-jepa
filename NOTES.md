# Build notes

Dated log of what was done, what broke and what was measured.

## 2026-11-05

- Started as an original-paper idea ("do JEPAs converge faster?"). Found the same question already answered, under the same title, by Balestriero and LeCun (`galilai-group/prh-proof`, released 2026-10-30, MIT). Turned the repo into an independent check of that paper, like `decision-models` and `rea-study`.
- Their repo, read today: 345 runs, about 5,500 GPU-hours; CPU tests; per-figure reproduction guide in `docs/reproduce.md`. No trained checkpoints found linked in the README; a demo on Hugging Face.
- `scripts/metrics.py`: my own mutual k-NN, CKA, shuffled baseline and bootstrap, written separately from theirs. 5 tests pass on synthetic data.
- `scripts/extract.py`, `scripts/align.py`: written, not run on real models.
- Wrote `docs/explainer.html`: the paper in plain words with drawings, for readers new to it. Its numbers are the authors' reported ones, not ours.

### Unparked and repaired before first run (2026-11-05)

- Working on the owner's Mac (`Darwin 25.0.0 arm64`, Apple M4, 16 GB RAM).
- Cloned `galilai-group/prh-proof` into `vendor/prh-proof` (gitignored) and pinned commit `a2b9949fa907d4fbe562742d4b1ffa7205ab7427` (with `galilai-group/stable-pretraining` at `ab836bf699a2be712dfcf980a5eb70d36391e876`). Read `AGENTS.md`, `docs/design.md` and `docs/reproduce.md`.
- Searched the web for independent replications or critiques of `arXiv:2610.39302` (`galilai-group/prh-proof`); found summaries of the paper and repo, and no independent replication yet.
- Read and recorded the licences and 16-hex SHA-256 fingerprints of `prh-proof`, `stable-pretraining`, `WIT-1024` (`minhuh/prh` / `google-research-datasets/wit`), `DINOv2`, `Qwen2.5` (0.5B, 1.5B, 3B, 7B; noting that 3B uses the Qwen Research License while 0.5B, 1.5B and 7B use Apache 2.0), `bert-base-uncased`, `all-mpnet-base-v2` and `clip-vit-base-patch32` in `docs/licences.md`.
- Repaired three code issues before any run:
  1. `bootstrap` in `scripts/metrics.py` previously used `np.unique(rng.integers(0, len(a), len(a)))`, which shrank the sample pool to about 63.2% of `N` on every draw and biased fixed-`k` nearest-neighbour overlap upward. Replaced it with resampling the `N` per-item overlap values (`per_item_mutual_knn`) with replacement on the full `N`-item neighbourhood graph, plus `subsample_ci` (subsets without replacement rescaled by `sqrt(size / (N - size))`, matching `prh.metrics.subsample_interval`).
  2. Added `unbiased_cka` (Song et al. 2012 unbiased HSIC estimator), `prh_prep` (0.95-quantile elementwise clipping and L2 normalisation) and `prh_score` (maximum over all layer pairs) to `scripts/metrics.py`. While writing unit tests in `tests/test_metrics.py`, found that `prh_prep`'s elementwise coordinate clipping is not invariant to orthogonal rotations: on a 200x32 Gaussian matrix `A` and random rotation `A @ Q`, unclipped `mutual_knn` is `1.000`, whereas `prh_score` after `prh_prep` drops to `0.836`.
  3. Updated `scripts/extract.py` and `scripts/align.py` from the old multi-image-encoder scaffold to the Phase 2 WIT-1024 benchmark (`DINOv2-B` against 7 open text encoders), and wrote `scripts/phase1_audit.py` to check `paper/runs.csv`, `paper/data/*.csv` and metric agreement.
- Froze `docs/analysis-plan.md` with a dated commit before running Phase 1 or Phase 2.

## 2026-11-06

### Phase 1: upstream test suite, CSV headline audit and metric cross-check

- Note on machine OS string: Python `platform.platform()` in our run JSON records `macOS-27.0.1-arm64-arm-64bit` (`Darwin 26.0.1`; the `Darwin 25.0.0` string on 2026-11-05 was typed before running `platform.platform()`).
- Ran `scripts/phase1_audit.py` and `pytest -m "slow"` on `vendor/prh-proof` unchanged (`results/phase1.json`, `results/raw/phase1_pytest_slow.txt`):
  - `139` tests collected: `133` passed in `-m "not slow"` (`51.80 s`), `2` passed in `-m "slow"` (`81.99 s`), `3` skipped for optional packages (`decord`, `pycocoevalcap`, `unpaired_rosetta`), and `1` failed (`test_unbiased_cka_matches_the_u_statistic_and_its_invariances`).
  - Inspected the single failure without editing upstream code: under `pytest 9.2.0`, `pytest.approx(reference)` calls `isinstance(expected, Iterable)` inside `_contains_decimal`, which raises `TypeError: iteration over a 0-d tensor` on a 0-d PyTorch tensor. Casting `float(reference)` matches `unbiased_cka(a, b)` to `< 1e-16`.
  - Audited `paper/runs.csv` (`345` unique tags: `321` `completed`, `18` `cancelled`, `6` `failed`) and rebuilt the README headline numbers from `paper/data/*.csv`:
    - `42x fewer samples` (`BERT-S` good views at `21.6M` samples, WIT mean `0.1080` vs `BERT-B` next-token at `900M` samples, WIT mean `0.1107`) is `41.67x` in samples and `13.89x` in tokens (`38.52B / 2.77B`), because each good-view sample reads 3 captions (`128.4` vs `42.8` tokens in `phase_real.csv`).
    - Held-out scaling-law RMSE is `0.0093` on the `85` `design == "random"` runs (beside `0.0096` pairwise seed noise) and `0.0109` across all `88` non-grid held-out runs when the `3` `320M`-sample `design == "long"` extrapolation runs are included.
    - All other README headlines rebuild from `paper/data/*.csv` except COCO captioner CIDEr `.838` (`tab:captioner_scale`), which `docs/reproduce.md` explicitly marks `typed` with no CSV in `paper/data/`.
  - Cross-checked `scripts/metrics.py` against `prh.metrics`: max difference `5.3e-9` (`mutual_knn`) and `6.1e-9` (`prh_score` CKA) on synthetic/outlier arrays, and confirmed on `N = 1024, d = 64` Gaussian features that an exact orthogonal rotation keeps unclipped `mutual_knn` and `unbiased_cka` at `1.0000` while dropping `prh_score` after `prh_prep` to `0.7746` (`0.9794` for CKA).

### Phase 2: 19 GB memory spike, `qwen7b` disk skip, and WIT-1024 baseline runs

- **Memory spike and repair (`task-188`):** First run of `scripts/extract.py --model dinov2` hit roughly `19 GB` of virtual memory and macOS disk swap because `list(ds["image"])` decoded all `1,024` full-resolution WIT images into RAM at once alongside `batch_size = 32` `518x518` MPS buffers; `WIT-1024` includes a `100,512,000`-pixel Wikimedia scan (`PIL.Image.DecompressionBombWarning`). Canceled `task-188` and rewrote `scripts/extract.py` to stream `ds.select_columns(["image"])` in chunks of `8`, close each PIL image right after transform (`im.close()`), run `gc.collect()` and `torch.mps.empty_cache()`, log `ru_maxrss`, and support `--cleanup-cache` via `huggingface_hub.scan_cache_dir()`.
- **Post-freeze deviation (`qwen7b` skipped for disk safety):** `df -h /` showed only `21 GiB` free disk (`17 GiB` after caching `WIT-1024`). `Qwen/Qwen2.5-7B` requires `15.2 GB` of safetensor shards on disk plus `~15.2 GB` in `bfloat16` unified memory on a 16 GB laptop, which would exhaust disk space under swap. Kept `docs/analysis-plan.md` untouched (frozen at commit `44f154d`) and skipped `qwen7b`, testing the remaining 6 open text encoders (`bert`, `clip`, `mpnet`, `qwen0.5b`, `qwen1.5b`, `qwen3b`).
- Extracted all 7 encoders (`dinov2` + 6 text models) on `WIT-1024` (`results/raw/extract-*.json`): peak RSS stayed at `2,220.2 MB` for `dinov2` (wall clock `25,056.69 s` because the laptop slept overnight while `time.time()` was used; switched to `time.perf_counter()` afterwards) and `1,379.6 MB` to `3,790.8 MB` across all 6 text encoders, freeing `0.42 GB` to `5.90 GB` of HF cache after each run.
- Scored Phase 2 (`scripts/align.py` -> `results/phase2_wit1024.json`):
  - **6 of 6** open text encoders reproduced the paper's WIT-1024 mutual 10-NN and unbiased CKA scores within `0.0012` (mKNN) and `0.0004` (CKA), well inside the `0.010` freeze threshold (`qwen0.5b` `0.1021` vs `0.102`; `qwen1.5b` `0.1077` vs `0.108`; `bert` `0.1114` vs `0.111`; `mpnet` `0.1183` vs `0.118`; `clip` `0.1242` vs `0.124`; `qwen3b` `0.1262` vs `0.125`).
  - Removing `prh_prep`'s 0.95-quantile elementwise clip cut `qwen3b` by 35% (`0.1262 -> 0.0820`) and flattened Qwen scaling (`0.0845, 0.0873, 0.0820`), while `mpnet` barely moved (`0.1183 -> 0.1156`).
  - On all 6 text encoders, `dinov2`'s best layer was block `10` (mKNN) and block `9` (CKA) rather than final block `11`; scoring only final layers dropped `clip` to `0.0914` and `qwen3b` to `0.1062`.

### Phase 3: `SlicedEppsPulley` MPS hang repair, finite worlds and laptop LeJEPA runs

- Recorded `yerevann/coco-karpathy` licence (`CC BY 4.0`, fingerprints `8de5af95431e73bb` and `ce1603db52be1d90`) in `docs/licences.md` before downloading.
- **Apple MPS deadlock and repair (`task-277`):** First run of `scripts/phase3_views.py` completed the exact finite-world spectral checks (`prh.theory`) and then hung silently on neural step 1 for 38 minutes (`357 MB` RSS, ~13% CPU). Isolated the hang to `stable_pretraining/losses/lejepa.py:125`, where `SlicedEppsPulley.forward` calls `torch.Generator(device='mps').manual_seed(step.item())`, which deadlocks in PyTorch `2.15.0` on MPS. Canceled `task-277`, kept `BERT-T` and the projector on MPS while running only the scalar `SlicedEppsPulley` term on CPU, and vectorized token view sampling in NumPy.
- Re-ran `scripts/phase3_views.py` cleanly (`results/phase3_views.json`, `2,135.35 s` total, `2,475.5 MB` peak RSS) and cleaned the `yerevann/coco-karpathy` dataset cache:
  - **Finite worlds (`prh.theory`):** Across 20 clustered product worlds (`16` states, `6` nuisances, `k = 5`), good views hit `0.0` cross-modal error vs `10.0` (`2k`) for bad views (`20/20` satisfy Theorem 3's bound). In 10-seed sample-density sweeps (`1` to `64` samples/state), good views hit `1.000` at `1` sample/state while `r = 0.02` bad views started at `0.436` (`0.390` on mixed classes, capping at `0.885` even at `64` samples/state).
  - **Laptop neural LeJEPA (`BERT-T`, `11,297,664` params, `400` steps x `128` scenes = `51,200` COCO scenes = `204,800` views, seeds `0, 1, 2`):** WIT-1024 `mutual_knn` moved from `0.0289` at step 0 to `0.0305` (bad) vs `0.0317` (good), a null separation at `1/844`th of the paper's smallest WIT run. Global `unbiased_cka` rose from `0.0273` to `0.0439` (bad) vs `0.0636` (good), with good > bad on all 3 seeds (`+0.0164, +0.0110, +0.0319`) despite bad views reaching lower training invariance loss (`0.328` vs `0.395`). Last-layer `CLS` 10-NN dropped below init on all 3 bad-view seeds (`0.0252 -> 0.0212`) and stayed higher on all 3 good-view seeds (`0.0257`).

### Report, figures and close-out

- Wrote `scripts/report_numbers.py` (`paper/numbers.json`) and `scripts/make_figures.py` (`paper/figures/fig1-study-design.svg` through `fig4-good-vs-bad-views.svg`).
- Completed `paper/claims.md` (24 claims, `C1` to `C24`), `paper/report.md` and `README.md`. Verified unit tests pass and `docs/analysis-plan.md` remains unchanged since commit `44f154d`.
