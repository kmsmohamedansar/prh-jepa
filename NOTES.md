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
