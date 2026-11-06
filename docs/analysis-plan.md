# Analysis plan (FROZEN 2026-11-05)

Frozen with a dated commit before the first scored run. Any changes after this freeze are recorded in `NOTES.md` with the reason.

## 1. Repository under test (Phase 1)
- **Repository:** `galilai-group/prh-proof`, pinned at commit `a2b9949fa907d4fbe562742d4b1ffa7205ab7427` (with `galilai-group/stable-pretraining` at `ab836bf699a2be712dfcf980a5eb70d36391e876`).
- **Tests:** Run `pytest -m "not slow"` on `vendor/prh-proof` unchanged. Record what passes, what fails, and what requires optional extras or CUDA. Do not silently edit upstream tests to make them pass.
- **Released tables:** Rebuild every headline number in their `README.md` from `paper/data/*.csv` and `paper/runs.csv` via `scripts/phase1_audit.py`. Record any headline number that is absent from `paper/data/*.csv` or any difference between sample ratios and token ratios.
- **Metric cross-check:** Compare `scripts/metrics.py` (`mutual_knn`, `unbiased_cka`, `prh_prep`, `prh_score`) against `prh.metrics` on shared synthetic and real feature arrays. They must agree within floating-point precision (`1e-6`).

## 2. Independent baseline measurements on WIT-1024 (Phase 2)
- **Dataset:** `minhuh/prh` (`revision="wit_1024"`, `split="train"`), 1,024 image-caption pairs, in dataset order. Scored once per frozen model.
- **Reference image encoder:** `dinov2` (`timm` `vit_base_patch14_dinov2.lvd142m`, float32, native resolution transform `518x518`, CLS token hooked after each of the 12 transformer blocks).
- **Text encoders tested (7 open models that fit in 16 GB RAM):**
  1. `qwen0.5b` (`Qwen/Qwen2.5-0.5B`, float32)
  2. `qwen1.5b` (`Qwen/Qwen2.5-1.5B`, float32)
  3. `qwen3b` (`Qwen/Qwen2.5-3B`, bfloat16, matching `prh.features`)
  4. `qwen7b` (`Qwen/Qwen2.5-7B`, bfloat16, matching `prh.features`)
  5. `bert` (`google-bert/bert-base-uncased`, float32)
  6. `mpnet` (`sentence-transformers/all-mpnet-base-v2`, float32)
  7. `clip` (`openai/clip-vit-base-patch32`, float32, text tower hidden states, `max_length=77`)
- **Feature extraction rule:** First caption of each WIT-1024 row (`ds["text"][i][0]`), tokenized with `padding=True, truncation=True, max_length=512` (`77` for `clip`), attention-masked mean of every hidden state (`output_hidden_states=True`).
- **Scoring rule:** Following `prh.metrics.prh_score` (and Huh et al. 2024), each layer's `[1024, d]` features are clipped elementwise at the 0.95 quantile of absolute values and L2-normalised (`prh_prep`), and the reported score is the maximum over all `(layer_a, layer_b)` pairs of mutual 10-NN (`k=10`) and of unbiased linear CKA (`unbiased_cka`).
- **Sensitivity checks reported alongside the main score:**
  1. Last-layer mutual 10-NN (scoring only the final layer of each encoder, without taking the maximum over layer pairs).
  2. Unclipped mutual 10-NN (maximum over layer pairs without the 0.95-quantile elementwise clip).
- **Uncertainty intervals:** Report both the 1,000-draw bootstrap 95% interval over the 1,024 per-item overlap values (`bootstrap`, which keeps the 1,024-item neighbourhood pool fixed) and the 200-draw 800-of-1,024 subsampling interval without replacement (`subsample_ci`, matching `prh.metrics.subsample_interval`).
- **Reproduction criterion:** A baseline counts as **reproduced** if our mutual 10-NN score on WIT-1024 is within `0.01` of the number reported in `paper/data/llm_scale.csv` or `paper/data/wit_cka.csv`.

## 3. Small good-views against bad-views check (Phase 3, if run)
- Same encoder architecture, corpus, sample count and seeds; only the view constructor changes (`good` views drawing two captions of the same scene against `bad` views drawing masked spans of one caption).
- Because a laptop run is orders of magnitude smaller than the paper's 5,500 GPU-hour runs, a null result at laptop scale does not refute the paper and is reported only as a scale limit.

## 4. Claims never made
- That the paper's 940M-image or 72B-parameter results are wrong or right beyond what was checked here.
- That a small run failing on a laptop refutes a large-scale training result.
- That the theorem holds outside its stated assumptions (exact global minimisers, matched world-state marginals, and conditional independence of good views given the world state).
