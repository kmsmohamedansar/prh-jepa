# Plan

The paper under test: `galilai-group/prh-proof`, pinned to commit `a2b9949fa907d4fbe562742d4b1ffa7205ab7427` read on 2026-11-05 (with `galilai-group/stable-pretraining` at `ab836bf699a2be712dfcf980a5eb70d36391e876`). Their own budget was about 5,500 GPU-hours over 345 runs; this study has one 16 GB Apple M4 MacBook Air and no paid compute, so it checks what that allows and states plainly what it cannot check.

## Phase 1: what they released (no training)

1. **Their tests.** Run `pytest -m "not slow"` on their repo (CPU, about 2 minutes by their README). Record what passes, fails or needs optional packages or CUDA.
2. **Their numbers.** Rebuild each headline number in their README from `paper/data/*.csv` and `paper/runs.csv` (`scripts/phase1_audit.py`). Check seeds and run counts match what the text says.
3. **Their measure against mine.** Run `scripts/metrics.py` and their `prh.metrics` on the same features. They must agree; if not, find out why.

## Phase 2: their baselines, measured independently (frozen models, no training)

On WIT-1024 (the PRH paper's benchmark), measure alignment between DINOv2-B (`vit_base_patch14_dinov2.lvd142m`) and seven open text encoders that fit in 16 GB (`Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`, `Qwen2.5-7B`, `bert-base-uncased`, `all-mpnet-base-v2`, `clip-vit-base-patch32`). Compare with the numbers they report for the same models in `paper/data/llm_scale.csv` and `paper/data/wit_cka.csv`. Also measure how much the score changes without layer-pair maximisation or without 0.95-quantile elementwise clipping.

## Phase 3: good views against bad views, small (training, only if phases 1 and 2 go cleanly)

Their README gives a Flickr text JEPA at 1 to 2 GPU-hours. If a small text corpus is available locally, try a small good-view and bad-view run on the Mac. Prediction to test: the good-view run aligns more with DINOv2 than the bad-view run at the same number of samples. Because a laptop run is far smaller than theirs, a failure here would not refute their result; it would be reported as a limit of the check.

## Out of scope, stated up front

- The 940M-image runs, the 72B language models, the image JEPAs on ImageNet, the full 145-run scaling-law training grid: far beyond a laptop. Reported as "not checked".
- The theorem itself is checked only as far as their executable checks go, plus reading the statement and its assumptions.

## What would count as a result

- Each claim is marked reproduced, partly reproduced, not reproduced or not checked, with the reason.
- A baseline that comes out different from theirs is reported with both numbers and the most likely cause, after asking them if it is unclear.
