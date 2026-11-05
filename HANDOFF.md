# Plan

The paper under test: galilai-group/prh-proof, pinned to the commit read on 2026-11-05 (record the hash in NOTES.md before the first run). Their own budget was about 5,500 GPU-hours over 345 runs; this study has one 16 GB MacBook Air and no paid compute, so it checks what that allows and says plainly what it cannot check.

## Phase 1: what they released (no training)

1. **Their tests.** Run `pytest -m "not slow"` on their repo (CPU, about 2 minutes by their README). Record what passes, fails or needs CUDA.
2. **Their numbers.** Rebuild each headline number in their README from `paper/data/*.csv` and `paper/runs.csv`. Check seeds and run counts match what the text says.
3. **Their measure against mine.** Run my `scripts/metrics.py` and their `prh.metrics.mutual_knn` on the same features. They must agree; if not, find out why.

## Phase 2: their baselines, measured independently (frozen models, no training)

On WIT-1024 (the PRH paper's benchmark), measure alignment between DINOv2-B and open language models that fit in 16 GB (e.g. Qwen2.5 0.5B to 7B). Compare with the numbers they report for the same models, and with the original PRH paper's numbers. This checks the comparison their main claim rests on, not their model.

## Phase 3: good views against bad views, small (training, only if phases 1 and 2 go cleanly)

Their README gives a Flickr text JEPA at 1 to 2 GPU-hours. Try one good-view and one bad-view run on the Mac, small enough to finish overnight. Prediction to test, written now: the good-view run aligns more with DINOv2 than the bad-view run at the same number of samples. Three seeds each if time allows. This is a far smaller run than theirs, so a failure here would not refute their result; it would be reported as a limit of the check.

## Out of scope, stated up front

- The 940M-image runs, the 72B language models, the image JEPAs on ImageNet, the scaling-law fit: far beyond a laptop. Reported as "not checked".
- The theorem itself is checked only as far as their executable checks go, plus reading the statement and its assumptions.

## What would count as a result

- Each claim is marked reproduced, partly reproduced, not reproduced or not checked, with the reason.
- A baseline that comes out different from theirs is reported with both numbers and the most likely cause, after asking them if it is unclear.

## Before starting

- Read their `AGENTS.md`, `docs/reproduce.md` and `docs/design.md`.
- Search for other independent replications or critiques of this paper.
