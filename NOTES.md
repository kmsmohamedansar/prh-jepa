# Build notes

Dated log of what was done, what broke and what was measured.

## 2026-11-05

- Started as an original-paper idea ("do JEPAs converge faster?"). Found the same question already answered, under the same title, by Balestriero and LeCun (galilai-group/prh-proof, released 2026-10-30, MIT). Turned the repo into an independent check of that paper, like decision-models and rea-study.
- Their repo, read today: 345 runs, about 5,500 GPU-hours; CPU tests; per-figure reproduction guide in docs/reproduce.md. No trained checkpoints found linked in the README; a demo on Hugging Face.
- `scripts/metrics.py`: my own mutual k-NN, CKA, shuffled baseline and bootstrap, written separately from theirs. 5 tests pass on synthetic data.
- `scripts/extract.py`, `scripts/align.py`: written, not run on real models.
- Open: pin their commit, choose the language models for phase 2, record licences, freeze the plan.
- Wrote `docs/explainer.html`: the paper in plain words with drawings, for readers new to it. Its numbers are the authors' reported ones, not ours.

### Parked (2026-11-05)

Paused by the owner, who is away from the Mac. Nothing has been run against the paper yet.

### Picking this up again

1. Phase 1 needs no Mac: their CPU tests, rebuilding their headline numbers from `paper/data/*.csv`, and checking my mutual k-NN against theirs. It could run in a cloud session (4 CPUs, 15 GB, no GPU). That bends rule 1 in `CLAUDE.md` (models run on the owner's Mac), so get the owner's yes and update the rule first.
2. Phase 2 needs huggingface.co, which the cloud environment's network policy blocked on 2026-11-05 (PyPI was allowed). Either allow that host in the environment settings or run phase 2 on the Mac.
3. Phase 3 (training) needs the Mac.
4. Before any of it: pin their commit hash here and freeze `docs/analysis-plan.md`.
