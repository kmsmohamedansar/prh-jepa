# Build notes

Dated log of what was done, what broke and what was measured.

## 2026-11-05

- Started as an original-paper idea ("do JEPAs converge faster?"). Found the same question already answered, under the same title, by Balestriero and LeCun (galilai-group/prh-proof, released 2026-10-30, MIT). Turned the repo into an independent check of that paper, like decision-models and rea-study.
- Their repo, read today: 345 runs, about 5,500 GPU-hours; CPU tests; per-figure reproduction guide in docs/reproduce.md. No trained checkpoints found linked in the README; a demo on Hugging Face.
- `scripts/metrics.py`: my own mutual k-NN, CKA, shuffled baseline and bootstrap, written separately from theirs. 5 tests pass on synthetic data.
- `scripts/extract.py`, `scripts/align.py`: written, not run on real models.
- Open: pin their commit, choose the language models for phase 2, record licences, freeze the plan.
