# Build notes

Dated log of what was done, what broke and what was measured.

## 2026-11-05

Repo created. Plan drafted in HANDOFF.md. Nothing measured yet.

- `scripts/metrics.py`: mutual k-NN alignment, linear CKA and a shuffled-pairs baseline. `tests/test_metrics.py` checks them on synthetic data (identical features score 1, a rotated copy scores 1, unrelated features score near the shuffled baseline).
- `scripts/extract.py` and `scripts/align.py` are written but have not been run against real models. They need the Mac and the model downloads.
- Open: choose the dataset and the language model, record licences, freeze `docs/analysis-plan.md`.
