# PRH-JEPA study

An independent test of the Platonic Representation Hypothesis (PRH; Huh, Cheung, Wang and Isola, 2024): the claim that AI models trained on different data and tasks are converging on the same internal picture of the world. Questions:

1. Do JEPA encoders (models trained to predict hidden parts of the input in their own internal space, not in pixels) line up with a language model more closely than other image encoders of similar size do?
2. Does that hold under more than one alignment measure, once known artefacts of those measures are controlled for?
3. If yes, can a JEPA-style training signal be used to increase alignment on purpose? (Phase 2, only if phase 1 shows a signal.)

**Status (2026-11-05):** scaffold only. Nothing measured. The analysis plan is a draft and is not frozen.

## Read this first

| File | What it is |
|---|---|
| `HANDOFF.md` | The plan: phases, models, data, what would count as a result |
| `docs/analysis-plan.md` | What will be reported and claimed, to be frozen before the first scored run |
| `paper/claims.md` | Every claim, the number behind it, where it comes from, how it was checked |
| `paper/report.md` | Report skeleton |
| `NOTES.md` | Dated log of every decision, mistake and repair |
| `docs/licences.md` | Every model's and dataset's licence and where it was read |

## Layout

```
scripts/metrics.py   alignment measures: mutual k-nearest-neighbours, linear CKA, a shuffled baseline
scripts/extract.py   runs one pretrained model over the image or caption set and saves its features
scripts/align.py     scores every image encoder against the language model, writes results/
tests/               tests for the measures (run in CI)
results/             summaries are committed; raw features are not (see .gitignore)
paper/               report, claims ledger, figures
```

## Reproducing (once the plan is frozen)

```
pip install -r requirements.txt
python3 scripts/extract.py --model ijepa --split dev
python3 scripts/align.py --split dev
```
