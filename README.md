# PRH study

An independent test of [*Proving and Improving the Platonic Representation Hypothesis with JEPAs*](https://github.com/galilai-group/prh-proof) (Balestriero and LeCun, released 2026-10-30, MIT licence).

The paper claims that two encoders trained separately, one per modality, with a JEPA objective (a model trained to predict one view of something from another view, in its own internal space) learn the same embedding up to one rotation, provided their training pairs are two observations of the same thing ("good views", such as two captions of one image). It reports that a 109M-parameter text JEPA lines up with image models better than language models up to 72B do.

The only checks so far come from the authors. This study asks, on one Mac with no paid compute:

1. Do their released tests and theorem checks pass as published?
2. Do the headline numbers follow from their own released result files?
3. Do their language-model baselines come out the same when measured independently, on the PRH benchmark (WIT-1024)?
4. At a scale a laptop can train, do good views beat bad views (masked or cropped copies of one input) as the paper predicts?

**Status (2026-11-05):** scaffold only. Nothing measured. The analysis plan is a draft and is not frozen.

## Read this first

| File | What it is |
|---|---|
| `HANDOFF.md` | The plan: which claims, how each is checked, what would count as a result |
| `docs/analysis-plan.md` | What will be reported and claimed, frozen before the first scored run |
| `paper/claims.md` | Every claim, the number behind it, where it comes from, how it was checked |
| `paper/report.md` | Report skeleton |
| `NOTES.md` | Dated log of every decision, mistake and repair |
| `docs/licences.md` | Every model's and dataset's licence and where it was read |

## Layout

```
scripts/metrics.py   my own mutual k-nearest-neighbours and CKA, written separately from theirs to cross-check it
scripts/extract.py   runs one pretrained model over a benchmark and saves its features (not yet run)
scripts/align.py     scores every saved model against a reference and writes results/
tests/               tests for the measures (run in CI)
results/             summaries are committed; raw features are not
paper/               report, claims ledger, figures
```
