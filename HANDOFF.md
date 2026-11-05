# Plan

## Phase 1: pilot (a few GPU-hours on the Mac)

Measure how closely frozen, pretrained image encoders line up with one fixed language model, on the same image-caption pairs, using the PRH paper's own measure (mutual k-nearest-neighbours) plus linear CKA.

**Image encoders, by training objective** (exact checkpoints and licences to be confirmed in `docs/licences.md` before use):

| Objective | Candidate |
|---|---|
| JEPA (latent prediction) | I-JEPA ViT-H/14 (`facebook/ijepa_vith14_1k`) |
| Masked pixel prediction | MAE ViT-H/14 |
| Self-distillation | DINOv2 ViT-g/14 or ViT-L/14 |
| Contrastive with text | CLIP ViT-L/14 (expected to align most; it saw captions) |
| Supervised classification | ViT-H/14 trained on ImageNet labels |

Matching model size matters: compare within a size class, and report size alongside every number.

**Language model:** one open model, small enough for 16 GB, fixed for every comparison. Caption features averaged over tokens, as in the PRH paper.

**Data:** a public image-caption set (candidate: the Wikipedia image-text set the PRH paper used, or COCO captions). 1,000 pairs for development, 1,000 different pairs held out and scored once.

## What would count as a result

Written before any measurement. Final wording goes in `docs/analysis-plan.md`.

- **Signal:** I-JEPA aligns more than MAE and DINOv2 of similar size, on both measures, with bootstrap intervals that do not overlap, on held-out pairs.
- **No signal:** I-JEPA sits inside the range of the other non-text encoders. This is still reported. It would mean the training objective alone does not drive convergence at this scale.
- CLIP is a reference point, not a competitor: it was trained on captions.

## Phase 2 (only if phase 1 shows a signal)

- A short theory section: under stated assumptions, latent-prediction objectives recover the shared latent variables up to a simple transformation. Assumptions stated in the same sentence as the result.
- A small training experiment: add a JEPA-style loss to a small encoder and measure whether alignment rises. Must fit on the Mac.
- A use for it: zero-shot image-to-caption retrieval through a single linear map.

## Before starting

- Search arXiv and Google Scholar for "JEPA" with "Platonic", "representation alignment" and "convergence". If someone has already done phase 1, change the question.

## Known weaknesses (recorded before any run)

- Pretrained models differ in data and compute, not only in objective. We cannot match those; we can only report them.
- Mutual k-NN alignment depends on k, the sample size and the feature dimension. The plan fixes k in advance and reports a sweep.
- One language model. A result could be specific to it.
