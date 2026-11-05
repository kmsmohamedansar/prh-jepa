# Analysis plan (DRAFT, not frozen)

Freeze this file, with a dated commit, before the first scored run. Changes after that go in NOTES.md with the reason.

1. **Primary measure:** mutual k-NN alignment, k = 10, 1,000 held-out pairs.
2. **Secondary:** linear CKA; mutual k-NN at k = 5, 20, 50.
3. **Uncertainty:** 1,000 bootstrap resamples of the pairs; 95% intervals.
4. **Baseline:** the same measure with captions shuffled against images, reported for every encoder.
5. **Claim allowed if:** I-JEPA's interval lies above both MAE's and DINOv2's on the primary measure, held-out.
6. **Claims never made:** that the PRH is proved; that the objective is the cause, as opposed to data or compute differences.
