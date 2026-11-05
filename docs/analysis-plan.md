# Analysis plan (DRAFT, not frozen)

Freeze this file, with a dated commit, before the first scored run. Changes after that go in NOTES.md with the reason.

1. **Their repo:** pinned commit; tests run unchanged; failures recorded, not fixed silently.
2. **Measure:** mutual 10-NN as in the PRH paper, computed by both my code and theirs.
3. **Baselines (phase 2):** WIT-1024, DINOv2-B, each language model's features taken the way the PRH paper does. A baseline counts as reproduced if my number is within 0.01 of theirs.
4. **Good against bad views (phase 3):** same encoder, data, samples and seed list; only the view type changes. Bootstrap 95% intervals over the evaluation pairs.
5. **Claims never made:** that the paper is wrong because a small run fails; that it is right beyond what was checked.
