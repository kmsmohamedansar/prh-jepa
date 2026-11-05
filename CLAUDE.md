# Rules for working on the PRH-JEPA study

Same rules as the owner's decision-models and rea-study projects. Follow them for every change.

1. **Plan limits.** No paid API or cloud GPU. Pretrained, free, openly licensed models only, run on the owner's Mac. Claude Pro is the owner's existing subscription.
2. **Data.** Only public image-caption data with a licence we have read and recorded. Nothing from the owner's employer.
3. **Fairness.** Every encoder sees the same images, the same preprocessing size, the same sample order and the same metric. Only the model changes. The analysis plan is frozen before the first scored run. The held-out image set is scored once, at the end.
4. **Honesty about the hypothesis.** We test what the models do. We never say we "proved" the Platonic Representation Hypothesis. Theory claims carry their assumptions in the same sentence.
5. **Language.** Plain words. No em dashes. Explain any jargon once.
6. **Evidence.** Keep NOTES.md dated. Run code and show real output before claiming it works. Record failures honestly.
7. **Numbers.** Only numbers from runs in this repo, with the machine named. Licence of every model and dataset checked and recorded before use.
8. **Publishing.** Don't change the portfolio site until the owner says so.
9. **Finish.** Report in the shape of decision-models/paper/report.md, claims ledger, results.
