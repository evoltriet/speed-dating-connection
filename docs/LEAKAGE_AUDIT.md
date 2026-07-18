# Leakage audit

1. `match` is the only outcome.
2. `dec` and `dec_o` are banned because `match = dec × dec_o`.
3. Post-event follow-up fields are banned.
4. Each directed record is collapsed into one dyad before splitting.
5. Entire waves are held out together.
6. Feature engineering never reads the outcome except to attach the dyad label.
7. Missing-value handling occurs inside each training fold.
8. Every feature-set ablation uses identical held-out folds and model settings.
9. Shared-interest ratings remain a separate feature set from other perceived traits.
10. The result is association in a constrained first meeting, not causal evidence of connection.
