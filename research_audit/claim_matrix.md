# Claim matrix

Each candidate claim, the evidence for and against, and the decision. "In paper" = what the
manuscript is allowed to say. Updated 2026-10-01 05:00 (B0, B1, B2, B3, B4 complete; B5, B6 running).

| # | Candidate claim | For | Against / caveat | Decision | In paper as |
|---|---|---|---|---|---|
| C1 | The released evaluation set is predictable without EEG better than the published EEG result | position 1-NN 18.82% (CI 16.49–21.57) vs 13.88% | none found; classifier uses training labels only | **Keep (headline)** | "A classifier that never sees EEG … scores 18.8%, above the published 13.88%." |
| C2 | The released recording order is clustered by category | lag-1 19.7% vs 2.95%; within-run shuffle → 6.0% | released subset (6,243 of 6,681), but in-order subsequence | **Keep** | descriptive, with numbers |
| C3 | EEG carries a recording-time signature strong enough to exploit C2 | run identity 98.8% from 244 summary stats; category 9.05% interleaved vs 3.55% run holdout | linear probe on summary stats only; does not prove the deep model uses it | **Keep, scoped** | "carries enough information"; not "the model uses" |
| C4 | The published 13.88% is not reproduced by the released code and configuration on our hardware | released script 9.69% (1 seed); re-implementation 8.75 ± 0.67 (3 seeds) | different hardware/seed/software; one released-script seed | **Keep, scoped** | "we could not reproduce"; cause unknown, stated |
| C5 | The published number is a maximum over evaluations on the evaluation set, with no separate test set | code lines 74, 173–174, 219–271; printed result = max of logged curve | the authors might have used a different script | **Keep** | "In the released code …" |
| C6 | The max rule inflates accuracy | +1.62 (real) and +1.36 (permuted) over val-selected; max − post-warm-up mean = 1.75 / 1.35 | modest effect; does not explain the 13.88 gap | **Keep, modest** | numbers only, no "inflates substantially" |
| C7 | With the published rule, a model with no label information still reaches ~6.4% | permuted 6.38 ± 0.30, vs paper's 5.52 | — | **Keep** | as the reference the paper's "random" baseline should be read against |
| C8 | On the released split, real labels beat permuted by ~2.1–2.6 pts (paper: 8.4) | B3 vs B4 | n = 3 seeds; leak present | **Keep** | numbers; no significance language |
| C9 | The real-label model's advantage on the released split is partly mediated by neighbour/time structure | copy-rate excess +3.06 (real) vs −0.19 (perm) | conditional accuracies partly class-frequency driven (perm also shows same>diff) | **Keep, hedged** | "at least partly mediated" |
| C10 | Under run holdout the EEG model beats its permuted null | +0.64 pts mean, 7/9 seed-fold pairs positive | within seed SD (0.49/0.60); range −2.42 to +2.71; reading control pending for power | **Keep as unresolved** | "cannot distinguish a residual signal of this size from none" — neither "no signal" nor "signal" |
| C16 | The published max rule reports above-chance accuracy for models with no information | session holdout: max 5.66 / 5.26 with val-selected at chance; permuted 500 Hz: 6.38 | — | **Keep** | numbers only |
| C11 | Chisco EEG contains no semantic information | — | one participant, one model, one pipeline | **Reject** | explicitly disclaimed |
| C12 | The authors inflated results deliberately | — | no evidence of intent | **Reject** | never stated or implied |
| C13 | Imagined-speech EEG decoding is infeasible | — | out of scope | **Reject** | explicitly disclaimed |
| C14 | We are the first to identify order confounds in Chisco | — | concurrent code release audits run identity and stimulus order | **Reject** | cite concurrent work; claim only the specific contributions |
| C15 | Consumer-grade channel counts cannot decode imagined speech (legacy ladder) | legacy era A only | code lost, not re-run, main effect unresolved | **Reject for this paper** | future work |
