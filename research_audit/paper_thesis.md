# Paper thesis

Status: written after B1, B2, B3 completed; B4 (null), B5 (splits), B6 (reading), and the
upstream-script run are pending. Items that depend on them are marked **[pending]**. This file is
revised when they land; the final version is at the bottom.

## 1. Central finding (provisional)

On the Chisco imagined-speech benchmark (S1, 39 semantic categories), the evaluation set of the
released code can be predicted better **without EEG** than the paper reports **with EEG**: a classifier
that only looks at where a test trial sits in the recording and copies the label of the neighbouring
training trial reaches 18.8%, against the published 13.88%. The released split interleaves test
trials between training trials in a recording whose stimulus order is semantically clustered, and
EEG carries a strong recording-time signature (run identity decodable at 98.8% from per-trial
channel means and variances). The published number is also a maximum over ~300 evaluations on
that same set. Running the released script itself with its documented command (seed 42, our hardware) prints
9.69%, not 13.88%. [pending, re-running after a bug fix: our instrumented re-implementation under
three selection rules, its permuted-label null, and run-level holdout.]

## 2. Why an ML researcher should care

The failure is a combination of three ordinary choices — an index-modulo split, a stimulus list
ordered by topic, and best-checkpoint reporting on the evaluation set — none of which looks wrong
in isolation. Together they produce a number that a no-EEG baseline beats. Any benchmark built
from long, temporally structured recordings (EEG, MEG, fMRI, wearables, video) has the same
exposure. The cheap diagnostic (a position-only classifier on the exact evaluation split) costs
seconds and should be run before any decoding claim.

## 3. Interpretation it complicates

The paper concludes that the results "indicate that the Chisco EEG signals contain valid semantic
information" from 13.88% vs a 5.52% shuffled baseline. Under the released protocol, that comparison
cannot separate semantic information from recording-time information.

## 4. Supporting evidence

- Position-only classifier 18.82% (95% CI 16.49–21.57) on the exact evaluation set; 2.4–5.7% under
  run holdout. (`results/label_baselines.json`)
- Lag-1 category agreement 19.7% vs 2.95% under independent order; within-run shuffle removes most
  of it (18.8% → 6.0%). (same)
- Drift features: run identity 98.8%; category 9.05% on the published split, 3.55% under run holdout
  (nulls 3.1% / 2.8%). (`results/drift_probe.json`)
- Released script, seed 42, our hardware: printed result 9.69% (best of 313 evaluations); last
  checkpoint 7.77%. (`results/upstream_seed42.json`)
- [VOID, re-running] first re-implementation runs were trained with a mode bug; see ledger.
- [pending] Same configuration with permuted training labels under each rule.
- [pending] Real vs permuted under run and session holdout.

## 5. What would weaken it

- If the unmodified upstream script reaches ~13.88% on our data and hardware while our
  re-implementation reaches ~8%, the reproduction-gap part becomes an implementation question
  and must be reported as such.
- If the EEG model under run holdout beats its permuted null by a clear margin across seeds and
  folds, the claim becomes "inflated", not "unsupported", and the paper must say so.
- If other participants (S2, S3) have random stimulus orders, the leak is S1-specific.

## 6. Claims we cannot make

- That Chisco EEG contains no semantic information. Absence of evidence under one model, one
  participant, one preprocessing pipeline is not evidence of absence.
- That the authors chose the split or the selection rule to inflate results. We only know what the
  released code does.
- That imagined-speech EEG decoding is infeasible in general.
- That our results transfer to S2–S5 (not run).
- That the gap to 13.88% is fully explained, unless the upstream run closes it.

## 7. What is new relative to existing work

- Block-design / temporal-correlation confounds in EEG classification are known (Li et al. 2021,
  TPAMI). A concurrent code release audits Chisco's sentence-level decoding for run identity and
  stimulus order. We do not claim to have discovered the confound class.
- New here, as far as we have checked: (a) the 39-way category benchmark *as released*, audited on
  its exact evaluation split; (b) a position-only classifier that beats the published EEG number;
  (c) the published best-checkpoint rule measured directly from saved learning curves, next to a
  validation-selected checkpoint and a permuted-label null run under the same rule; (d) a two-line
  diagnostic protocol (position-only baseline + run-held-out permutation null) for any
  temporally ordered decoding benchmark.


---

## FINAL (2026-10-01, all experiments complete)

**1. Central finding.** On the released Chisco 39-way benchmark (S1), the published accuracy cannot be
attributed to imagined-speech content: a position-only classifier that never sees EEG scores 18.8% on
the released evaluation set (published: 13.88%); EEG carries a near-perfect recording-run signature
(98.8%); the released script reproduces 9.69%, not 13.88%; the published best-checkpoint rule gives
6.4% to a permuted-label model; and once whole runs are held out, real labels lead permuted labels by
0.64 points, within seed variation, in a pipeline that a reading-task control shows has limited
sensitivity (1.10 points for reading).

**2. Why an ML researcher should care.** Three routine choices (index-modulo split, topic-ordered
stimuli, best-checkpoint reporting on the evaluation set) combine into a benchmark that a no-EEG
baseline beats. The diagnostic costs seconds and applies to any temporally ordered recording.

**3. Interpretation it complicates.** "The Chisco EEG signals contain valid semantic information"
(Zhang et al., Technical Validation) — the released protocol cannot separate that from
recording-time information.

**4. Evidence.** See `final_claim_audit.md` rows 1–26.

**5. What would weaken it.** Reproducing 13.88% in the authors' environment with the released code;
a clear real-vs-null margin under run holdout with more seeds/folds or other participants; random
stimulus order in S2–S5.

**6. Claims we cannot make.** No semantic information in Chisco EEG; intent; infeasibility of
imagined-speech decoding; anything about S2–S5; why 13.88% differs from our runs.

**7. What is new.** Category benchmark audited on its exact split; position-only classifier beating
the published number; published selection rule measured from saved curves against a null run under
the same rule; reading-task sensitivity control; a five-item protocol. The confound class
(Li et al. 2021) and a Chisco run-identity audit for sentence decoding (concurrent code release) are
prior/concurrent work and are cited.

**Title.** *Trial Order, Recording Time, and Checkpoint Selection in a 39-Way Imagined-Speech EEG
Benchmark.*
