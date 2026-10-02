# Reviewer attack

Written as a skeptical NeurIPS E&D reviewer after B0–B4 and the run-holdout cells of B5 were in.
Each objection: valid? → fixed experimentally / narrowed / acknowledged. Final status at bottom.

## Quality

**Q1. "Your reproduction gets 8.75%, theirs 13.88%. You just trained it worse."**
Valid concern, and it bit us once: the first re-implementation had a train/eval-mode bug.
*Fixed experimentally:* (a) the bug was found by comparing against the released script and fixed
with a regression test; (b) we ran the released `EEGclassify.py` itself with the README command
(two recorded changes: device, numerically equivalent convolution) and it printed 9.69%;
(c) training-loss curves of our code and theirs agree to ≤0.043 per epoch.
*Narrowed:* we state that 13.88% is not reproduced on our hardware and that we do not know why.
We do **not** use the reproduction gap as evidence of anything beyond that.

**Q2. "The equivalent-convolution change could alter results."**
Tested: identical parameters and init, outputs equal to float32 rounding (`tests/test_model.py`),
and the released script with this change tracks our re-implementation. Residual risk: kernel
numerics on MPS vs CUDA. Acknowledged in Limitations.

**Q3. "The position-only classifier exploits something irrelevant to the original evaluation."**
Not valid: it is scored on exactly the released evaluation trials and uses only training labels.
It shows what the released split *permits*, which is what matters for interpreting the reported
number. We do not claim the EEG model reached its accuracy this way; we show (drift probe,
copy-rate excess) that EEG carries the needed information and that the real-label model's
predictions lean on the neighbour more than its class frequencies predict.

**Q4. "Statistics: three seeds, no tests."**
Valid that power is low. We report per-seed values and SDs, run-clustered bootstrap CIs for
single classifiers, and avoid significance language. The run-holdout real-vs-permuted difference
(+0.64 points) is explicitly reported as within the seed spread. A reviewer may want more seeds;
each 30-epoch run-holdout cell costs ~45 GPU-minutes on our hardware, so 10 seeds is feasible
in principle but not done.

**Q5. "Session holdout is confounded by label shift."**
Valid. We say so and treat run holdout as the primary leak-free split.

**Q6. "Split grid uses 250 Hz/30 epochs, not the published configuration."**
Valid. Within-grid comparisons only; the interleaved cell of the grid (7.58% val-selected) bridges
to the 500 Hz reproduction (7.13%).

## Significance

**S1. "This is one failed reproduction on one participant."**
Partly valid. The contribution is not the failed reproduction; it is that the released evaluation
set admits an EEG-free classifier that beats the published number, plus a cheap diagnostic that
applies to any temporally ordered decoding benchmark. One participant is a real limitation;
the no-EEG analysis for S2–S5 is cheap and is the first extension (not done).

**S2. "Block-design confounds are known (Li et al. 2021)."**
Valid; we cite it. Our audit applies the lesson to a 2024 benchmark whose released protocol does not
guard against it, quantifies it on the exact split, and adds the selection-rule measurement.

## Originality

**O1. "A concurrent repository already audits Chisco for stimulus-order and run-identity confounds."**
Valid and acknowledged. We read its README and manifest only; it targets sentence-embedding
decoding. We claim only the specific results listed in the contributions, not the confound class.
Risk: if that work also covers the 39-way category task and the max-over-checkpoints rule, our
novelty shrinks to the measured selection-rule decomposition and the protocol checklist.

## Clarity

**C1. "Can I reproduce every number?"** `scripts/reproduce_main_results.sh`, `paper/README.md`
table → script → file map, `research_audit/final_claim_audit.md`.

## Likely objections, status

| Objection | Valid? | Action |
|---|---|---|
| single subject | yes | Limitations; first extension named |
| single dataset | yes | scope stated in title and abstract (benchmark-specific) |
| implementation mismatch | was valid | released script run; curves overlap; bug fixed |
| unfair comparison to published number | partly | we compare to the released code's own output and say the gap is unexplained |
| trial-order discrepancy across versions | checked | released trials are in-order subsequences of the stimulus lists |
| model-selection mismatch | addressed | three rules from the same runs |
| insufficient power | yes | descriptive statistics only; run-holdout margin called unresolved |
| non-EEG baseline exploits irrelevant artifact | no | scored on the exact released split |
| incompatible dataset versions | checked | one release (ds005170), counts verified |
| lack of biological interpretation | yes | out of scope; we make no neural claims |
| claim stronger than evidence | guarded | claim matrix C11–C15 rejected |

**Reading control (final).** Reading reaches only +1.10 points over its permuted null under run
holdout (7/9 pairs positive, mostly one fold), and does not beat imagined speech on the released
split. *Consequence:* the pipeline has limited sensitivity without the leak, so the imagined-speech
run-holdout margin (+0.64) is reported as weak evidence either way. This narrows our claim: we say the
benchmark *cannot separate* semantic from recording-time information, not that there is none.

**Remaining risks a reviewer will raise:** single participant (no fix this week); 13.88% unexplained
(cannot test the authors' environment); low power for the run-holdout comparison (more seeds would
help: ~45 GPU-min per cell); novelty overlap with the concurrent code release (cannot be resolved
without its paper).
