# Missing evidence and open items

Ordered by how much each gap limits the paper's claims.

## Limits the scope of claims (cannot be fixed on this machine this week)

1. **One participant.** Everything is sub-01 (= S1). The paper reports S2 and S3 at 13.59% and
   14.02%, and the dataset has sub-02..05. Each extra participant is ~10 GB of imagine pickles and
   ~10 GPU-hours for the faithful reproduction + null on MPS. The no-EEG analysis for other
   participants costs seconds once their stimulus order is known. *Cheapest high-value extension:*
   label-order analysis on sub-02..05 (needs only the pickles' `text` fields, but those are inside
   the ~10 GB files).
2. **Upstream environment not replicated.** We run their configuration on Apple MPS with our seeds,
   not seed 42 on an A100. We compare distributions over 3 seeds to one published number.
3. **Grouped-split grid uses a shorter schedule** (30 epochs at 250 Hz) than the faithful
   reproduction (100 epochs at 500 Hz). The comparisons *within* the split grid are like for like;
   comparisons *between* it and B3 are not.
4. **Session holdout mixes two effects.** Sessions differ in category mix (session 1 has total-variation
   distance 0.39 from the rest), so a session-holdout accuracy near chance cannot separate "no
   signal" from "label shift". Run holdout is the cleaner leak-free split.

## Legacy claims we are NOT carrying forward (not re-run)

5. **Channel-count ladder.** Era A ran 8 channel layouts (122 → 4 channels) × 2 models. Its code and
   channel-name mapping are lost, the released pickles carry no channel names, and the result
   (all within the null band) cannot be interpreted until the main effect is settled. Dropped from
   the paper; noted as future work.
6. **EOG-only / frontal / posterior artifact controls** (era A). Same reasons.

## Would strengthen the paper; not yet done

7. **Upstream code run unmodified.** Running `vendor/chisco/EEGclassify.py` itself (CUDA or CPU) with
   seed 42 would show whether its printed `max_accuracy` lands near 13.88%. On MPS it needs a
   device patch; on CPU a 100-epoch run is days.
8. **Checkpoint-count sensitivity.** How `upstream_max` grows with the number of evaluations (e.g.
   every 25 vs 100 vs 500 steps) can be computed from our saved curves by subsampling, no training.
9. **Extreme-value null for the max rule.** The permuted-label runs give the empirical null of the
   max rule; an analytic check (max of ~281 correlated binomials at p≈chance) is optional.
10. **Other participants' published numbers** (13.59, 14.02) are not audited.

## Could not verify

- Whether the authors produced 13.88% with the released script and README command, or with a
  local variant. The paper does not describe the split or the selection rule.
- The relationship between the paper's "9 blocks per day" and the BIDS session layout (7/8/10/10/10).
- Whether the concurrent audit (`mortadajafar/imagined-speech-eeg-audit`) reports the 39-way
  category task or the max-over-checkpoints rule. We read its README and manifest only; we did not
  find a paper.
