# Evidence ledger

Strength: **Strong** = directly measured, reproducible from this repo, robust to the obvious
alternative; **Moderate** = measured but with a caveat that limits generality; **Weak** = suggestive.
All paths are relative to the repo root. "S1" = sub-01.

## Source-work facts (verified from paper text and released code)

| Claim | Evidence | File | Strength | Alternative explanation | Safe wording |
|---|---|---|---|---|---|
| The published S1 result is 13.88% accuracy with a 5.52% "random" baseline | Table 3 of the paper | `sources/zhang2024_chisco_PMC11582579.xml` | Strong | — | "Zhang et al. report 13.88% for S1 and 5.52% for a label-shuffled baseline (their Table 3)." |
| The released code evaluates on trials with `index % 5 == 1` and trains on the rest | `EEGclassify.py:74` | `vendor/chisco/EEGclassify.py` | Strong (code) | The authors might have used a different local script | "In the released code, …" never "the authors did …" |
| The released code has no held-out test set; the evaluation set is the only non-training set | `EEGclassify.py:173–174` | same | Strong (code) | same | same |
| The released code reports the maximum evaluation-set accuracy over all evaluations | `EEGclassify.py:219, 244–271` | same | Strong (code) | same | "The script prints the maximum over its evaluations of accuracy on that set." |
| The paper states neither the split nor the selection rule | full text search | paper XML | Strong | — | "The paper does not describe …" |
| Released S1 imagine data: 6,243 trials, 39 classes, all mapped | cache build + test | `tests/test_splits.py`, `data/cache/*meta.json` | Strong | — | "The released preprocessed data contain 6,243 of the 6,681 presented trials for S1." |

## Trial-order structure (no EEG)

| Claim | Evidence | File | Strength | Alternative explanation | Safe wording |
|---|---|---|---|---|---|
| Consecutive trials share a category far more often than independent order would give | lag-1 same-category 19.74% vs 2.95% expected under independence (prior-matching) | `results/label_baselines.json` (`lag_agreement`) | Strong | Could be a property of the released subset rather than presentation; but the pickles are in-order subsequences of the stimulus lists | "In the released trial order, 19.7% of consecutive trials share a category; independent ordering with the same class frequencies would give 2.9%." |
| On the published split, a classifier using only trial position and training labels scores 18.82% | index 1-NN (label of nearest earlier training trial) | same (`splits.interleaved`) | Strong | It uses no test labels; every neighbour it copies is a training trial | "A nearest-neighbour classifier over trial position, which never observes EEG, reaches 18.82% (95% CI 16.49–21.57, run-clustered bootstrap) on the published evaluation set, above the published 13.88%." |
| The leak is mostly local adjacency, partly run composition | global shuffle 2.92% (max 4.24% of 1,000); within-run shuffle 5.96% (max 8.25%); run-majority classifier 14.41% | same | Strong | — | "Shuffling labels within runs lowers the position classifier from 18.8% to 6.0%; knowing only the run gives 14.4%." |
| Within-run position carries no category information | position-majority 3.68% vs majority 4.72% | same | Strong | — | "Stimulus lists do not share a layout across runs." |
| The leak disappears when whole runs are held out | index 1-NN prev 2.42–5.67% across 5 run folds; majority 2.64–4.60% | same | Strong | Adjacent-run boundary trials give small residual | "Holding out whole runs removes the position classifier's advantage." |

## EEG carries recording time (drift probe)

| Claim | Evidence | File | Strength | Alternative explanation | Safe wording |
|---|---|---|---|---|---|
| Per-trial channel means and log-variances identify the recording run almost perfectly | 98.80% 45-way run identity (chance 2.22%) on the published split | `results/drift_probe.json` | Strong | — | "A linear classifier on 244 per-trial summary statistics identifies which of 45 runs a test trial came from with 98.8% accuracy." |
| The same content-poor features predict category on the published split but not under run holdout | 9.05% (null 3.11%) vs 3.55% (null 2.76%, null max 3.77%) | same | Strong | Summary statistics might carry a little category signal; the collapse under run holdout says it is mostly time | "Features with no waveform information reach 9.05% on the published split and 3.55% when runs are held out." |

## EEG model results

**2026-10-01 00:10 — all rows below that came from `eegwords/train.py` before this time are VOID.**
A bug left the network in eval mode after each mid-epoch evaluation (no dropout, frozen batch
norm for ~2/3 of training steps). Found by comparing training loss with the released script
(epoch 99: 2.26 ours vs 3.40 upstream). Fixed in `_predict` (restores mode) with a regression test
(`tests/test_train.py`). Buggy outputs are quarantined in `results/superseded_evalmode_bug/` and
are not used. The released-script run (`results/upstream_seed42.json`) and all no-EEG / drift
results are unaffected.

| Claim | Evidence | File | Strength | Alternative explanation | Safe wording |
|---|---|---|---|---|---|
| Released script, seed 42, our hardware: prints 9.69% (best of 313 evaluations, epoch 66); last checkpoint 7.77%; post-warm-up mean 7.40% (SD 0.84) | parsed log | `results/upstream_seed42.json`, `logs/upstream_seed42_e100.txt` | Strong for one seed | Seed and hardware differ from the authors' | "Running the released script with its documented command on our hardware prints 9.69%." |
| Re-implementation, published configuration, 3 seeds: max rule 8.75 ± 0.67 (8.33/8.41/9.53); val-selected 7.13 ± 0.24; last 7.39 ± 0.73 | B3 | `results/imagine_full_500hz_interleaved_real_e100_seed*.json`, `results/summary.json` | Strong (3 seeds; agrees with released script) | hardware/seed | "Our re-implementation scores 8.75% ± 0.67 under the published rule." |
| Training loss of re-implementation vs released script: max per-epoch difference 0.043 (mean 0.016), 3 seeds | B0 vs B3 | `results/upstream_seed42.json` + B3 files | Strong | single released-script seed | "differs by at most 0.043 per epoch" |
| Permuted labels, same configuration: max rule 6.38 ± 0.30; val-selected 5.02 ± 0.76; last 4.80 ± 0.70 | B4 | `results/imagine_full_500hz_interleaved_perm_e100_seed*.json` | Strong | — | "6.38% under the published rule with permuted labels; the paper's shuffled baseline is 5.52%" |
| Real − permuted on the released split: 2.38 (max), 2.11 (val), 2.59 (last); published 8.36 | B3 − B4 | `results/summary.json` | Moderate (n=3) | leak present on this split | numbers only |
| Max rule over val-selected: +1.62 real, +1.36 permuted; max − post-warm-up mean: 1.75 / 1.35 | per-eval curves | `results/checkpoint_analysis.json` | Strong | — | numbers only |
| Real-label models copy the preceding training trial's label 2.4–3.7 pts above class-frequency expectation; permuted −0.4 to +0.2 | neighbour analysis | `results/prediction_analysis.json` | Moderate | conditional accuracies partly class-frequency driven | "at least partly mediated by recording-time structure" |
| Split grid, val-selected, real vs permuted: interleaved 7.58 vs 5.47 (+2.11); random 5.79 vs 4.78 (+1.01); run holdout 4.37 vs 3.73 (+0.64; 7/9 seed-fold pairs positive, range −2.42 to +2.71); session 2.55 vs 2.75 (−0.20) | B5 | `results/imagine_full_250hz_*_e30_seed*.json`, `results/summary.json` | Moderate (n=3 seeds × 3 folds) | low power; session confounded by label shift | "We cannot distinguish a residual signal of this size from none." |
| Max rule under session holdout: 5.66 real, 5.26 permuted, while val-selected is at chance | B5 | same | Strong | — | "The published rule reports more than double chance for models with no information." |
| Reading-task control: interleaved 6.99 ± 0.53 (below imagined 7.58); run holdout 4.85 ± 0.40 vs permuted 3.75 ± 0.54 (+1.10; 7/9 pairs positive; ~80% of the summed difference from one fold) | B6 | `results/read_full_250hz_*_e30_seed*.json`, `results/summary.json` | Moderate | reading epochs are 5 s vs 3.3 s; one fold dominates | "the pipeline has limited sensitivity once recording-time cues are removed" |
