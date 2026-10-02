# Original paper audit — Zhang et al. 2024 (Chisco)

**Source.** Zhang Z, Ding X, Bao Y, Zhao Y, Liang X, Qin B, Liu T. *Chisco: An EEG-based BCI
dataset for decoding of imagined speech.* Scientific Data 11, 1265 (2024).
DOI 10.1038/s41597-024-04114-1. PMC11582579.
Full text saved as `sources/zhang2024_chisco_PMC11582579.xml` (Europe PMC full-text XML).

**Code.** github.com/zhangzihan-is-good/Chisco, commit `e051c5be36ab8ff8fa375fb93f70cb2da297195d`
(`sources/upstream_commit.txt`), vendored unmodified at `vendor/chisco/`.

**Data.** OpenNeuro ds005170, `derivatives/preprocessed_pkl/sub-01/eeg/*.pkl`,
`json/textmaps.json`, `textdataset/split_data_{1..45}.xlsx`.

Legend for the last column: **code-confirmed** = the upstream code does this; **paper-only** = stated
in the paper, not checkable in code; **not stated** = neither the paper nor the code says.

| # | Item | Where (paper) | Paraphrase / exact | Implication for us | Code? |
|---|---|---|---|---|---|
| 1 | Participants | Methods, "Participants"; "Experimental setup" | 3 participants in the paper (S1–S3); repo README adds 2 more (sub-04, sub-05) | We use sub-01 = S1 only | n/a |
| 2 | Trials per participant | Methods, schedule paragraph | "The total number of sentences is 6,681, which equals 150 multiplied by 44 plus 81" | Our cache must have 6,681 trials | verified in data (see experiment_inventory) |
| 3 | Session/block structure | same paragraph | "Each participant engaged in a five-day experiment. Nine experimental blocks were conducted daily. Except for the final block, each experimental block consists of 150 trials" | 45 blocks = 45 runs; 5 days = 5 sessions. BIDS layout has 7/8/10/10/10 imagine runs in ses-01..05, not 9/day | BIDS listing (`data/json/run_session.json`) |
| 4 | Stimulus order within blocks | — | **not stated.** The paper never says whether sentences were randomised or grouped by category | Order must be measured from `split_data_N.xlsx` / trial `text` | n/a |
| 5 | Categories | Methods, "Stimuli" | 39 semantic categories | 39-way task; uniform chance 1/39 = 2.56% | `--cls 39` in README command |
| 6 | Channels | Methods, recording | 128-channel Quik-Cap; data carry 125 channels | Upstream keeps `[:122]` (drops 2 EOG + trigger) | code-confirmed (`data_imagine.py`) |
| 7 | Sampling / preprocessing | Methods, preprocessing (a)–(g) | 1 kHz recorded, downsampled to 500 Hz, PREP, 50 Hz notch + high-pass, segmentation, Autoreject, ICA | We use the released preprocessed pickles as-is, same as upstream | paper-only |
| 8 | Imagined-speech window | Methods, "Data segmentation" | "the phase from 5 to 8.3s was defined as the imagined speech segment"; reading = 0–5 s | 3.3 s × 500 Hz = 1,651 samples | pickle shape (1,125,1651) confirmed |
| 9 | Model | Technical Validation | "built on EEGNet and Transformer Encoder"; ŷ = MLP(Transformer_encoder(EEGNet(x))) | We copy upstream `EEGcnn` + `EEGclassification` | code-confirmed |
| 10 | Training hyperparameters | — (paper) / repo README | **not stated in paper.** README "Reproducing the Paper Results": `--lr1 5e-4 --epoch 100 --layer 1 --pooling mean --cls 39 --dropout1 0.5 --dropout2 0.5 --feel1 20 --feel2 10 --subset_ratio 1`; defaults batch 16, wd 0.01, warmup 0.1, evals every 100 steps | Our `repro` uses exactly these | code-confirmed |
| 11 | Train/test split | — (paper) | **not stated in paper.** Code: trial `index % 5 == 1` → evaluation set, else training (`EEGclassify.py:74`), index over all trials in run order | 1,336 eval / 5,345 train; eval trials interleaved with training trials | code-confirmed |
| 12 | Held-out test set | — | **none in code.** Only `trainset` and `validset` are built (`EEGclassify.py:173–174`) | The evaluation set is also the model-selection set | code-confirmed |
| 13 | Reported metric | — (paper) | **not stated in paper.** Code prints `max_accuracy = max(max_accuracy, valid_accu)` over every evaluation (`EEGclassify.py:244–271`), i.e. the best of ~300 evaluations on the evaluation set | The published figure is a maximum over checkpoints on the set used to choose it | code-confirmed |
| 14 | Reported result | **Table 3** | S1 Accuracy 13.88, Random Accuracy 5.52, CKC 8.85; S2 13.59/5.55/8.51; S3 14.02/5.93/8.60 | Target number 13.88% (S1) | n/a |
| 15 | "Random" baseline | Technical Validation | "we obtain random values by disrupting the correspondence between the input EEG signals x and their associated labels y … while still allowing it to learn from the label distribution" | Code: `--rand_guess 1` shuffles training inputs only; evaluation labels intact. Same `max_accuracy` metric applies | code-confirmed |
| 16 | CKC | Technical Validation | CKC = (p_o − p_e)/(1 − p_e) | (13.88 − 5.52)/(100 − 5.52) = 8.85: p_e is their random baseline, not 1/39 | arithmetic check |
| 17 | Seeds / variance | — | **not stated.** Code default `--seed 42`; no seed variance reported | Single-run numbers; we report 3 seeds | code default |
| 18 | Claim made from the result | Technical Validation | "all three subjects achieving significantly higher than random classification accuracy and F1 … The experimental results indicate that the Chisco EEG signals contain valid semantic information." | This is the interpretive claim our audit tests. No statistical test is described | paper-only |

## Discrepancies between paper and code worth stating in our paper

- The paper says results are "significantly higher than random" but describes no test, no seeds, no
  split, and no selection rule. All protocol facts we use come from the code, not the paper.
- The Table 3 header says "Accuracy", "Random Accuracy", "CKC"; the text mentions F1 but Table 3 has no F1.
- The paper describes 9 blocks/day × 5 days; the BIDS release has 7, 8, 10, 10, 10 imagine runs per
  session for sub-01. We use the BIDS session labels as released.

## What we could not verify

- Whether the exact 13.88% came from seed 42 with the README command. We cannot rerun their A100
  environment; we run their configuration on Apple MPS (numerics differ).
- Whether the paper's authors used `index % 5 == 1` for the published numbers or a different local
  script. The released code is our only evidence.
