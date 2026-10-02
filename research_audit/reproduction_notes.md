# Reproduction notes

## What "faithful" means here

The reproduction target is the configuration the upstream README gives under "Reproducing the Paper
Results" (commit e051c5b):

```
python -u EEGclassify.py --rand_guess 0 --lr1 5e-4 --epoch 100 --layer 1 --pooling mean \
  --dataset imagine_decode --sub "01" --cls 39 --dropout1 0.5 --dropout2 0.5 --feel1 20 \
  --feel2 10 --subset_ratio 1
```

with the script's remaining defaults: batch 16, AdamW weight decay 0.01, warmup ratio 0.1 with
linear decay, `--tau 0` (no logit adjustment), evaluation every 100 steps, seed 42.

| Aspect | Upstream | Ours | Effect on comparison |
|---|---|---|---|
| Data | `derivatives/preprocessed_pkl/sub-01`, runs 1..45 in order, `[:122]` channels, ×1e6 | identical (`eegwords/data.py`) | none |
| Labels | `textmaps.json`, 39 classes | identical | none |
| Split | `index % 5 == 1` → evaluation set | identical (`splits.py`, tested) | none |
| Training set | all 4,994 non-evaluation trials | 4,495 (10% held out as validation) | ours trains on 10% less data; conservative for our model |
| Model | `EEGclassification` + `EEGcnn` | copied; temporal convs computed with unfold+einsum (identical parameters and init; equal to float32 rounding, `tests/test_model.py`) | none expected |
| Optimiser / schedule | AdamW 5e-4, wd 0.01, linear warmup 0.1 | identical (`_linear_warmup` = `get_linear_schedule_with_warmup`) | none |
| Epochs / batch | 100 / 16 | 100 / 16 | none |
| Evaluation cadence | every 100 steps on the evaluation set | every 100 steps on validation and evaluation sets | none on `upstream_max` |
| Reported number | `max_accuracy` over evaluations | `upstream_max` (same rule) plus `val_selected` and `final` | we report all three |
| Seeds | 42, one run | 0, 1, 2 | we cannot match their RNG stream; we report spread |
| Hardware | 1× A100 (CUDA) | Apple M-series (MPS), float32 | nondeterministic kernels differ; results are distributional, not bitwise |
| DataLoader shuffling | `shuffle=True` on train *and* eval loaders | train shuffled per epoch; eval order fixed | eval order does not change accuracy |

## Data facts verified during the rebuild

- The paper presents 6,681 sentences per participant. The released sub-01 imagine pickles contain
  **6,243** trials. In each of the 45 runs, the pickled sentences are an in-order subsequence of the
  run's stimulus list (`textdataset/split_data_N.xlsx`): 438 trials were removed by preprocessing,
  none reordered.
- All 6,243 sentences map to one of the 39 categories.
- No two sub-01 imagine run files are byte-identical in their first 16 MB (the concurrent audit
  reports a duplicated run file for sub-05, not sub-01).
- Sessions: the BIDS release puts sub-01's 45 imagine runs in 5 sessions of 7/8/10/10/10 runs; the
  paper describes 9 blocks per day. We use the BIDS labels.
- Reading epochs are cut from the same runs (the BIDS tree has only `task-imagine` recordings), so
  run→session mapping is shared.

## Environment

- Python 3.12, uv-managed (`pyproject.toml`, `uv.lock`); torch 2.14.1 (MPS).
- `scripts/fetch.sh imagine|read` downloads the public pickles (resumable, sharded).
- `scripts/run_grid.sh` builds the cache, gates on `pytest tests`, then runs 3 GPU workers.

## Speed note

On MPS, upstream's first temporal convolution (1→8 filters, 1×20 kernel, `padding="same"`) took
~280 ms per training step. The unfold+einsum formulation takes ~30 ms. Whole-step time fell from
337 ms to 96 ms at 500 Hz. The parameterisation is unchanged.
