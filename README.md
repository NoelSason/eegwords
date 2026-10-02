# Trial order, recording time, and checkpoint selection in a 39-way imagined-speech EEG benchmark

**Paper:** [Sason_2026_imagined_speech_eeg_audit.pdf](paper/Sason_2026_imagined_speech_eeg_audit.pdf) ·
also at [noelsason.com/work](https://noelsason.com/work#imagined-speech-eeg-audit)

An audit of the semantic-category benchmark released with **Chisco** ([Zhang et al., *Scientific
Data* 2024](https://doi.org/10.1038/s41597-024-04114-1)), a public imagined-speech EEG corpus. This
repository contains the code, every result file, and the evidence trail behind the paper.

## Findings (participant S1, 39 categories, chance 2.56%)

| | Accuracy |
|---|---|
| Published result (Chisco Table 3) | 13.88% |
| **No EEG:** copy the label of the preceding training trial | **18.82%** (95% CI 16.49–21.57) |
| Released script, documented configuration, our hardware | 9.69% |
| Re-implementation, published reporting rule (3 seeds) | 8.75% ± 0.67 |
| Same, trained on permuted labels | 6.38% ± 0.30 |
| Real − permuted, released split / **whole runs held out** | 2.11 / **0.64** points |

- The released stimulus order is clustered by category (19.7% of consecutive trials share one; 2.95% if shuffled), and the released split puts every evaluation trial between training trials from the same run.
- Per-trial EEG summary statistics identify a held-out trial's recording run with 98.8% accuracy, which gives a model the information it needs to exploit that order.
- The released code reports the best of 313 evaluations on its only evaluation set.
- When whole runs are held out, the real-vs-permuted margin falls within seed variation. A reading-task control shows the pipeline has limited sensitivity once those cues are gone.

These results show what this benchmark, as released, cannot establish. They do not show that the
EEG has no semantic content. All results are for one participant (see the paper's Limitations).

## Quick check (no EEG needed)

The headline number takes three lines, using only the released trial order and labels
(`data/cache/sub-01_imagine_meta.json` ships with the repo; it holds no EEG):

```python
import json
y = json.load(open("data/cache/sub-01_imagine_meta.json"))["label"]
test = [i for i in range(len(y)) if i % 5 == 1]          # the released split
print(sum(y[i] == y[i - 1] for i in test) / len(test))   # 0.188
```

## What's here

| Path | Contents |
|---|---|
| `paper/` | LaTeX source, figures, generated tables, and the [PDF](paper/Sason_2026_imagined_speech_eeg_audit.pdf). `paper/README.md` maps every table and figure to the script and file that produce it. |
| `results/` | One JSON per EEG configuration and seed (see below), plus the no-EEG baselines, drift probe, per-trial prediction analysis, checkpoint analysis, `summary.json`, and `number_audit.json`. |
| `results/superseded_evalmode_bug/` | First-pass runs voided by a training bug (found and fixed; see `research_audit/`). Kept for transparency, not used. |
| `eegwords/` | Data loader (follows the released code's trial order and label mapping), splits, model, trainer, no-EEG baselines. |
| `scripts/` | Download, grid runner, the released-script runner, analyses, table and figure generation, number audit. |
| `tests/` | Split integrity, model equivalence, trainer regression tests. |
| `vendor/chisco/` | The authors' code, unmodified (MIT), commit `e051c5b`. |
| `logs/` | Output of the released script (`upstream_seed42_e100.txt`) and the two changes made to run it (`upstream_patch.diff`). |
| `sources/` | Full text of the Chisco paper (Europe PMC XML) and the upstream commit hash. |
| `research_audit/` | Evidence ledger, experiment inventory, claim matrix, original-paper audit, reviewer objections, and a claim-by-claim trace of every number in the paper. |

### Result files

`results/{task}_full_{rate}hz_{split}_{labels}_e{epochs}_seed{n}.json`

- `task`: `imagine` or `read` (reading-epoch control)
- `split`: `interleaved` (released rule `index % 5 == 1`), `random_trial`, `run_holdout`, or `session_holdout`
- `labels`: `real` or `perm` (training labels permuted; evaluation labels intact)

Each file stores the config, and for every fold the validation and test accuracy at each of the
~280 evaluations, the test predictions at three checkpoints, and accuracy under three selection
rules: `upstream_max` (best evaluation, the published rule), `val_selected` (checkpoint chosen on
held-out validation data), and `final`.

## Reproduce

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and about 26 GB of disk for the
public data. Developed on an Apple-silicon laptop (PyTorch MPS); CUDA and CPU also work.

```bash
uv sync
scripts/fetch.sh imagine && scripts/fetch.sh read    # public data from OpenNeuro ds005170
uv run --with pytest pytest tests -q                 # 13 tests

# minutes, from existing results/ (no training)
uv run python scripts/label_baselines.py
uv run python scripts/generate_tables.py
uv run python scripts/generate_figures.py
uv run python scripts/audit_numbers.py

# everything from scratch (~12 h of GPU time)
scripts/reproduce_main_results.sh
```

## Data

The data are Chisco, [OpenNeuro ds005170](https://openneuro.org/datasets/ds005170)
(doi:10.18112/openneuro.ds005170.v1.1.2), released under CC0 by Zhang, Zhao, Bao and Ding. This
repository redistributes no EEG; `scripts/fetch.sh` downloads it from the public bucket.

## Citation

```bibtex
@misc{sason2026chiscoaudit,
  title  = {Trial Order, Recording Time, and Checkpoint Selection in a 39-Way
            Imagined-Speech {EEG} Benchmark},
  author = {Sason, Noel},
  year   = {2026},
  note   = {\url{https://noelsason.com/papers/sason-2026-imagined-speech-eeg-audit.pdf}}
}
```

Please also cite the dataset: Zhang et al., *Chisco: An EEG-based BCI dataset for decoding of
imagined speech*, Scientific Data 11, 1265 (2024).

## License

Code: MIT (see `LICENSE`). `vendor/chisco/` keeps its original MIT license. The paper is © 2026
Noel Sason.

## AI assistance

Most of the analysis code and the paper's first draft were written with Claude (Anthropic) under
the author's direction; the paper's "Use of AI assistance" section gives the details.
