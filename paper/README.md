# Paper: build and provenance

`main.tex` is a standard 11pt article (1-inch margins, natbib, numeric citations).

Build: `tectonic -X compile main.tex` (or `latexmk -pdf main.tex`).

Every number in the paper is read from a file under `../results/` by a script in `../scripts/`.
Nothing is typed in by hand except the published values from Zhang et al. (2024), Table 3
(13.88% accuracy, 5.52% random baseline for S1).

| Item | Produced by | Reads |
|---|---|---|
| Figure 1 `figures/fig_order.pdf` | `scripts/generate_figures.py` (`fig_order`) | `results/label_baselines.json`, `data/cache/sub-01_imagine_meta.json` |
| Figure 2 `figures/fig_main.pdf` | `generate_figures.py` (`fig_main`) | `results/imagine_full_500hz_interleaved_*_e100_seed*.json`, `results/label_baselines.json` |
| Figure 3 `figures/fig_curves.pdf` | `generate_figures.py` (`fig_curves`) | same EEG result files (saved learning curves) |
| Figure 4 `figures/fig_splits.pdf` | `generate_figures.py` (`fig_splits`) | `results/imagine_full_250hz_*_e30_seed*.json`, `results/label_baselines.json` |
| Table `tables/t_noeeg.tex` | `scripts/generate_tables.py` (`table_noeeg`) | `results/label_baselines.json` |
| Table `tables/t1_repro.tex` | `generate_tables.py` | 500 Hz result files, `results/label_baselines.json` |
| Table `tables/t2_splits.tex` | `generate_tables.py` | 250 Hz result files, `results/label_baselines.json` |
| Drift-probe numbers (Sec. 4) | `scripts/drift_probe.py` | `data/cache/sub-01_imagine_X.npy` → `results/drift_probe.json` |
| Neighbour analysis (Sec. 5) | `scripts/prediction_analysis.py` | EEG result files → `results/prediction_analysis.json` |
| Checkpoint-interval analysis | `scripts/checkpoint_analysis.py` | 500 Hz result files → `results/checkpoint_analysis.json` |
| Released-script result | `scripts/run_upstream.sh 42 100` | `logs/upstream_seed42_e100.txt` (last line `result:`) |

Regenerate all tables and figures from existing results (minutes, no training):

```
uv run python scripts/label_baselines.py
uv run python scripts/drift_probe.py
uv run python scripts/prediction_analysis.py
uv run python scripts/checkpoint_analysis.py
uv run python scripts/generate_tables.py
uv run python scripts/generate_figures.py
```

Re-run everything from public data: `scripts/reproduce_main_results.sh` (~12 h on an
Apple-silicon laptop, ~26 GB download).

The claim-by-claim audit trail is in `../research_audit/final_claim_audit.md`.
