#!/bin/zsh
# Reproduce every number, table and figure in the paper from public data.
#   1. download S1 pickles (imagine + read, ~26 GB)      scripts/fetch.sh
#   2. build cache, run tests, run the EEG grid (3 GPU workers, resumable)   scripts/run_grid.sh
#   3. run the released upstream script, seed 42        scripts/run_upstream.sh
#   4. no-EEG classifiers, drift probe, prediction and checkpoint analyses (CPU)
#   5. tables and figures -> paper/tables, paper/figures; compile paper/main.pdf
# Steps 2-3 take ~12 h on an Apple-silicon laptop; everything else takes minutes.
set -eu
cd "$(dirname "$0")/.."
uv sync
mkdir -p logs
for task in imagine read; do  # run_grid.sh waits on these logs
  for s in 0 1 2 3 4 5; do scripts/fetch.sh $task $s 6 > logs/fetch_${task}_$s.txt 2>&1 & done; wait
done
scripts/run_grid.sh
scripts/run_upstream.sh 42 100
uv run python scripts/label_baselines.py
uv run python scripts/drift_probe.py
uv run python scripts/prediction_analysis.py
uv run python scripts/checkpoint_analysis.py
uv run python scripts/generate_tables.py
uv run python scripts/generate_figures.py
(cd paper && tectonic -X compile main.tex)
