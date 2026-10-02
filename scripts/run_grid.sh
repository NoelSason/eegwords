#!/bin/zsh
# Wait for the imagine download, build the cache, gate on the tests, then run the whole grid with
# 3 GPU workers (the reading-task files download in the background meanwhile). Resumable.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
until [ "$(cat logs/fetch_imagine_*.txt 2>/dev/null | grep -c DONE)" -ge 6 ]; do sleep 20; done
pgrep -f "fetch.sh read" > /dev/null || [ "$(cat logs/fetch_read_*.txt 2>/dev/null | grep -c DONE)" -ge 6 ] || \
  for s in 0 1 2 3 4 5; do nohup scripts/fetch.sh read $s 6 > logs/fetch_read_$s.txt 2>&1 & done
uv run python -c "from eegwords import data; data.build('01','imagine')" > logs/cache_imagine.txt 2>&1
uv run --with pytest pytest tests -q > logs/tests.txt 2>&1 || { echo "TESTS FAILED"; exit 1; }
N=3  # 4 workers at 500 Hz exhausted 128 GB of unified memory and thrashed swap
for i in $(seq 0 $((N-1))); do
  uv run python -m eegwords.ablate all --queue > logs/worker_$i.txt 2>&1 &
  sleep 20  # stagger the start so claims and cache reads do not collide
done
wait
echo "GRID COMPLETE"
