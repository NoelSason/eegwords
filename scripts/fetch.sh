#!/bin/zsh
# Resumable, sharded download of Chisco sub-01 preprocessed pickles (public OpenNeuro bucket).
# usage: scripts/fetch.sh imagine|read [shard n_shards]
set -u
cd "$(dirname "$0")/.."
task=${1:-imagine}; shard=${2:-0}; nsh=${3:-1}
base=https://s3.amazonaws.com/openneuro.org
prefix=ds005170/derivatives/preprocessed_pkl/sub-01/eeg/
keys=(${(f)"$(curl -s "$base?prefix=$prefix&max-keys=200" | grep -oE "<Key>[^<]+" | sed 's/<Key>//' | grep "task-$task" | sort)"})
i=0
for k in $keys; do
  i=$((i+1)); [ $(( (i-1) % nsh )) -eq $shard ] || continue
  f=data/raw/sub-01/${k:t}
  sz=$(curl -sI "$base/$k" | awk 'tolower($1)=="content-length:"{print $2+0}')
  if [ -f "$f" ] && [ "$(stat -f%z "$f")" = "$sz" ]; then continue; fi
  curl -sS -C - --retry 5 -o "$f" "$base/$k" && echo "ok $f"
done
echo "FETCH $task shard $shard DONE"
