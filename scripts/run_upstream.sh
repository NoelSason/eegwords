#!/bin/zsh
# Run upstream's EEGclassify.py with the README "reproduce the paper" command and its default seed
# (42), on a scratch copy (the vendored original is untouched). Two changes, both recorded in
# logs/upstream_patch.diff: (1) device cuda/cpu -> mps; (2) the two 1xK temporal convolutions in
# eegcnn.py use eegwords.model.SameConv1xK, which has identical parameters and init and matches
# nn.Conv2d(padding="same") to float32 rounding (tests/test_model.py) but runs ~9x faster on MPS.
# Training loop, data loading, split and reported metric are upstream's. Writes logs/upstream_seed<seed>.txt; its last line is
# "result: <max_accuracy> <max_f1>" as printed by upstream.
# usage: scripts/run_upstream.sh [seed] [epochs]
set -eu
cd "$(dirname "$0")/.."
seed=${1:-42}; epochs=${2:-100}
work=build/upstream_run
rm -rf $work && mkdir -p $work/Chisco/derivatives/preprocessed_pkl/sub-01 $work/Chisco/json
cp vendor/chisco/*.py $work/
ln -s "$PWD/data/raw/sub-01" $work/Chisco/derivatives/preprocessed_pkl/sub-01/eeg
cp data/json/textmaps.json $work/Chisco/json/
sed -i '' 's/torch.device("cuda" if torch.cuda.is_available() else "cpu")/torch.device("mps")/' $work/EEGclassify.py
python3 - "$work/eegcnn.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "nn.Conv2d(1, F1, (1, kernLength1), padding='same', bias=False)"
b = "nn.Conv2d(D*F1, D*F1, (1, kernLength2), groups=D*F1, padding='same', bias=False)"
assert s.count(a) == 1 and s.count(b) == 1, "upstream eegcnn.py changed"
s = s.replace(a, "SameConv1xK(1, F1, (1, kernLength1), padding='same', bias=False)")
s = s.replace(b, "SameConv1xK(D*F1, D*F1, (1, kernLength2), groups=D*F1, padding='same', bias=False)")
s = s.replace("import math\n", "import math\nfrom eegwords.model import SameConv1xK\n", 1)
p.write_text(s)
PY
{ diff --strip-trailing-cr vendor/chisco/EEGclassify.py $work/EEGclassify.py; diff --strip-trailing-cr vendor/chisco/eegcnn.py $work/eegcnn.py; } > logs/upstream_patch.diff || true
cd $work
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH="$OLDPWD" uv run --no-project --python 3.12 --with torch --with transformers --with scikit-learn --with scipy \
  --with mat73 --with h5py --with numpy python -u EEGclassify.py --seed $seed --rand_guess 0 \
  --lr1 5e-4 --epoch $epochs --layer 1 --pooling mean --dataset imagine_decode --sub "01" --cls 39 \
  --dropout1 0.5 --dropout2 0.5 --feel1 20 --feel2 10 --subset_ratio 1 \
  > ../../logs/upstream_seed${seed}_e${epochs}.txt 2>&1
