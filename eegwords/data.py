"""Chisco sub-XX loader that reproduces upstream's trial order and label mapping.

Upstream (vendor/chisco/data_imagine.py) concatenates runs 1..45 in numeric order, keeps every
trial, scales volts -> microvolts, keeps the first 122 of 125 channels, and maps the sentence to
one of 39 categories through textmaps.json (-1 = unmapped, dropped by the dataset class *after*
the `index % 5` split test is evaluated on the unfiltered index).
"""

from __future__ import annotations

import fcntl
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
N_CLASSES = 39
N_TIMES = 1651


def _path(sub: str, task: str, run: int) -> Path:
    return RAW / f"sub-{sub}" / f"sub-{sub}_task-{task}_run-0{run}_eeg.pkl"


def build(sub: str = "01", task: str = "imagine") -> Path:
    """Write float32 (N,122,T) memmap + meta.json. Idempotent and safe across processes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    xp, mp = CACHE / f"sub-{sub}_{task}_X.npy", CACHE / f"sub-{sub}_{task}_meta.json"
    with open(CACHE / f".{sub}_{task}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # a second worker waits here, then sees the files
        if not (xp.exists() and mp.exists()):
            _build(sub, task, xp, mp)
    return xp


def _build(sub: str, task: str, xp: Path, mp: Path) -> None:
    textmaps = json.load(open(ROOT / "data" / "json" / "textmaps.json"))
    missing = [r for r in range(1, 46) if not _path(sub, task, r).exists()]
    assert not missing, f"missing {task} runs {missing}; run scripts/fetch.sh {task} first"
    chunks, text, run = [], [], []
    for r in range(1, 46):
        trials = pickle.load(open(_path(sub, task, r), "rb"))
        for t in trials:
            f = t["input_features"]
            assert f.shape[:2] == (1, 125), f.shape  # imagine: 1651 samples; read: longer
            chunks.append(np.float32(f[0, :122, :] * 1e6))
            text.append(t["text"].strip())
            run.append(r)
        print(f"run {r}: {len(trials)} trials", flush=True)
    n_times = chunks[0].shape[-1]
    assert all(c.shape[-1] == n_times for c in chunks)
    X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float32, shape=(len(chunks), 122, n_times))
    for i, c in enumerate(chunks):
        X[i] = c
    X.flush()
    label = [int(textmaps.get(s, -1)) for s in text]
    # meta is written last: its presence marks a complete cache
    json.dump({"text": text, "run": run, "label": label}, open(mp, "w"), ensure_ascii=False)


class Trials:
    """All trials in upstream order. `index` is the unfiltered position used by the split."""

    def __init__(self, sub: str = "01", task: str = "imagine"):
        self.X = np.load(build(sub, task), mmap_mode="r")
        m = json.load(open(CACHE / f"sub-{sub}_{task}_meta.json"))
        self.text = m["text"]
        self.run = np.array(m["run"])
        self.label = np.array(m["label"])
        self.index = np.arange(len(self.label))
        # reading and imagining epochs are cut from the same recorded runs (BIDS has task-imagine only)
        ses = json.load(open(ROOT / "data" / "json" / "run_session.json"))["imagine"]
        self.session = np.array([int(ses[str(r)].split("-")[1]) for r in self.run])
        self.keep = self.label >= 0

    def __len__(self):
        return len(self.label)


def decimate2(x: np.ndarray) -> np.ndarray:
    """500 Hz -> 250 Hz by averaging sample pairs (1651 -> 825 samples)."""
    n = (x.shape[-1] // 2) * 2
    return 0.5 * (x[..., 0:n:2] + x[..., 1:n:2])
