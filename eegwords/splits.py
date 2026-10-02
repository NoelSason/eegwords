"""Train/test splits. All return folds over *original* trial indices, restricted to mapped trials.

interleaved     upstream: test = original index % 5 == 1 (vendor/chisco/EEGclassify.py:74)
random_trial    random 20% of trials as test (seeded); same test size as `interleaved`
run_holdout     5 folds of 9 runs each (seeded permutation of the 45 runs); test = whole runs
session_holdout 5 folds, test = one whole recording session

Every grouped split leaves ~80% of trials for training, matching `interleaved`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Trials

SPLITS = ("interleaved", "random_trial", "run_holdout", "session_holdout")


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray  # held out of train, used ONLY for checkpoint selection


def _finish(name: str, t: Trials, test_mask: np.ndarray, seed: int, group: np.ndarray | None) -> Fold:
    keep = t.keep
    train = t.index[keep & ~test_mask]
    test = t.index[keep & test_mask]
    rng = np.random.default_rng(seed + 1000)
    if group is None:  # trial-level splits: random 10% of train trials
        val = rng.choice(train, size=len(train) // 10, replace=False)
    else:  # grouped splits: 10% of the training groups (whole runs), so selection is leak-free too
        groups = np.unique(group[train])
        gval = rng.choice(groups, size=max(1, len(groups) // 10), replace=False)
        val = train[np.isin(group[train], gval)]
    train = np.setdiff1d(train, val)
    return Fold(name, np.sort(train), np.sort(test), np.sort(val))


def make(split: str, t: Trials, seed: int = 0, max_folds: int | None = None) -> list[Fold]:
    if split == "interleaved":
        return [_finish("interleaved", t, t.index % 5 == 1, seed, None)]
    if split == "random_trial":
        rng = np.random.default_rng(seed)
        test = np.zeros(len(t), bool)
        test[rng.choice(len(t), size=int(round(len(t) / 5)), replace=False)] = True
        return [_finish("random_trial", t, test, seed, None)]
    if split == "run_holdout":
        runs = np.unique(t.run)
        perm = np.random.default_rng(0).permutation(runs)  # fixed fold layout; seed varies model only
        folds = [perm[i::5] for i in range(5)]
        out = [_finish(f"run_fold{i}", t, np.isin(t.run, f), seed, t.run) for i, f in enumerate(folds)]
    elif split == "session_holdout":
        out = [_finish(f"ses{s}", t, t.session == s, seed, t.run) for s in np.unique(t.session)]
    else:
        raise ValueError(split)
    return out[:max_folds] if max_folds else out
