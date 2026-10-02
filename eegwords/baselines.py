"""Classifiers that never see EEG. Each one is fit on training labels only and is evaluated on
exactly the test trials of the EEG experiments (same `splits.make`), so rows are comparable.

  majority        most frequent training label
  prior_sample    draws from the training label distribution (expected accuracy, analytic)
  index_1nn_prev  label of the nearest *earlier* training trial in presentation order
  index_1nn_next  label of the nearest *later* training trial
  index_knn_k     majority label of the k nearest training trials by index (ties -> nearest)
  run_majority    most frequent training label within the test trial's run (falls back to majority
                  when the run has no training trials, as under run/session holdout)
  position_majority  most frequent training label at the same within-run position (trial number
                  inside its run) across all runs; tests whether stimulus lists share a layout

Index-based classifiers use only the test trial's position in the recording, never its label.
"""

from __future__ import annotations

import numpy as np

from . import splits
from .data import N_CLASSES, Trials


def _nearest_train(train: np.ndarray, test: np.ndarray, side: str) -> np.ndarray:
    pos = np.searchsorted(train, test)
    if side == "prev":
        j = np.clip(pos - 1, 0, len(train) - 1)
    else:
        j = np.clip(pos, 0, len(train) - 1)
    return train[j]


def _knn(train: np.ndarray, ytrain_full: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(test), int)
    for n, i in enumerate(test):
        d = np.abs(train - i)
        order = np.lexsort((train, d))[:k]  # nearest first, earlier index breaks ties
        labs = ytrain_full[train[order]]
        counts = np.bincount(labs, minlength=N_CLASSES)
        best = np.flatnonzero(counts == counts.max())
        out[n] = next(lab for lab in labs if lab in best)  # tie -> the nearest one's label
    return out


def evaluate(t: Trials, fold: splits.Fold) -> dict[str, float]:
    y = t.label
    train = np.sort(np.concatenate([fold.train, fold.val]))  # everything the model may see
    test = fold.test
    yt = y[test]
    prior = np.bincount(y[train], minlength=N_CLASSES) / len(train)
    maj = int(prior.argmax())
    res = {
        "majority": float((yt == maj).mean()),
        "prior_sample": float(prior[yt].mean()),
        "index_1nn_prev": float((y[_nearest_train(train, test, "prev")] == yt).mean()),
        "index_1nn_next": float((y[_nearest_train(train, test, "next")] == yt).mean()),
    }
    for k in (3, 5, 11):
        res[f"index_knn_{k}"] = float((_knn(train, y, test, k) == yt).mean())
    run_pred = np.full(len(test), maj)
    for r in np.unique(t.run[test]):
        tr_r = train[t.run[train] == r]
        if len(tr_r):
            run_pred[t.run[test] == r] = np.bincount(y[tr_r], minlength=N_CLASSES).argmax()
    res["run_majority"] = float((run_pred == yt).mean())
    pos = np.empty(len(t), int)
    for r in np.unique(t.run):
        m = np.flatnonzero(t.run == r)
        pos[m] = np.arange(len(m))
    pos_pred = np.full(len(test), maj)
    for q in np.unique(pos[test]):
        tr_q = train[pos[train] == q]
        if len(tr_q):
            pos_pred[pos[test] == q] = np.bincount(y[tr_q], minlength=N_CLASSES).argmax()
    res["position_majority"] = float((pos_pred == yt).mean())
    return res


def lag_agreement(t: Trials, lags=(1, 2, 3, 5, 10, 20, 50)) -> dict:
    """P(label_i == label_{i+k}) over consecutive mapped trials, and within-run only."""
    idx = t.index[t.keep]
    y, run = t.label[idx], t.run[idx]
    prior = np.bincount(y, minlength=N_CLASSES) / len(y)
    out = {"prior_matching_null": float((prior ** 2).sum()), "lags": []}
    for k in lags:
        same = y[:-k] == y[k:]
        within = run[:-k] == run[k:]
        out["lags"].append({"lag": k, "p_same": float(same.mean()),
                            "p_same_within_run": float(same[within].mean())})
    return out
