"""Chisco (ds005170) stimulus-order leakage check. Metadata only, no EEG.

Claim being tested: the trial order is semantically blocked, and upstream's split
(`index % 5 == 1` -> test, see vendor/chisco/EEGclassify.py:74) puts every test trial
between two training neighbours, so a no-EEG "copy the previous label" predictor beats the
published 13.88% EEG result.

Run:  uv run --no-project --with pandas --with openpyxl python adjacent_leakage.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TEXT = Path(__file__).parent / "data" / "text"
PUBLISHED_EEG_ACC = 0.1388  # Zhang et al. 2024, Sci Data 11:1265, subject 1, 125ch
PUBLISHED_RANDOM = 0.0552


def load_labels() -> tuple[np.ndarray, np.ndarray]:
    """Concatenate split_data_1..45 in run order -> (labels, run index)."""
    labels, runs = [], []
    for n in range(1, 46):
        d = pd.read_excel(TEXT / f"split_data_{n}.xlsx", header=0)
        col = d.iloc[:, 1].dropna().astype(str).str.strip()
        labels += col.tolist()
        runs += [n] * len(col)
    cats = sorted(set(labels))
    idx = {c: i for i, c in enumerate(cats)}
    return np.array([idx[c] for c in labels]), np.array(runs)


def lag_table(y: np.ndarray, lags=(1, 2, 3, 5, 10, 50)) -> pd.DataFrame:
    prior = np.bincount(y) / len(y)
    null = float((prior**2).sum())  # P(two independent draws match)
    rows = []
    for k in lags:
        p = float((y[:-k] == y[k:]).mean())
        rows.append({"lag": k, "P(same category)": p, "x null": p / null})
    t = pd.DataFrame(rows)
    t.attrs["null"] = null
    return t


def published_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n)
    test = i % 5 == 1
    return i[~test], i[test]


def main() -> None:
    y, run = load_labels()
    print(f"trials={len(y)}  classes={len(set(y))}  files=45")

    t = lag_table(y)
    print(f"\nConsecutive-trial category agreement (null = {t.attrs['null']:.2%})")
    print(t.to_string(index=False, formatters={
        "P(same category)": "{:.2%}".format, "x null": "{:.2f}x".format}))

    train, test = published_split(len(y))
    majority = Counter(y[train]).most_common(1)[0][0]
    prev = float((y[test - 1] == y[test]).mean())
    nxt = float((y[np.minimum(test + 1, len(y) - 1)] == y[test]).mean())
    maj = float((y[test] == majority).mean())
    print(f"\nPublished split (index % 5 == 1): train={len(train)} test={len(test)}")
    print(f"  copy previous trial's label (no EEG): {prev:.2%}")
    print(f"  copy next trial's label     (no EEG): {nxt:.2%}")
    print(f"  predict train-majority class        : {maj:.2%}")
    print(f"  published EEG result (S1, 125ch)    : {PUBLISHED_EEG_ACC:.2%}")
    print(f"  published 'random' baseline         : {PUBLISHED_RANDOM:.2%}")
    print(f"  chance (1/{len(set(y))} classes)              : {1/len(set(y)):.2%}")

    # Same predictor under a run-wise holdout: a test trial's neighbours are in the same
    # held-out run, so the previous-trial label is not available from training data.
    rng = np.random.default_rng(0)
    held = rng.choice(np.unique(run), size=9, replace=False)
    te = np.isin(run, held)
    tr_maj = Counter(y[~te]).most_common(1)[0][0]
    print(f"\nRun-wise holdout (9 of 45 runs held out): majority-class = "
          f"{(y[te] == tr_maj).mean():.2%}; previous-label copy is unavailable to a "
          f"model, since test neighbours are also test.")


if __name__ == "__main__":
    main()
