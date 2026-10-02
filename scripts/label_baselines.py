"""No-EEG classifiers on the exact test folds of the EEG grid, plus order controls.

Writes results/label_baselines.json. Runs in seconds; needs only data/cache meta (labels, runs).

Uncertainty: 95% intervals from a cluster bootstrap that resamples test *runs* with replacement
(2,000 draws), because trials within a run share stimulus-list structure and are not independent.

Order controls on the published (interleaved) split, 1,000 permutations each:
  global_shuffle      permute all labels across the 6,681 positions: destroys adjacency AND run
                      composition
  within_run_shuffle  permute labels only within each run: keeps which sentences a run contains,
                      destroys their order inside the run
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eegwords import baselines, splits
from eegwords.data import N_CLASSES, Trials

OUT = Path(__file__).resolve().parents[1] / "results" / "label_baselines.json"
B = 2000


def cluster_ci(correct: np.ndarray, run: np.ndarray, rng: np.random.Generator) -> list[float]:
    runs = np.unique(run)
    per = {r: (correct[run == r].sum(), (run == r).sum()) for r in runs}
    stats = []
    for _ in range(B):
        pick = rng.choice(runs, size=len(runs), replace=True)
        k = sum(per[r][0] for r in pick)
        n = sum(per[r][1] for r in pick)
        stats.append(k / n)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def prev_correct(y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    pos = np.clip(np.searchsorted(train, test) - 1, 0, len(train) - 1)
    return y[train[pos]] == y[test]


def main() -> None:
    t = Trials("01", "imagine")
    rng = np.random.default_rng(0)
    out: dict = {"n_trials": int(len(t)), "n_mapped": int(t.keep.sum()),
                 "n_classes": int(len(np.unique(t.label[t.keep]))),
                 "uniform_chance": 1 / N_CLASSES, "lag_agreement": baselines.lag_agreement(t),
                 "splits": {}}

    for name in splits.SPLITS:
        rows = []
        for f in splits.make(name, t, seed=0):
            r = baselines.evaluate(t, f)
            train = np.sort(np.concatenate([f.train, f.val]))
            c = prev_correct(t.label, train, f.test)
            r["index_1nn_prev_ci95"] = cluster_ci(c, t.run[f.test], rng)
            r["fold"], r["n_test"] = f.name, int(len(f.test))
            rows.append(r)
        out["splits"][name] = rows

    # order controls on the published split
    (f,) = splits.make("interleaved", t, seed=0)
    train = np.sort(np.concatenate([f.train, f.val]))
    y0 = t.label.copy()
    ctrl = {"global_shuffle": [], "within_run_shuffle": []}
    for _ in range(1000):
        y = y0.copy()
        y[t.keep] = rng.permutation(y0[t.keep])
        ctrl["global_shuffle"].append(float(prev_correct(y, train, f.test).mean()))
        y = y0.copy()
        for r in np.unique(t.run):
            m = t.run == r
            y[m] = rng.permutation(y0[m])
        ctrl["within_run_shuffle"].append(float(prev_correct(y, train, f.test).mean()))
    out["order_controls_interleaved"] = {
        k: {"mean": float(np.mean(v)), "p2.5": float(np.percentile(v, 2.5)),
            "p97.5": float(np.percentile(v, 97.5)), "max": float(np.max(v))}
        for k, v in ctrl.items()}
    out["order_controls_interleaved"]["observed"] = float(prev_correct(y0, train, f.test).mean())

    OUT.parent.mkdir(exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    s = out["splits"]
    print(f"trials={out['n_trials']} mapped={out['n_mapped']} classes={out['n_classes']}")
    for k in out["lag_agreement"]["lags"][:3]:
        print(f"  lag {k['lag']}: {k['p_same']:.4f} (within-run {k['p_same_within_run']:.4f})")
    print(f"  prior-matching null {out['lag_agreement']['prior_matching_null']:.4f}")
    for name, rows in s.items():
        m = {k: np.mean([r[k] for r in rows]) for k in ("majority", "index_1nn_prev", "index_knn_5",
                                                          "run_majority", "position_majority")}
        print(f"{name:16s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    print("interleaved prev CI95:", s["interleaved"][0]["index_1nn_prev_ci95"])
    print("order controls:", json.dumps(out["order_controls_interleaved"]))


if __name__ == "__main__":
    main()
