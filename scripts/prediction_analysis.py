"""Per-trial analysis of EEG model predictions -> results/prediction_analysis.json.

For each grid cell and each reported checkpoint (val_selected, upstream_max, final):
  acc, 95% CI       cluster bootstrap over test runs (2,000 draws); a seed's folds are pooled
  copy_rate         P(prediction == label of the nearest earlier training trial)
  acc_nb_same       accuracy on test trials whose nearest earlier training trial has the SAME label
  acc_nb_diff       accuracy on test trials whose neighbour has a DIFFERENT label
  copy_rate_indep   copy_rate expected if predictions were independent of the neighbour's label
                    (sum over classes of P(pred=c) * P(neighbour=c)); a class-frequency control
  runmaj_nb_same/diff  accuracy of the no-EEG run-majority classifier on the same two subsets
If a model decodes category from the trial's own signal, acc_nb_same and acc_nb_diff should be
similar. If it decodes *when* the trial was recorded and borrows the neighbour's label, accuracy
concentrates on neighbour-same trials and acc_nb_diff falls to (or below) chance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eegwords import splits
from eegwords.data import Trials

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
B = 2000


def cluster_ci(correct: np.ndarray, run: np.ndarray, rng) -> list[float]:
    runs = np.unique(run)
    k = np.array([correct[run == r].sum() for r in runs])
    n = np.array([(run == r).sum() for r in runs])
    draws = rng.integers(0, len(runs), size=(B, len(runs)))
    stats = k[draws].sum(1) / n[draws].sum(1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main() -> None:
    rng = np.random.default_rng(0)
    trials = {"imagine": Trials("01", "imagine")}
    out = {}
    for p in sorted(RES.glob("*.json")):
        r = json.load(open(p))
        if "config" not in r or "test_pred" not in r["folds"][0]:
            continue
        c = r["config"]
        if c["task"] not in trials:
            trials[c["task"]] = Trials("01", c["task"])
        t = trials[c["task"]]
        folds = {f.name: f for f in splits.make(c["split"], t, seed=c["seed"], max_folds=c["max_folds"])}
        cell = {}
        for ck in ("val_selected", "upstream_max", "final"):
            corr, runs, nb_same, pred_is_nb, preds, nbs, rmaj = [], [], [], [], [], [], []
            for fr in r["folds"]:
                f = folds[fr["fold"]]
                test = np.array(fr["test_index"])
                assert np.array_equal(test, f.test), "fold mismatch"
                pred = np.array(fr["test_pred"][ck])
                y = t.label[test]
                train = np.sort(np.concatenate([f.train, f.val]))
                nb = t.label[train[np.clip(np.searchsorted(train, test) - 1, 0, len(train) - 1)]]
                corr.append(pred == y)
                runs.append(t.run[test])
                nb_same.append(nb == y)
                pred_is_nb.append(pred == nb)
                preds.append(pred)
                nbs.append(nb)
                rm = np.full(len(test), np.bincount(t.label[train], minlength=39).argmax())
                for rr in np.unique(t.run[test]):
                    tr_r = train[t.run[train] == rr]
                    if len(tr_r):
                        rm[t.run[test] == rr] = np.bincount(t.label[tr_r], minlength=39).argmax()
                rmaj.append(rm == y)
            corr, runs = np.concatenate(corr), np.concatenate(runs)
            nb_same, pred_is_nb = np.concatenate(nb_same), np.concatenate(pred_is_nb)
            preds, nbs, rmaj = np.concatenate(preds), np.concatenate(nbs), np.concatenate(rmaj)
            pp = np.bincount(preds, minlength=39) / len(preds)
            pn = np.bincount(nbs, minlength=39) / len(nbs)
            cell[ck] = {"acc": float(corr.mean()), "ci95": cluster_ci(corr, runs, rng),
                        "copy_rate": float(pred_is_nb.mean()),
                        "frac_nb_same": float(nb_same.mean()),
                        "acc_nb_same": float(corr[nb_same].mean()),
                        "acc_nb_diff": float(corr[~nb_same].mean()),
                        "copy_rate_indep": float((pp * pn).sum()),
                        "n_distinct_pred": int(len(np.unique(preds))),
                        "runmaj_nb_same": float(rmaj[nb_same].mean()),
                        "runmaj_nb_diff": float(rmaj[~nb_same].mean()),
                        "n": int(len(corr))}
        out[r["key"]] = {"config": c, **cell}
        v = cell["val_selected"]
        print(f"{r['key']:60s} acc={v['acc']:.4f} [{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}] "
              f"copy={v['copy_rate']:.3f} nb_same={v['acc_nb_same']:.3f} nb_diff={v['acc_nb_diff']:.3f}")
    json.dump(out, open(RES / "prediction_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
