"""Does EEG carry recording-time information that could carry the order leak?

Features are deliberately content-poor: per-channel mean and log-variance of each 3.3 s trial
(122 x 2 = 244 numbers). These track slow drift (electrode impedance, arousal, cap shifts) and
contain no temporal waveform. Multinomial logistic regression, standardised on training data.

  run probe       predict the run (45-way) of each test trial, published split
  category probe  predict the semantic category (39-way) under every split, with a
                  permuted-training-label null (5 permutations) per split

Writes results/drift_probe.json. CPU only, a few minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegwords import splits
from eegwords.data import Trials

OUT = Path(__file__).resolve().parents[1] / "results" / "drift_probe.json"


def features(t: Trials) -> np.ndarray:
    out = np.empty((len(t), 244), np.float32)
    for i in range(0, len(t), 256):
        x = np.asarray(t.X[i:i + 256], dtype=np.float64)
        out[i:i + 256, :122] = x.mean(-1)
        out[i:i + 256, 122:] = np.log(x.var(-1) + 1e-6)
    return out


def fit_score(F, ytrain, ytest, train, test) -> float:
    """Fit on ytrain[train], score against ytest[test] (differ only for the permutation null)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(F[train], ytrain[train])
    return float((clf.predict(F[test]) == ytest[test]).mean())


def main() -> None:
    t = Trials("01", "imagine")
    F = features(t)
    rng = np.random.default_rng(0)
    res: dict = {"features": "per-channel mean + log-variance (244)", "model": "logreg C=1"}

    (f,) = splits.make("interleaved", t, seed=0)
    train = np.concatenate([f.train, f.val])
    res["run_probe_interleaved"] = {"acc": fit_score(F, t.run, t.run, train, f.test),
                                    "chance": 1 / len(np.unique(t.run))}
    res["category"] = {}
    for sp in splits.SPLITS:
        real, null = [], []
        for fold in splits.make(sp, t, seed=0, max_folds=3):
            tr = np.concatenate([fold.train, fold.val])
            real.append(fit_score(F, t.label, t.label, tr, fold.test))
            for _ in range(5):
                yp = t.label.copy()
                yp[tr] = rng.permutation(t.label[tr])
                null.append(fit_score(F, yp, t.label, tr, fold.test))
        res["category"][sp] = {"real_mean": float(np.mean(real)), "real_folds": real,
                               "null_mean": float(np.mean(null)), "null_max": float(np.max(null)),
                               "null_n": len(null)}
        print(sp, res["category"][sp]["real_mean"], res["category"][sp]["null_mean"], flush=True)
    print("run probe:", res["run_probe_interleaved"])
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
