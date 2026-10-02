"""How much does the published max-over-evaluations rule depend on how often you evaluate?

From each saved curve of the faithful reproduction (real and permuted labels), subsample the
evaluations to every k-th one (k*100 optimiser steps apart) and recompute the max. Also reports
the distribution of test accuracy over checkpoints after warmup. No training.

Writes results/checkpoint_analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
EVERY = (1, 2, 5, 10, 28, 100)  # x100 steps; 28 ~ once per epoch at 4,495 training trials


def main() -> None:
    out: dict = {"every_x100_steps": list(EVERY), "runs": {}}
    for p in sorted(RES.glob("imagine_full_500hz_interleaved_*_e100_seed*.json")):
        r = json.load(open(p))
        cv = r["folds"][0]["curve"]
        acc = np.array([x["test_acc"] for x in cv])
        ep = np.array([x["epoch"] for x in cv])
        post = acc[ep >= 10]  # after the 10-epoch warmup
        out["runs"][r["key"]] = {
            "perm": r["config"]["perm"], "seed": r["config"]["seed"], "n_evals": int(len(acc)),
            "max_by_interval": {str(k): float(acc[k - 1::k].max()) for k in EVERY},
            "post_warmup": {"mean": float(post.mean()), "sd": float(post.std(ddof=1)),
                            "p05": float(np.percentile(post, 5)), "p50": float(np.median(post)),
                            "p95": float(np.percentile(post, 95)), "max": float(post.max())},
            "argmax_epoch": float(ep[acc.argmax()]),
            "val_selected_epoch": float(ep[int(np.argmax([x["val_acc"] for x in cv]))]),
        }
    json.dump(out, open(RES / "checkpoint_analysis.json", "w"), indent=1)
    for k, v in out["runs"].items():
        print(k, {i: round(m, 4) for i, m in v["max_by_interval"].items()}, v["post_warmup"])


if __name__ == "__main__":
    main()
