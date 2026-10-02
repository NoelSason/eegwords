"""logs/upstream_seed<seed>_e100.txt (released EEGclassify.py output) -> results/upstream_seed<seed>.json.

Keeps the printed `result:` line, the evaluation-set accuracy at each of the 313 evaluations, and the
training loss upstream prints every 10 steps, averaged per epoch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(seed: int = 42) -> None:
    txt = (ROOT / "logs" / f"upstream_seed{seed}_e100.txt").read_text()
    ev = re.findall(r"step:(\d+)\(epoch(\d+) [^)]*\) valid_loss:([0-9.]+) accuracy:([0-9.]+) "
                    r"max_accuracy:([0-9.]+) f1:([0-9.]+)", txt)
    tl = re.findall(r"step:(\d+)\(epoch(\d+) \d+/(\d+)\) loss:([0-9.]+)", txt)
    acc = np.array([float(r[3]) for r in ev])
    ep_eval = np.array([int(r[1]) for r in ev])
    steps_per_epoch = int(tl[0][2])
    loss_by_epoch: dict[int, list[float]] = {}
    for _, e, _, l in tl:
        loss_by_epoch.setdefault(int(e), []).append(float(l))
    post = acc[ep_eval >= 10]
    out = {
        "source": f"logs/upstream_seed{seed}_e100.txt", "seed": seed,
        "printed_result": float(re.search(r"^result: ([0-9.]+)", txt, re.M).group(1)),
        "n_evals": int(len(acc)), "steps_per_epoch": steps_per_epoch,
        "max": float(acc.max()), "argmax_epoch": int(ep_eval[acc.argmax()]),
        "final": float(acc[-1]),
        "post_warmup_mean": float(post.mean()), "post_warmup_sd": float(post.std(ddof=1)),
        "curve": [{"step": int(r[0]), "epoch": (int(r[0]) - 1) / steps_per_epoch, "test_acc": float(r[3])}
                  for r in ev],
        "train_loss_by_epoch": {str(e): float(np.mean(v)) for e, v in sorted(loss_by_epoch.items())},
    }
    assert abs(out["printed_result"] - out["max"]) < 1e-6, "printed result is not the max of the curve"
    json.dump(out, open(ROOT / "results" / f"upstream_seed{seed}.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("curve", "train_loss_by_epoch")})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
