"""Grid driver. Each cell is one train.run() call and is skipped if its results/*.json exists.

  repro        upstream's documented command, faithfully: 122ch, 500 Hz, interleaved split,
               lr 5e-4, 100 epochs, batch 16, warmup 0.1, eval every 100 steps. 3 seeds.
  repro_null   the same with training labels permuted (upstream's --rand_guess analogue).
  splits       the same model under four splits, real and permuted labels, at 250 Hz and a
               shorter schedule (see SPLIT_EPOCHS) so the grid fits on one machine.
  read         reading-task positive control under the splits that matter.
  read_null    the reading control under run holdout with permuted training labels.

usage: python -m eegwords.ablate repro|repro_null|splits|read|all [--dry-run]
       [--queue]   (start several; each takes the next unclaimed cell. 3 fit in 128 GB at 500 Hz)
"""

from __future__ import annotations

import argparse
import os
import time

from .train import RESULTS, Config, run

CLAIMS = RESULTS / ".claims"


def _claim(cfg: Config) -> bool:
    """Atomically claim a cell for this process. A claim whose owner is dead can be retaken, so
    a killed grid resumes cleanly."""
    CLAIMS.mkdir(parents=True, exist_ok=True)
    path = CLAIMS / cfg.key()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(path.read_text() or 0), 0)  # owner alive -> not ours
            return False
        except (ProcessLookupError, ValueError):
            path.unlink(missing_ok=True)  # stale claim
            return _claim(cfg)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True

SEEDS = (0, 1, 2)
SPLIT_EPOCHS = 30


def cells(name: str) -> list[Config]:
    if name == "repro":
        return [Config(seed=s) for s in SEEDS]
    if name == "repro_null":
        return [Config(seed=s, perm=True) for s in SEEDS]
    if name == "splits":
        out = []
        for split in ("interleaved", "run_holdout", "session_holdout", "random_trial"):  # by priority
            for perm in (False, True):
                for s in SEEDS:
                    out.append(Config(split=split, sfreq=250, epochs=SPLIT_EPOCHS, seed=s,
                                      perm=perm, max_folds=3))
        return out
    if name == "read":
        return [Config(task="read", split=sp, sfreq=250, epochs=SPLIT_EPOCHS, seed=s, max_folds=3)
                for sp in ("interleaved", "run_holdout") for s in SEEDS]
    if name == "read_null":  # permuted-label reference for the reading control's leak-free split
        return [Config(task="read", split="run_holdout", sfreq=250, epochs=SPLIT_EPOCHS, seed=s,
                       perm=True, max_folds=3) for s in SEEDS]
    if name == "all":  # priority order: the faithful reproduction and its null come first
        return [c for n in ("repro", "repro_null", "splits", "read", "read_null") for c in cells(n)]
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--queue", action="store_true",
                    help="take the next unclaimed cell in priority order (run several of these)")
    a = ap.parse_args()
    cfgs = cells(a.experiment)
    print(f"{a.experiment}: {len(cfgs)} cells", flush=True)
    if a.dry_run:
        for c in cfgs:
            print(" ", c.key())
        return
    for i, cfg in enumerate(cfgs, 1):
        if (RESULTS / f"{cfg.key()}.json").exists() or (a.queue and not _claim(cfg)):
            continue
        t0 = time.time()
        try:
            r = run(cfg, verbose=True)
        except Exception as e:  # one bad cell must not kill an overnight grid
            print(f"[{i}/{len(cfgs)}] FAILED {cfg.key()} {type(e).__name__}: {e}", flush=True)
            continue
        print(f"[{i}/{len(cfgs)}] {cfg.key()} upstream_max={r['upstream_max']:.4f} "
              f"val_selected={r['val_selected']:.4f} final={r['final']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
