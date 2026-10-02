"""One config -> one results/*.json holding the full evaluation curve.

The curve is the point: upstream reports max-over-evaluations on its only evaluation set. We log
val and test accuracy at every evaluation step so three protocols are read off the SAME run:

  upstream_max   max over evaluation steps of test accuracy        (upstream's reported metric)
  val_selected   test accuracy at the step with the best val accuracy (val never overlaps test)
  final          test accuracy at the last step
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from . import splits
from .data import N_CLASSES, Trials, decimate2
from .model import ChiscoNet

RESULTS = Path(__file__).resolve().parents[1] / "results"


@dataclass
class Config:
    task: str = "imagine"
    split: str = "interleaved"
    sfreq: int = 500  # 500 = native (faithful), 250 = 2x decimated
    channels: str = "full"  # "full" or a named subset (see channels.py)
    seed: int = 0
    perm: bool = False  # shuffle TRAIN labels; test labels untouched (upstream --rand_guess)
    # upstream's documented reproduction command:
    lr: float = 5e-4
    wd: float = 0.01
    epochs: int = 100
    batch: int = 16
    warmup_ratio: float = 0.1
    evals_log: int = 100
    layer: int = 1
    pooling: str = "mean"
    dropout: float = 0.5
    feel1: int = 20
    feel2: int = 10
    max_folds: int | None = None
    tag: str = ""

    def key(self) -> str:
        p = "perm" if self.perm else "real"
        return (f"{self.task}_{self.channels}_{self.sfreq}hz_{self.split}_{p}_e{self.epochs}"
                f"_seed{self.seed}{self.tag}")


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _linear_warmup(opt, warmup: int, total: int):
    def scale(step: int) -> float:  # == transformers.get_linear_schedule_with_warmup
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return torch.optim.lr_scheduler.LambdaLR(opt, scale)


@torch.no_grad()
def _predict(net, x: torch.Tensor, bs: int = 256) -> torch.Tensor:
    """Predict in eval mode, then restore the caller's mode. Evaluation happens mid-epoch; leaving
    the net in eval mode would silently train the rest of the epoch without dropout and with
    frozen batch-norm statistics (upstream calls model.train() before every step)."""
    was_training = net.training
    net.eval()
    try:
        return torch.cat([net(x[i:i + bs]) for i in range(0, len(x), bs)]).argmax(1).cpu()
    finally:
        net.train(was_training)


def _prepare(cfg: Config, t: Trials, idx: np.ndarray, chan_idx) -> torch.Tensor:
    x = np.asarray(t.X[idx])  # sorted fancy-index read from the memmap
    if chan_idx is not None:
        x = x[:, chan_idx, :]
    if cfg.sfreq == 250:
        x = decimate2(x)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _run_fold(cfg: Config, t: Trials, fold: splits.Fold, chan_idx, verbose: bool) -> dict:
    dev = device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    ytr = t.label[fold.train].copy()
    if cfg.perm:  # destroy the EEG<->label correspondence in training only
        ytr = np.random.default_rng(cfg.seed + 7).permutation(ytr)
    xtr = _prepare(cfg, t, fold.train, chan_idx).to(dev)
    xva = _prepare(cfg, t, fold.val, chan_idx).to(dev)
    xte = _prepare(cfg, t, fold.test, chan_idx).to(dev)
    ytr_t = torch.from_numpy(ytr).long().to(dev)
    yva, yte = t.label[fold.val], t.label[fold.test]

    net = ChiscoNet(chans=xtr.shape[1], n_times=xtr.shape[2], cls=N_CLASSES, dropout1=cfg.dropout,
                    dropout2=cfg.dropout, layer=cfg.layer, pooling=cfg.pooling,
                    feel1=cfg.feel1, feel2=cfg.feel2).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps_per_epoch = -(-len(xtr) // cfg.batch)
    total = steps_per_epoch * cfg.epochs
    sched = _linear_warmup(opt, int(np.ceil(total * cfg.warmup_ratio)), total)

    curve, step, run_loss, n_loss, t0 = [], 0, 0.0, 0, time.time()
    best_val, best_test = -1.0, -1.0
    preds: dict[str, list[int]] = {}  # test predictions at the three checkpoints we report
    for epoch in range(cfg.epochs):
        net.train()
        perm = torch.randperm(len(xtr))  # host RNG so shuffling is seed-stable
        for i in range(0, len(perm), cfg.batch):
            b = perm[i:i + cfg.batch].to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(net(xtr[b]), ytr_t[b])
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            n_loss += 1
            if step % cfg.evals_log == 0:
                pv, pt = _predict(net, xva).numpy(), _predict(net, xte).numpy()
                row = {"step": step, "epoch": epoch + (i // cfg.batch + 1) / steps_per_epoch,
                       "train_loss": run_loss / n_loss,
                       "val_acc": float((pv == yva).mean()), "test_acc": float((pt == yte).mean()),
                       "test_macro_f1": float(f1_score(yte, pt, average="macro"))}
                curve.append(row)
                run_loss, n_loss = 0.0, 0
                if row["val_acc"] > best_val:  # strict: first maximum wins, as in val_selected
                    best_val, preds["val_selected"] = row["val_acc"], pt.tolist()
                if row["test_acc"] > best_test:
                    best_test, preds["upstream_max"] = row["test_acc"], pt.tolist()
                preds["final"] = pt.tolist()
                if verbose:
                    print(f"  {fold.name} step {step}/{total} val={row['val_acc']:.4f} "
                          f"test={row['test_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    ta = np.array([r["test_acc"] for r in curve])
    va = np.array([r["val_acc"] for r in curve])
    sel = int(va.argmax())  # first maximum
    maj = np.bincount(t.label[fold.train], minlength=N_CLASSES).argmax()
    return {
        "fold": fold.name, "n_train": len(fold.train), "n_val": len(fold.val),
        "n_test": len(fold.test), "n_evals": len(curve),
        "upstream_max": float(ta.max()), "val_selected": float(ta[sel]),
        "val_selected_step": curve[sel]["step"], "final": float(ta[-1]),
        "mean_over_evals": float(ta.mean()),
        "majority_train_on_test": float((yte == maj).mean()),
        "seconds": time.time() - t0, "curve": curve,
        "test_index": fold.test.tolist(), "test_pred": preds,
    }


def run(cfg: Config, overwrite: bool = False, verbose: bool = False) -> dict:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{cfg.key()}.json"
    if path.exists() and not overwrite:
        return json.load(open(path))
    from .channels import indices as chan_indices
    t = Trials(task=cfg.task)
    chan_idx = None if cfg.channels == "full" else chan_indices(cfg.channels)
    folds = splits.make(cfg.split, t, seed=cfg.seed, max_folds=cfg.max_folds)
    out = [_run_fold(cfg, t, f, chan_idx, verbose) for f in folds]
    agg = {k: float(np.mean([o[k] for o in out]))
           for k in ("upstream_max", "val_selected", "final", "mean_over_evals", "majority_train_on_test")}
    res = {"config": asdict(cfg), "key": cfg.key(), **agg, "folds": out}
    json.dump(res, open(path, "w"))
    return res
