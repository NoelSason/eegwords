"""results/*.json -> paper/figures/*.pdf. Restrained styling; every number comes from results/.

fig_order.pdf     label sequence of the first runs + lag agreement vs. prior-matching null
fig_main.pdf      published number vs. the same model under three evaluation rules, real vs permuted
                  training labels, and no-EEG classifiers (per-seed points, mean bars)
fig_curves.pdf    test accuracy at every evaluation step, real vs permuted, with the running max
fig_splits.pdf    val-selected accuracy by split, real vs permuted, with no-EEG references
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eegwords.data import Trials  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
REAL, PERM, NOEEG, REF = "#1f4e79", "#b0b0b0", "#8c510a", "#000000"

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42})


def _rows():
    rows = [json.load(open(p)) for p in sorted(RES.glob("*.json"))]
    return [r for r in rows if "config" in r]  # grid cells only


def fig_order(lb: dict) -> None:
    t = Trials("01", "imagine")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.75, 2.0), gridspec_kw={"width_ratios": [2.2, 1]})
    runs = [1, 2]
    m = np.flatnonzero(np.isin(t.run, runs) & t.keep)
    y = t.label[m]
    x = np.arange(len(y))
    same = np.flatnonzero(y[1:] == y[:-1])  # consecutive pairs sharing a category
    for i in same:
        a.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=REAL, lw=3.0, solid_capstyle="round", alpha=0.85)
    a.plot(x, y, "o", color="#808080", ms=1.2, mew=0)
    edge = int(np.sum(t.run[m] == runs[0]))
    a.axvline(edge - 0.5, color="k", lw=0.5, ls=":")
    a.text(edge / 2, 40, "run 1", ha="center", fontsize=6)
    a.text(edge + (len(y) - edge) / 2, 40, "run 2", ha="center", fontsize=6)
    a.set_ylim(-1, 42)
    a.set_xlim(-2, len(y) + 1)
    a.set_ylabel("Category (0-38)")
    a.set_xlabel("Trial, in presentation order (lines join consecutive trials of the same category)")
    lags = lb["lag_agreement"]["lags"]
    b.plot([z["lag"] for z in lags], [100 * z["p_same"] for z in lags], "o-", color=REAL, ms=3,
           lw=1, label="released order")
    b.axhline(100 * lb["lag_agreement"]["prior_matching_null"], color=PERM, lw=1, ls="--",
              label="random order")
    b.set_xscale("log")
    b.set_xlabel("Lag $k$ (trials)")
    b.set_ylabel("P(same category) (%)")
    b.set_ylim(0, None)
    b.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_order.pdf")
    plt.close(fig)


def _seed_vals(rows, metric, **kw):
    sel = [r for r in rows if all(r["config"][k] == v for k, v in kw.items())]
    return [100 * r[metric] for r in sorted(sel, key=lambda r: r["config"]["seed"])]


def fig_main(rows, lb) -> None:
    il = lb["splits"]["interleaved"][0]
    base = dict(task="imagine", split="interleaved", sfreq=500, epochs=100)
    items = [
        ("Published (Table 3, S1)", [13.88], REF),
        ("Released script, seed 42", [100 * json.load(open(RES / "upstream_seed42.json"))["printed_result"]], REF),
        ("Max over ckpts, real", _seed_vals(rows, "upstream_max", perm=False, **base), REAL),
        ("Max over ckpts, permuted", _seed_vals(rows, "upstream_max", perm=True, **base), PERM),
        ("Val-selected, real", _seed_vals(rows, "val_selected", perm=False, **base), REAL),
        ("Val-selected, permuted", _seed_vals(rows, "val_selected", perm=True, **base), PERM),
        ("No EEG: prev. training trial", [100 * il["index_1nn_prev"]], NOEEG),
        ("No EEG: majority class", [100 * il["majority"]], NOEEG),
        ("Published random baseline", [5.52], REF),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for i, (name, vals, c) in enumerate(items):
        yv = len(items) - 1 - i
        if vals:
            ax.barh(yv, np.mean(vals), color=c, alpha=0.35 if c != REF else 0.15, height=0.6)
            ax.plot(vals, [yv] * len(vals), "o", color=c, ms=3)
    ax.axvline(100 / 39, color="k", lw=0.6, ls=":")
    ax.text(100 / 39, len(items) - 0.4, " chance", fontsize=6, va="bottom")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _, _ in items][::-1])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 21)
    fig.tight_layout()
    fig.savefig(FIG / "fig_main.pdf")
    plt.close(fig)


def fig_curves(rows) -> None:
    base = dict(task="imagine", split="interleaved", sfreq=500, epochs=100)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for perm, c, lab in ((False, REAL, "real"), (True, PERM, "permuted")):
        sel = [r for r in rows if all(r["config"][k] == v for k, v in base.items())
               and r["config"]["perm"] == perm]
        for j, r in enumerate(sorted(sel, key=lambda r: r["config"]["seed"])):
            cv = r["folds"][0]["curve"]
            ep = [x["epoch"] for x in cv]
            acc = 100 * np.array([x["test_acc"] for x in cv])
            ax.plot(ep, acc, color=c, lw=0.4, alpha=0.8, label=lab if j == 0 else None)
            ax.plot(ep, np.maximum.accumulate(acc), color=c, lw=1.0, ls="--",
                    label=f"{lab}, running max" if j == 0 else None)
    ax.axhline(13.88, color=REF, lw=0.6)
    ax.text(99, 13.6, "published 13.88", fontsize=6, va="top", ha="right")
    ax.axhline(100 / 39, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 17.5)
    ax.legend(fontsize=6, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig_curves.pdf")
    plt.close(fig)


def fig_splits(rows, lb) -> None:
    order = ("interleaved", "random_trial", "run_holdout", "session_holdout")
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for i, sp in enumerate(order):
        for dx, perm, c in ((-0.15, False, REAL), (0.15, True, PERM)):
            v = _seed_vals(rows, "val_selected", task="imagine", split=sp, sfreq=250, perm=perm)
            if v:
                ax.bar(i + dx, np.mean(v), width=0.28, color=c, alpha=0.4)
                ax.plot([i + dx] * len(v), v, "o", color=c, ms=2.5)
        prev = 100 * json.load(open(RES / "summary.json"))["splits"][sp]["no_eeg_prev"]  # matched folds
        ax.plot([i - 0.35, i + 0.35], [prev, prev], color=NOEEG, lw=1.6)
    ax.axhline(100 / 39, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["interleaved\n(published)", "random\ntrial", "run\nholdout", "session\nholdout"])
    ax.set_ylabel("Accuracy, val-selected (%)")
    ax.plot([], [], "s", color=REAL, alpha=0.6, label="EEG, real labels")
    ax.plot([], [], "s", color=PERM, alpha=0.6, label="EEG, permuted")
    ax.plot([], [], color=NOEEG, label="no EEG: prev. training trial")
    ax.set_ylim(0, 24)
    ax.legend(fontsize=5.5, loc="lower right", ncol=3, bbox_to_anchor=(1.0, 1.0), handlelength=1.2,
              columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_splits.pdf")
    plt.close(fig)


def fig_upstream_check(rows) -> None:
    """Appendix: our re-implementation vs the released script, same configuration."""
    up = json.load(open(RES / "upstream_seed42.json"))
    base = dict(task="imagine", split="interleaved", sfreq=500, epochs=100, perm=False)
    sel = sorted([r for r in rows if all(r["config"][k] == v for k, v in base.items())],
                 key=lambda r: r["config"]["seed"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.75, 2.1))
    for j, r in enumerate(sel):
        cv = r["folds"][0]["curve"]
        lab = "re-implementation (seeds 0-2)" if j == 0 else None
        a.plot([x["epoch"] for x in cv], [x["train_loss"] for x in cv], color=REAL, lw=0.6, alpha=0.8, label=lab)
        b.plot([x["epoch"] for x in cv], [100 * x["test_acc"] for x in cv], color=REAL, lw=0.4, alpha=0.6, label=lab)
    ue = sorted(int(e) for e in up["train_loss_by_epoch"])
    a.plot([e + 0.5 for e in ue], [up["train_loss_by_epoch"][str(e)] for e in ue], color=NOEEG, lw=1.0,
           label="released script (seed 42)")
    b.plot([x["epoch"] for x in up["curve"]], [100 * x["test_acc"] for x in up["curve"]], color=NOEEG,
           lw=0.5, label="released script (seed 42)")
    a.set_xlabel("Epoch"); a.set_ylabel("Training loss"); a.legend(fontsize=6)
    b.axhline(13.88, color=REF, lw=0.6); b.text(1, 13.88, "published 13.88", fontsize=6, va="bottom")
    b.axhline(100 / 39, color="k", lw=0.6, ls=":")
    b.set_xlabel("Epoch"); b.set_ylabel("Evaluation-set accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_upstream_check.pdf")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    lb = json.load(open(RES / "label_baselines.json"))
    rows = _rows()
    fig_order(lb)
    fig_main(rows, lb)
    fig_curves(rows)
    fig_splits(rows, lb)
    fig_upstream_check(rows)
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
