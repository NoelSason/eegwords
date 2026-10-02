"""results/*.json -> paper/tables/*.tex and results/summary.json.

Unit of replication for EEG models: a training seed. For grouped splits, a seed's value is the mean
over its folds (each fold is a different set of held-out runs/sessions), so n = 3 seeds per cell.
We report mean and sample SD across seeds and the per-seed values; no significance tests on n = 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "paper" / "tables"
PUBLISHED = {"acc": 0.1388, "random": 0.0552}  # Zhang et al. 2024, Table 3, S1
METRICS = ("upstream_max", "val_selected", "final")


def load() -> list[dict]:
    rows = [json.load(open(p)) for p in sorted(RES.glob("*.json"))]
    return [r for r in rows if "config" in r]  # grid cells only


def cell(rows: list[dict], **kw) -> list[dict]:
    return sorted([r for r in rows if all(r["config"][k] == v for k, v in kw.items())],
                  key=lambda r: r["config"]["seed"])


def stat(rows: list[dict], metric: str) -> dict:
    v = np.array([r[metric] for r in rows])
    return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "values": [float(x) for x in v]}


def pct(x: float) -> str:
    if np.isnan(x):
        return "--"
    return f"$-${-100 * x:.2f}" if x < 0 else f"{100 * x:.2f}"


def ms(s: dict) -> str:
    if s["n"] == 0:
        return "--"
    return f"{pct(s['mean'])} $\\pm$ {pct(s['sd'])}" if s["n"] > 1 else pct(s["mean"])


def table_noeeg(lb: dict) -> None:
    """No-EEG classifiers on every split. Grouped splits: mean over all 5 folds [min, max]."""
    names = [("majority", "Majority class"),
             ("index_1nn_prev", "Previous training trial"),
             ("index_1nn_next", "Next training trial"),
             ("index_knn_5", "5 nearest training trials"),
             ("run_majority", "Majority within run"),
             ("position_majority", "Majority at run position")]
    order = ("interleaved", "random_trial", "run_holdout", "session_holdout")
    t = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Classifiers that never see EEG}, scored on the evaluation trials of each split "
         r"(accuracy, \%). Each uses only training labels and the trial's position in the recording. "
         r"Interleaved is the released split. Grouped splits: mean over the five folds, "
         r"with the range in brackets. Uniform chance is 2.56\%. Source: \texttt{results/label\_baselines.json}.}",
         r"\label{tab:noeeg}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Classifier & Interleaved & Random trial & Run holdout & Session holdout \\",
         r"\midrule"]
    for key, name in names:
        cells = []
        for sp in order:
            v = [x[key] for x in lb["splits"][sp]]
            if len(v) == 1:
                cells.append(pct(v[0]))
            else:
                cells.append(f"{pct(np.mean(v))} [{pct(min(v))}, {pct(max(v))}]")
        t.append(f"{name} & " + " & ".join(cells) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "t_noeeg.tex").write_text("\n".join(t) + "\n")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    rows = load()
    lb = json.load(open(RES / "label_baselines.json"))
    table_noeeg(lb)
    if "--noeeg-only" in __import__("sys").argv:
        return
    summary: dict = {"published": PUBLISHED, "label_baselines": lb}

    # --- Table 1: faithful reproduction, three readings of the same runs -------------------
    rep = cell(rows, task="imagine", split="interleaved", sfreq=500, epochs=100, perm=False)
    nul = cell(rows, task="imagine", split="interleaved", sfreq=500, epochs=100, perm=True)
    summary["repro"] = {m: stat(rep, m) for m in METRICS}
    summary["repro_null"] = {m: stat(nul, m) for m in METRICS}
    summary["repro_n_evals"] = [r["folds"][0]["n_evals"] for r in rep]
    il = lb["splits"]["interleaved"][0]
    t1 = [r"\begin{tabular}{lccc}", r"\toprule",
          r"Selection rule & Real labels & Permuted training labels & Difference \\",
          r"\midrule"]
    names = {"upstream_max": "Max over checkpoints (published rule)",
             "val_selected": "Validation-selected checkpoint",
             "final": "Last checkpoint"}
    for m in METRICS:
        a, b = summary["repro"][m], summary["repro_null"][m]
        d = a["mean"] - b["mean"] if a["n"] and b["n"] else float("nan")
        t1.append(f"{names[m]} & {ms(a)} & {ms(b)} & {pct(d)} \\\\")
    up = json.load(open(RES / "upstream_seed42.json"))
    summary["upstream_seed42"] = {k: up[k] for k in ("printed_result", "final", "post_warmup_mean",
                                                    "post_warmup_sd", "argmax_epoch", "n_evals")}
    t1 += [r"\midrule",
           f"Released script, seed 42, max over checkpoints & {pct(up['printed_result'])} & -- & -- \\\\",
           f"Released script, seed 42, last checkpoint & {pct(up['final'])} & -- & -- \\\\",
           f"Published \\citep[Table 3, S1]{{zhang2024chisco}} & {pct(PUBLISHED['acc'])} & {pct(PUBLISHED['random'])} & "
           f"{pct(PUBLISHED['acc'] - PUBLISHED['random'])} \\\\",
           r"\midrule",
           f"No EEG: label of previous training trial & \\multicolumn{{3}}{{l}}{{{pct(il['index_1nn_prev'])} "
           f"(95\\% CI {pct(il['index_1nn_prev_ci95'][0])}--{pct(il['index_1nn_prev_ci95'][1])})}} \\\\",
           f"No EEG: majority class & \\multicolumn{{3}}{{l}}{{{pct(il['majority'])}}} \\\\",
           f"Uniform chance ($1/39$) & \\multicolumn{{3}}{{l}}{{{pct(1 / 39)}}} \\\\",
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "t1_repro.tex").write_text("\n".join(t1) + "\n")

    # --- Table 2: split comparison (250 Hz, 30 epochs) ------------------------------------
    from eegwords import baselines, splits as splits_mod
    from eegwords.data import Trials
    t_im = Trials("01", "imagine")
    label = {"interleaved": "Interleaved (released)", "random_trial": "Random trial",
             "run_holdout": "Run holdout", "session_holdout": "Session holdout"}
    order = ("interleaved", "random_trial", "run_holdout", "session_holdout")
    summary["splits"] = {}
    for sp in order:
        real = cell(rows, task="imagine", split=sp, sfreq=250, perm=False)
        perm = cell(rows, task="imagine", split=sp, sfreq=250, perm=True)
        # no-EEG classifiers on exactly the folds each EEG seed used, averaged over seeds and folds
        ne = [baselines.evaluate(t_im, f) for r in real
              for f in splits_mod.make(sp, t_im, seed=r["config"]["seed"], max_folds=r["config"]["max_folds"])]
        summary["splits"][sp] = {
            "folds": sorted({f["fold"] for r in real for f in r["folds"]}),
            "real": {m: stat(real, m) for m in METRICS}, "perm": {m: stat(perm, m) for m in METRICS},
            "no_eeg_prev": float(np.mean([x["index_1nn_prev"] for x in ne])),
            "no_eeg_majority": float(np.mean([x["majority"] for x in ne])),
            "no_eeg_run_majority": float(np.mean([x["run_majority"] for x in ne]))}
    t2 = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}", r"\toprule",
          r" & \multicolumn{3}{c}{EEG model} & \multicolumn{2}{c}{No EEG} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
          r"Split & Real & Permuted & Difference & Prev.\ trial & Majority \\", r"\midrule"]
    for m, head in (("val_selected", "Validation-selected checkpoint"),
                    ("upstream_max", "Max over checkpoints (published rule)")):
        t2.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{head}}}}} \\\\")
        for sp in order:
            v = summary["splits"][sp]
            a, b = v["real"][m], v["perm"][m]
            ne_cells = f"{pct(v['no_eeg_prev'])} & {pct(v['no_eeg_majority'])}" if m == "val_selected" else " & "
            t2.append(f"{label[sp]} & {ms(a)} & {ms(b)} & {pct(a['mean'] - b['mean'])} & {ne_cells} \\\\")
    t2 += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t2_splits.tex").write_text("\n".join(t2) + "\n")

    # --- Table 3: reading-task control -------------------------------------------------------
    summary["read"] = {}
    t3 = [r"\begin{tabular}{lcccc}", r"\toprule",
          r"Task, split & Real & Permuted & Difference & Imagined: difference \\", r"\midrule"]
    for sp, name in (("interleaved", "Reading, interleaved (released)"), ("run_holdout", "Reading, run holdout")):
        rd = cell(rows, task="read", split=sp, sfreq=250, perm=False)
        rp = cell(rows, task="read", split=sp, sfreq=250, perm=True)
        summary["read"][sp] = {"real": {m: stat(rd, m) for m in METRICS},
                               "perm": {m: stat(rp, m) for m in METRICS}}
        a_, b_ = stat(rd, "val_selected"), stat(rp, "val_selected")
        im = summary["splits"][sp]
        d_im = im["real"]["val_selected"]["mean"] - im["perm"]["val_selected"]["mean"]
        diff = pct(a_["mean"] - b_["mean"]) if b_["n"] else "--"
        t3.append(f"{name} & {ms(a_)} & {ms(b_) if b_['n'] else 'not run'} & {diff} & {pct(d_im)} \\\\")
    t3 += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t3_read.tex").write_text("\n".join(t3) + "\n")

    json.dump(summary, open(RES / "summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("repro", "repro_null")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
