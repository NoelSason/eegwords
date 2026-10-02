"""Check every number in the manuscript against values computed from results/.

For each number in paper/sections/*.tex, paper/appendix.tex and paper/tables/*.tex, look for a value
in the registry below that rounds to it at the precision written. Unmatched numbers are printed for
manual tracing. Writes results/number_audit.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"


def registry() -> dict[str, float]:
    reg: dict[str, float] = {}

    def add(name, v, pct=True):
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            reg[name] = 100 * v if pct else v

    lb = json.load(open(RES / "label_baselines.json"))
    for sp, rows in lb["splits"].items():
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float):
                    add(f"lb.{sp}.{r['fold']}.{k}", v)
            for i, v in enumerate(r["index_1nn_prev_ci95"]):
                add(f"lb.{sp}.{r['fold']}.ci{i}", v)
        for k in rows[0]:
            if isinstance(rows[0][k], float):
                vals = [x[k] for x in rows]
                add(f"lb.{sp}.mean.{k}", float(np.mean(vals)))
                add(f"lb.{sp}.min.{k}", min(vals)); add(f"lb.{sp}.max.{k}", max(vals))
                add(f"lb.{sp}.mean3.{k}", float(np.mean(vals[:3])))
    for x in lb["lag_agreement"]["lags"]:
        add(f"lag{x['lag']}", x["p_same"]); add(f"lag{x['lag']}.within", x["p_same_within_run"])
        add(f"lag{x['lag']}.within_minus", x["p_same_within_run"] - x["p_same"])
    add("prior_null", lb["lag_agreement"]["prior_matching_null"])
    for k, v in lb["order_controls_interleaved"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                add(f"order.{k}.{kk}", vv)
        else:
            add(f"order.{k}", v)
    add("chance", 1 / 39)

    dp = json.load(open(RES / "drift_probe.json"))
    add("drift.run", dp["run_probe_interleaved"]["acc"]); add("drift.run_chance", dp["run_probe_interleaved"]["chance"])
    for sp, v in dp["category"].items():
        for k in ("real_mean", "null_mean", "null_max"):
            add(f"drift.{sp}.{k}", v[k])
    add("drift.frac_of_published", dp["category"]["interleaved"]["real_mean"] / 0.1388)

    s = json.load(open(RES / "summary.json"))
    for grp in ("repro", "repro_null"):
        for m, st in s[grp].items():
            add(f"{grp}.{m}.mean", st["mean"]); add(f"{grp}.{m}.sd", st["sd"])
            for i, v in enumerate(st["values"]):
                add(f"{grp}.{m}.seed{i}", v)
    for m in ("upstream_max", "val_selected", "final"):
        add(f"repro.diff.{m}", s["repro"][m]["mean"] - s["repro_null"][m]["mean"])
    add("repro.max_minus_val.real", s["repro"]["upstream_max"]["mean"] - s["repro"]["val_selected"]["mean"])
    add("repro.max_minus_val.perm", s["repro_null"]["upstream_max"]["mean"] - s["repro_null"]["val_selected"]["mean"])
    add("perm_max_minus_published_random", s["repro_null"]["upstream_max"]["mean"] - 0.0552)
    for sp, v in s["splits"].items():
        for lab in ("real", "perm"):
            for m, st in v[lab].items():
                add(f"split.{sp}.{lab}.{m}.mean", st["mean"]); add(f"split.{sp}.{lab}.{m}.sd", st["sd"])
                for i, x in enumerate(st["values"]):
                    add(f"split.{sp}.{lab}.{m}.seed{i}", x)
        for m in ("upstream_max", "val_selected", "final"):
            add(f"split.{sp}.diff.{m}", v["real"][m]["mean"] - v["perm"][m]["mean"])
        for k in ("no_eeg_prev", "no_eeg_majority", "no_eeg_run_majority"):
            add(f"split.{sp}.{k}", v[k])
    for sp, v in s["read"].items():
        for lab in ("real", "perm"):
            for m, st in v[lab].items():
                if st["n"]:
                    add(f"read.{sp}.{lab}.{m}.mean", st["mean"]); add(f"read.{sp}.{lab}.{m}.sd", st["sd"])
        if v["perm"]["val_selected"]["n"]:
            add(f"read.{sp}.diff", v["real"]["val_selected"]["mean"] - v["perm"]["val_selected"]["mean"])

    up = json.load(open(RES / "upstream_seed42.json"))
    for k in ("printed_result", "final", "post_warmup_mean", "post_warmup_sd"):
        add(f"upstream.{k}", up[k])
    add("published_minus_upstream", 0.1388 - up["printed_result"])
    reg["upstream.argmax_epoch"] = up["argmax_epoch"]; reg["upstream.n_evals"] = up["n_evals"]

    ck = json.load(open(RES / "checkpoint_analysis.json"))["runs"]
    for perm in (False, True):
        rs = [v for v in ck.values() if v["perm"] == perm]
        pw = np.mean([v["post_warmup"]["mean"] for v in rs]); mx = np.mean([v["max_by_interval"]["1"] for v in rs])
        add(f"ckpt.{'perm' if perm else 'real'}.post_warmup_mean", pw)
        add(f"ckpt.{'perm' if perm else 'real'}.max_minus_mean", mx - pw)

    pa = json.load(open(RES / "prediction_analysis.json"))
    for perm in (False, True):
        rs = [v["val_selected"] for v in pa.values() if v["config"]["sfreq"] == 500 and v["config"]["perm"] == perm]
        tag = "perm" if perm else "real"
        for k in ("copy_rate", "copy_rate_indep", "acc_nb_same", "acc_nb_diff"):
            vals = [r[k] for r in rs]
            add(f"pa.{tag}.{k}.min", min(vals)); add(f"pa.{tag}.{k}.max", max(vals)); add(f"pa.{tag}.{k}.mean", float(np.mean(vals)))
        ex = [r["copy_rate"] - r["copy_rate_indep"] for r in rs]
        add(f"pa.{tag}.excess.min", min(ex)); add(f"pa.{tag}.excess.max", max(ex)); add(f"pa.{tag}.excess.mean", float(np.mean(ex)))
        add("pa.frac_nb_same", rs[0]["frac_nb_same"]); add("pa.runmaj_nb_diff", rs[0]["runmaj_nb_diff"])

    # paired seed-fold differences, run/session holdout (imagine and read)
    for task, sps in (("imagine", ("run_holdout", "session_holdout")), ("read", ("run_holdout",))):
        for sp in sps:
            d = []
            for sd in (0, 1, 2):
                r = json.load(open(RES / f"{task}_full_250hz_{sp}_real_e30_seed{sd}.json"))
                q = json.load(open(RES / f"{task}_full_250hz_{sp}_perm_e30_seed{sd}.json"))
                d += [fr["val_selected"] - fq["val_selected"] for fr, fq in zip(r["folds"], q["folds"])]
            add(f"paired.{task}.{sp}.min", min(d)); add(f"paired.{task}.{sp}.max", max(d))
            add(f"paired.{task}.{sp}.mean", float(np.mean(d)))
            reg[f"paired.{task}.{sp}.npos"] = int(sum(x > 0 for x in d))

    # verified constants: paper (Table 3), dataset, protocol, code
    consts = {"published.S1": 13.88, "published.S1.random": 5.52, "published.S1.ckc": 8.85,
              "published.S2": 13.59, "published.S3": 14.02, "published.S2.random": 5.55,
              "published.S3.random": 5.93, "published.margin": 8.36,
              "n_presented": 6681, "n_released": 6243, "n_removed": 438, "n_runs": 45, "n_classes": 39,
              "n_eeg_ch": 122, "n_ch": 125, "sfreq": 500, "n_samples": 1651, "read_samples": 2501,
              "n_read": 6387, "n_eval": 1249, "n_train_upstream": 4994, "n_train_ours": 4495,
              "n_val": 499, "evals_upstream": 313, "evals_ours": 281, "drift_feats": 244,
              "same_run_both": 98.24, "loss_diff_max": 0.043, "loss_diff_mean": 0.016,
              "tv_ses1": 0.39, "tv_min": 0.16, "tv_max": 0.23, "cats_per_run_median": 30,
              "top_cat_share_median": 12.9, "top_cat_share_max": 33.6, "pos_majority": 3.68,
              "loss_end_lo": 3.37, "loss_end_hi": 3.40, "fold2_share": 80}
    reg.update({k: float(v) for k, v in consts.items()})
    return reg


SKIP = re.compile(r"\\(ref|label|cite[pt]?|input|includegraphics|url|texttt|setlength|cmidrule|multicolumn|"
                  r"multirow|section|paragraph)\{[^}]*\}|\[[^\]]*(width|linewidth|Table 3, S1)[^\]]*\]|"
                  r"\\\\?[a-zA-Z]+\{?|\{[0-9.]+cm\}|fig:[a-z]+|tab:[a-z]+|app:[a-z]+|sec:[a-z]+")


def numbers(text: str):
    for line_no, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("%"):
            continue
        clean = SKIP.sub(" ", line).replace("{,}", "").replace("$-$", "-").replace("\\%", "%")
        for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?", clean):
            yield line_no, m.group(0), line.strip()[:140]


def matches(x: str, reg: dict[str, float]) -> list[str]:
    v = float(x)
    dec = len(x.split(".")[1]) if "." in x else 0
    hits = [k for k, r in reg.items() if round(abs(r), dec) == round(abs(v), dec)]
    return hits


def main() -> None:
    reg = registry()
    files = sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "appendix.tex"] + sorted((PAPER / "tables").glob("*.tex"))
    out, unmatched = [], []
    trivial = {str(i) for i in range(0, 11)} | {"100", "2024", "2026", "1000", "2000", "20", "30", "16",
               "42", "64", "8", "12", "15", "25", "50", "80", "95", "154", "294", "17"}
    for f in files:
        for ln, x, ctx in numbers(f.read_text()):
            hits = matches(x, reg)
            rec = {"file": str(f.relative_to(ROOT)), "line": ln, "number": x, "matches": hits[:4], "context": ctx}
            out.append(rec)
            if not hits and x.lstrip("-") not in trivial:
                unmatched.append(rec)
    json.dump({"n_numbers": len(out), "n_unmatched": len(unmatched), "unmatched": unmatched, "all": out},
              open(RES / "number_audit.json", "w"), indent=1)
    print(f"{len(out)} numbers checked, {len(unmatched)} not matched to a result value:")
    for u in unmatched:
        print(f"  {u['file']}:{u['line']}  {u['number']:>8}  | {u['context']}")


if __name__ == "__main__":
    main()
