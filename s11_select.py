"""Select each method's configuration on the tuning split by the lowest mean total loss (protocol rule)."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load, pair_table
ROOT = Path(__file__).resolve().parents[1]
from analyze import instance_table
spec = json.load(open(ROOT / "probe" / "spec_tuning.json"))
spec2 = json.load(open(ROOT / "probe" / "spec_tuning2.json"))
retuned = ["practice", "float", "detmpc", "saampc"]   # amendment 3: quote corrected, re-tuned in tune2 only
df1 = load([ROOT / "results" / "tune.jsonl"]); df2 = load([ROOT / "results" / "tune2.jsonl"])
spec3 = json.load(open(ROOT / "probe" / "spec_tuning3.json"))
if (ROOT / "results" / "tune3.jsonl").exists():
    df2 = pd.concat([df2, load([ROOT / "results" / "tune3.jsonl"])])
df1 = df1[~df1.method.str.split("@").str[0].isin(retuned)]
spec4 = json.load(open(ROOT / "probe" / "spec_tuning4.json"))
if (ROOT / "results" / "tune7.jsonl").exists():   # amendment 13: every method re-tuned on seven episode types
    g7 = json.load(open(ROOT / "probe" / "spec_tuning7.json"))["grid"]; files = [ROOT / "results" / "tune7.jsonl"]
    if (ROOT / "results" / "tune8.jsonl").exists():   # amendment 14: edge extension on the same instances, episodes and seeds
        files.append(ROOT / "results" / "tune8.jsonl")
        for k, v in json.load(open(ROOT / "probe" / "spec_tuning8.json"))["grid"].items():
            g7[k] = g7[k] + v
    if (ROOT / "results" / "tune9.jsonl").exists():   # amendment 15: second edge extension
        files.append(ROOT / "results" / "tune9.jsonl")
        for k, v in json.load(open(ROOT / "probe" / "spec_tuning9.json"))["grid"].items():
            g7[k] = g7[k] + v
    if (ROOT / "results" / "tune10.jsonl").exists():   # amendment 16 (A7): plan-time lead and constrained selection
        files.append(ROOT / "results" / "tune10.jsonl")
        for k, v in json.load(open(ROOT / "probe" / "spec_tuning10.json"))["grid"].items():
            g7[k] = g7[k] + v
    if (ROOT / "results" / "tune11.jsonl").exists():   # amendment 17 (A8): need-date safety margin
        files.append(ROOT / "results" / "tune11.jsonl")
        for k, v in json.load(open(ROOT / "probe" / "spec_tuning11.json"))["grid"].items():
            g7[k] = g7[k] + v
    RAW = load(files)
    t = instance_table(RAW)
    spec = {"grid": g7}
elif (ROOT / "results" / "tune4.jsonl").exists():   # amendment 8: every affected method re-tuned in tune4 only
    df4 = load([ROOT / "results" / "tune4.jsonl"])
    grid = dict(spec4["grid"]); grid["buffer"] = [{"buffer_days": 28}]
    if (ROOT / "results" / "tune5.jsonl").exists():   # edge extension of the advance window
        df4 = pd.concat([df4, load([ROOT / "results" / "tune5.jsonl"])])
        spec5 = json.load(open(ROOT / "probe" / "spec_tuning5.json"))
        for k, v in spec5["grid"].items():
            grid[k] = grid[k] + v
    if (ROOT / "results" / "tune6.jsonl").exists():
        df4 = pd.concat([df4, load([ROOT / "results" / "tune6.jsonl"])])
        grid["detmpc"] = grid["detmpc"] + json.load(open(ROOT / "probe" / "spec_tuning6.json"))["grid"]["detmpc"]
    t = instance_table(df4)
    spec = {"grid": grid}
else:
    t = instance_table(pd.concat([df1, df2]))
m = t.groupby("method")[["loss", "p95", "delay", "idle"]].mean()
grid = spec["grid"] if (ROOT / "results" / "tune7.jsonl").exists() else ({k: v for k, v in spec["grid"].items() if k not in retuned} if "grid" in spec and not (ROOT / "results" / "tune4.jsonl").exists() else spec["grid"])
if not (ROOT / "results" / "tune4.jsonl").exists() and not (ROOT / "results" / "tune7.jsonl").exists():
  for k, v in spec2["grid"].items():
    grid[k] = (grid.get(k, []) + v) if k == "ablation" else v
if (ROOT / "results" / "tune3.jsonl").exists() and not (ROOT / "results" / "tune4.jsonl").exists() and not (ROOT / "results" / "tune7.jsonl").exists():
    for k, v in spec3["grid"].items():
        grid[k] = grid[k] + v
spec = {"grid": grid}
chosen, table, best_lab = {}, [], {}
A7 = (ROOT / "results" / "tune10.jsonl").exists()
for base, cfgs in spec["grid"].items():
    if A7 and base == "carp":
        continue
    labs = [base + "@" + ",".join(f"{k}={v}" for k, v in c.items()) for c in cfgs]
    sub = m.loc[labs]
    best = sub.loss.idxmin(); best_lab[base] = best
    chosen[base] = cfgs[labs.index(best)]
    for lab in labs:
        table.append(dict(method=base, config=lab.split("@")[1], loss=round(sub.loc[lab, "loss"], 3),
                          p95=round(sub.loc[lab, "p95"], 3), delay=round(sub.loc[lab, "delay"], 3), idle=round(sub.loc[lab, "idle"], 3), chosen=lab == best))
if A7:   # amendment 16: the planner's setting is the lowest mean loss among settings within half the margin of the best comparator
    FAM = {"covid": "covid", "covid+evergiven": "covid", "covid+ruua": "covid", "redsea": "redsea", "redsea+production": "redsea",
           "evergiven": "single", "ruua": "single"}   # amendment 16 (c): the two single episodes form one family
    best_lab["naive"] = "naive"
    RAW = RAW.assign(fam=RAW.episode.map(FAM))
    fam_means = RAW.groupby(["method", "fam"])[["delay", "idle"]].mean()
    comp = [best_lab[b] for b in best_lab]
    ref = {("all", k): m.loc[comp, k].min() for k in ("delay", "idle")}
    for f in sorted(set(FAM.values())):
        for k in ("delay", "idle"):
            ref[(f, k)] = min(fam_means.loc[(c, f), k] for c in comp)
    cfgs = spec["grid"]["carp"]
    labs = ["carp@" + ",".join(f"{k}={v}" for k, v in c.items()) for c in cfgs]
    def excess(lab):
        e = [m.loc[lab, k] - ref[("all", k)] for k in ("delay", "idle")]
        e += [fam_means.loc[(lab, f), k] - ref[(f, k)] for f in sorted(set(FAM.values())) for k in ("delay", "idle")]
        return max(e)
    ok = [l for l in labs if excess(l) <= 0.5]
    best = min(ok, key=lambda l: m.loc[l, "loss"]) if ok else min(labs, key=excess)
    chosen["carp"] = cfgs[labs.index(best)]
    for lab in labs:
        table.append(dict(method="carp", config=lab.split("@")[1], loss=round(m.loc[lab, "loss"], 3), p95=round(m.loc[lab, "p95"], 3),
                          delay=round(m.loc[lab, "delay"], 3), idle=round(m.loc[lab, "idle"], 3),
                          max_excess=round(excess(lab), 3), chosen=lab == best))
    print("reference (best comparator per family and metric):", {f"{a}|{b}": round(v, 3) for (a, b), v in ref.items()})
chosen["naive"] = {}
(ROOT / "results" / "tuned_configs.json").write_text(json.dumps(chosen, indent=1))
pd.DataFrame(table).to_csv(ROOT / "results" / "tuning_table.csv", index=False)
print(pd.DataFrame(table).to_string()); print(chosen)
