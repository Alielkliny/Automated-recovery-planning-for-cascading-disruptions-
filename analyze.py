"""Pair-level aggregation and paired tests. Unit of analysis: instance x episode pair; per pair the mean loss,
the 95th-percentile loss over replications, mean delay and mean idle days. Wilcoxon signed-rank (two-sided) of
the proposed method against each comparator, Holm correction within each metric."""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

NAMES = {"naive": "No recovery action", "practice": "Reactive expediting", "buffer": "Static safety lead time",
         "float": "Float-based rescheduling", "saaplan": "Two-stage stochastic plan",
         "detmpc": "Deterministic rolling-horizon planner", "saampc": "Sample-average rolling-horizon planner",
         "ablation": "Planner without recovery-dependent state", "carp": "Cascade-aware recovery planner"}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    return pd.DataFrame(rows)


def pair_table(df):
    g = df.groupby(["method", "instance", "episode"])
    t = g.agg(loss=("loss", "mean"), p95=("loss", lambda x: np.percentile(x, 95)), delay=("delay", "mean"),
              idle=("idle", "mean"), interv=("intervention", "mean"), hold=("holding", "mean"), stby=("standby", "mean"),
              pen=("penalty", "mean"), n=("loss", "size"), done=("completed", "mean")).reset_index()
    return t


def instance_table(df):
    """Primary inferential unit (protocol amendment 2): per method and instance, the mean over all episodes and
    replications, and the 95th percentile over all of the instance's runs."""
    g = df.groupby(["method", "instance"])
    return g.agg(loss=("loss", "mean"), p95=("loss", lambda x: np.percentile(x, 95)), delay=("delay", "mean"),
                 idle=("idle", "mean"), interv=("intervention", "mean"), hold=("holding", "mean"),
                 stby=("standby", "mean"), pen=("penalty", "mean"), n=("loss", "size")).reset_index()


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj


def compare(t, proposed="carp", metrics=("loss", "p95", "delay", "idle")):
    out = []
    idx = ["instance", "episode"] if "episode" in t.columns else ["instance"]
    piv = {m: t.pivot_table(index=idx, columns="method", values=m) for m in metrics}
    for m in metrics:
        P = piv[m].dropna()
        comps = [c for c in P.columns if c != proposed]
        ps = []
        for c in comps:
            d = P[proposed] - P[c]
            ps.append(wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0)
        adj = holm(ps)
        for c, p_raw, p_adj in zip(comps, ps, adj):
            a, b = P[proposed].mean(), P[c].mean()
            out.append(dict(metric=m, comparator=c, proposed_mean=a, comparator_mean=b,
                            rel_impr=(b - a) / b if b else np.nan, p_holm=p_adj, n_pairs=len(P),
                            wins=int((P[proposed] < P[c] - 1e-9).sum()), ties=int((abs(P[proposed] - P[c]) <= 1e-9).sum()),
                            losses=int((P[proposed] > P[c] + 1e-9).sum())))
    return pd.DataFrame(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prop = next((a[7:] for a in sys.argv[1:] if a.startswith("--prop=")), "carp")
    drop = next((a[7:].split(",") for a in sys.argv[1:] if a.startswith("--drop=")), [])
    df = load(args); df = df[~df.method.isin(drop)]
    t = pair_table(df)
    ti = instance_table(df)
    pd.set_option("display.width", 200)
    print(t.groupby("method")[["loss", "p95", "delay", "idle", "interv", "hold", "stby", "pen", "done"]].mean().round(2)
          .sort_values("loss").to_string())
    print(t.groupby(["episode", "method"]).loss.mean().unstack().round(1).to_string())
    c = compare(ti, proposed=prop)
    print("instance-level (primary)"); print(c.round(4).to_string())
