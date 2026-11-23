"""Misspecification layer (protocol amendment 5): the proposed planner re-scored with perturbed internal
constants on a development subset and, after the test opens, on a J60 test subset, against the strongest comparators
(static safety lead time, deterministic and sample-average MPC) on the same worlds. Usage: python s22_misspec.py dev|test3"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import run
from model.network import EPISODES_DEV, EPISODES_TEST
ROOT = Path(__file__).resolve().parents[1]
PERTURB = {"phi_lo": {"phi": 0.870}, "phi_hi": {"phi": 0.996}, "u_lo": {"u_sup": 0.80}, "u_hi": {"u_sup": 0.90},
           "amax_lo": {"amax_sup": 1.05}, "amax_hi": {"amax_sup": 1.15}, "ramp_lo": {"ramp": 14}, "ramp_hi": {"ramp": 35},
           "fleet_lo": {"fleet": 120}, "fleet_hi": {"fleet": 360}, "rt_lo": {"round_trip": 0.8 * 77}, "rt_hi": {"round_trip": 1.2 * 77}}
if __name__ == "__main__":
    split = sys.argv[1]
    tuned = json.load(open(ROOT / "results/tuned_configs.json"))
    man = json.load(open(ROOT / f"results/{split}_manifest.json"))
    sub = "j30" if split == "dev" else "j60"
    if split == "dev":
        eps_all = EPISODES_DEV + EPISODES_TEST
    insts = [str(ROOT / "data/raw/psplib" / sub / i) for i in man["instances"][:6]]
    eps = (EPISODES_DEV + EPISODES_TEST) if split == "dev" else EPISODES_TEST
    seeds = man["seeds"][:3]
    methods = ["buffer", "detmpc", "saampc", "carp"] + [[f"carp_{k}", "carp"] for k in PERTURB]
    cfg = {m: tuned[m] for m in ("buffer", "detmpc", "saampc", "carp")}
    for k, v in PERTURB.items():
        cfg[f"carp_{k}"] = dict(tuned["carp"], mis=v)
    if split == "dev":   # design-choice comparison on the same subset (protocol amendment 6)
        methods += [["carp_norec", "carp"], ["carp_defer", "carp"], ["carp_antic", "carp"]]
        cfg["carp_norec"] = dict(tuned["carp"], recourse=False); cfg["carp_defer"] = dict(tuned["carp"], defer=True)
        cfg["carp_antic"] = dict(tuned["carp"], groups=12)
    out = ROOT / f"results/misspec_{split}.jsonl"
    if out.exists():
        raise SystemExit("exists")
    json.dump(dict(stage=f"misspec_{split}", instances=[Path(i).name for i in insts], episodes=eps, seeds=seeds, configs=cfg,
                   started=datetime.datetime.now().isoformat(timespec="seconds")), open(ROOT / f"results/misspec_{split}_manifest.json", "w"), indent=1)
    run([(i, e, s, [tuple(m) if isinstance(m, list) else m for m in methods], cfg, {}) for i in insts for e in eps for s in seeds], out, workers=9)
