"""One-at-a-time sensitivity of the assumed parameters on a development subset: 4 instances, the seven
episode types, 2 replications, for the planner, the deterministic and sample-average MPC, the static safety lead time
and no action."""
import sys, json, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import run
from model.network import EPISODES_DEV, EPISODES_TEST
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
 "standby 0.25": {"STANDBY": 0.25}, "standby 1.0": {"STANDBY": 1.0},
 "lot value 50": {"BASE_VALUE": 50.0}, "lot value 200": {"BASE_VALUE": 200.0},
 "Cape detour 10 d": {"CAPE_EXTRA": 10}, "Cape detour 18 d": {"CAPE_EXTRA": 18},
 "approval x0.5": {"APPROVAL_X": 0.5}, "approval x1.5": {"APPROVAL_X": 1.5},
 "manufacturing x0.75": {"PI_X": 0.75}, "manufacturing x1.25": {"PI_X": 1.25},
 "supplier utilization 0.75-0.85": {"SUP_U_LO": 0.75, "SUP_U_HI": 0.85}, "supplier utilization 0.85-0.95": {"SUP_U_LO": 0.85, "SUP_U_HI": 0.95},
 "congestion depth 0.10": {"CONGESTION_DEPTH": 0.10}, "congestion depth 0.25": {"CONGESTION_DEPTH": 0.25},
 "demand surge 0.20": {"DEMAND_SURGE_SUP": 0.20}, "demand surge 0.40": {"DEMAND_SURGE_SUP": 0.40},
 "period 5 days": {"DAYS_PER_PERIOD": 5},
 "material share 0.25": {"mat": 0.25}, "material share 0.55": {"mat": 0.55},
}
if __name__ == "__main__":
    tuned = json.load(open(ROOT / "results/tuned_configs.json"))
    man = json.load(open(ROOT / "results/dev_manifest.json"))
    insts = [str(ROOT / "data/raw/psplib/j30" / i) for i in man["instances"][10:14]]
    methods = ["naive", "buffer", "detmpc", "saampc", "carp"]
    cfg = {m: tuned.get(m, {}) for m in methods}
    tasks = []
    for vname, ov in VARIANTS.items():
        params = {k: v for k, v in ov.items() if k != "mat"}
        kw = {"params": params, "variant": vname}
        if "mat" in ov:
            kw["mat_fraction"] = ov["mat"]
        tasks += [(i, e, s, methods, cfg, kw) for i in insts for e in EPISODES_DEV + EPISODES_TEST for s in (111, 112)]
    out = ROOT / "results/sensitivity.jsonl"
    if out.exists():
        raise SystemExit("exists")
    json.dump(dict(stage="sensitivity", variants=VARIANTS, instances=[Path(i).name for i in insts], seeds=[111, 112],
                   configs=cfg, started=datetime.datetime.now().isoformat(timespec="seconds")), open(ROOT / "results/sensitivity_manifest.json", "w"), indent=1)
    run(tasks, out, workers=9)
