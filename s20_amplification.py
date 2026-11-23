"""Matched disruption experiments (compound-shock amplification). For each development instance, seed and spacing,
the same world is run with no shock, the first shock only, the second shock only and both shocks (identical shock
parameters and random numbers), under no recovery action and under the cascade-aware planner. Amplification ratio
= excess loss of the combined episode / (excess loss of the first + excess loss of the second), excess measured
against the no-shock run. Also records the backlog the first shock leaves at the onset of the second."""
import sys, json, time, hashlib, datetime
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
SPACINGS = [0, 30, 60, 90, 120, 180, 240]
COMPOUNDS = ["covid+evergiven", "covid+ruua", "redsea+production"]


def one(args):
    inst_path, ep, seed, spacing, methods, cfgs = args
    from model.psplib import parse
    from model.network import build_project, make_world, SUP
    from model.sim import Sim, make_policy
    proj = build_project(parse(inst_path))
    rng = np.random.default_rng([seed, 17])
    onset = int(rng.integers(proj.T0 - 60, proj.T0 + int(0.25 * proj.C0)))
    out = []
    worlds = {c: make_world(proj, ep, seed, onset=onset, spacing=spacing, components=c)
              for c in [(), ("A",), ("B",), ("A", "B")]}
    # backlog left by the first shock at the onset of the second (mean over supplier and lane nodes it reaches)
    wA = worlds[("A",)]; tB = onset + spacing
    nodes = [n for n in wA.names if n.startswith("S|") or n.startswith("LANE")]
    backlog = float(np.mean([wA.Q[n][min(tB, len(wA.Q[n]) - 1)] for n in nodes]))
    for m in methods:
        rec = dict(instance=Path(inst_path).stem, episode=ep, seed=seed, spacing=spacing, onset=onset, method=m,
                   backlog_at_B=backlog)
        for c, w in worlds.items():
            r = Sim(proj, w, make_policy(m, dict(cfgs.get(m, {})))).run()
            key = {(): "L0", ("A",): "LA", ("B",): "LB", ("A", "B"): "LAB"}[c]
            rec[key] = float(r["loss"]); rec[key.replace("L", "D")] = float(r["delay"])
        out.append(rec)
    return out


if __name__ == "__main__":
    dev = json.load(open(ROOT / "results" / "dev_manifest.json"))
    tuned = json.load(open(ROOT / "results" / "tuned_configs.json"))
    insts = [str(ROOT / "data/raw/psplib/j30" / i) for i in dev["instances"][:8]]
    methods = ["naive", "carp"]
    cfgs = {m: tuned.get(m, {}) for m in methods}
    tasks = [(i, ep, s, d, methods, cfgs) for i in insts for ep in COMPOUNDS for s in range(401, 404) for d in SPACINGS]
    outp = ROOT / "results" / "amplification.jsonl"
    if outp.exists():
        raise SystemExit("exists")
    t0 = time.time()
    with Pool(9) as pool, open(outp, "w") as f:
        for i, res in enumerate(pool.imap_unordered(one, tasks)):
            for r in res:
                f.write(json.dumps(r) + "\n")
    json.dump(dict(stage="amplification", instances=[Path(i).name for i in insts], spacings=SPACINGS, compounds=COMPOUNDS,
                   seeds=list(range(401, 404)), methods=methods, configs=cfgs, finished=datetime.datetime.now().isoformat(timespec="seconds"),
                   code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "code/model").glob("*.py")}),
              open(ROOT / "results" / "amplification_manifest.json", "w"), indent=1)
    print("done", time.time() - t0)
