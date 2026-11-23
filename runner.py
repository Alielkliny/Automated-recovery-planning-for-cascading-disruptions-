"""Parallel experiment runner: one task = (instance file, episode, seed); every method runs on the same world
(common random numbers). Results are appended as JSON lines."""
import sys, os, json, time, hashlib, argparse
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")


def task(args):
    inst_path, episode, seed, methods, cfgs, kw = args
    import numpy as np
    from model.psplib import parse
    from model.network import build_project, make_world
    from model.sim import Sim, make_policy
    from model import params as PR
    saved = {}
    for key, val in kw.get("params", {}).items():   # sensitivity overrides, restored after the task
        if key.endswith("_X"):
            name = key[:-2]; saved[name] = dict(getattr(PR, name))
            setattr(PR, name, {k: max(1, int(round(v * val))) for k, v in getattr(PR, name).items()})
        else:
            saved[key] = getattr(PR, key); setattr(PR, key, val)
    try:
        return _task(inst_path, episode, seed, methods, cfgs, kw)
    finally:
        for k, v in saved.items():
            setattr(PR, k, v)


def _task(inst_path, episode, seed, methods, cfgs, kw):
    import numpy as np
    from model.psplib import parse
    from model.network import build_project, make_world
    from model.sim import Sim, make_policy
    inst = parse(inst_path)
    proj = build_project(inst, mat_fraction=kw.get("mat_fraction"))
    world = make_world(proj, episode, seed, **kw.get("world", {}))
    out = []
    for m in methods:
        label, base = (m if isinstance(m, (list, tuple)) else (m, m))
        cfg = dict(cfgs.get(label, {}))
        t0 = time.time()
        r = Sim(proj, world, make_policy(base, cfg)).run()
        r = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}
        r.update(instance=Path(inst_path).stem, episode=episode, seed=seed, method=label, base=base, cfg=json.dumps(cfg, sort_keys=True),
                 secs=round(time.time() - t0, 3), n_lots=len(proj.lots), C0=proj.C0, tardcost=inst.tardcost,
                 variant=kw.get("variant", ""))
        out.append(r)
    return out


def run(tasks, out_path, workers=9):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); n = 0
    with Pool(workers) as pool, open(out_path, "a") as f:
        for res in pool.imap_unordered(task, tasks, chunksize=1):
            for r in res:
                f.write(json.dumps(r) + "\n")
            n += 1
            if n % max(1, len(tasks) // 20) == 0:
                el = time.time() - t0
                print(f"{n}/{len(tasks)} tasks, {el/60:.1f} min, eta {el/n*(len(tasks)-n)/60:.1f} min", flush=True)
    return out_path
