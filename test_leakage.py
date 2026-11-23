"""Leakage check: a planner's forecasts and decisions at day t depend only on the world up to t.
Two worlds share the history up to t and differ afterwards; every forecaster must return identical arrivals."""
import sys, copy, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.psplib import parse
from model.network import build_project, make_world, queue_path
from model.forecast import Forecaster, Obs
from model.sim import Sim, make_policy

INST = Path(__file__).resolve().parents[2] / "data" / "raw" / "psplib" / "j30" / "j301_1.sm"


def _fork(world, t):
    w2 = copy.deepcopy(world)
    rng = np.random.default_rng(1)
    for n in w2.names:
        w2.a[n][t + 1:] = rng.uniform(0.0, 1.0, size=len(w2.a[n]) - t - 1)
        w2.upath[n][t + 1:] = rng.uniform(0.5, 1.2, size=len(w2.upath[n]) - t - 1)
        w2.Q[n], w2.cum[n] = queue_path(w2.a[n], w2.upath[n], w2.amax[n], 0.0)
    w2.divert[t + 1:] = rng.uniform(size=len(w2.divert) - t - 1)
    return w2


def test_forecasts_ignore_future():
    proj = build_project(parse(INST))
    res = {}
    for t in (280, 350):
        w1 = make_world(proj, "covid+ruua", 4)
        w2 = _fork(w1, t)
        s1 = Sim(proj, w1, make_policy("naive")); s1.pol.setup(s1)
        for lot in s1.lots:
            if lot.plan_order < t:
                s1._order(lot, lot.plan_order)
        s1.t = t
        names = s1.names
        for kind in ("carp", "onoff", "det", "iid"):
            f1 = Forecaster(kind, K=6, seed=3); f1.prepare(Obs(w1, t, names), names)
            f2 = Forecaster(kind, K=6, seed=3); f2.prepare(Obs(w2, t, names), names)
            for lot in s1.lots:
                if lot.ordered and lot.t_arr <= t:
                    continue
                for o in ("keep", "alt:EU"):
                    a1 = f1.arrival(lot, o, proj, {}); a2 = f2.arrival(lot, o, proj, {})
                    assert np.allclose(a1, a2), (kind, t, lot.idx, o)
            res[f"{kind}@{t}"] = "identical"
    out = Path(__file__).resolve().parents[2] / "probe" / "leakage_check.json"
    out.write_text(json.dumps({"test": "forecasts at day t are identical for worlds that share history to t",
                               "result": "PASS", "cases": res}, indent=1))
