import sys, time, numpy as np
sys.path.insert(0, ".")
from model.psplib import parse
from model.network import build_project, make_world
from model.sim import Sim, make_policy, METHODS
inst = parse("../data/raw/psplib/j30/j301_1.sm")
proj = build_project(inst)
print("lots", len(proj.lots), "C0", proj.C0, "pkgs", len(proj.pkg_cat), [(l.cat, l.region, l.need, l.plan_order) for l in proj.lots][:6])
for ep in ["covid", "evergiven", "ruua", "covid+ruua"]:
    w = make_world(proj, ep, seed=1)
    print(ep, [(s.kind, s.region, s.start, s.dur, round(s.level,2)) for s in w.shocks])
    for m in METHODS:
        t0 = time.time(); r = Sim(proj, w, make_policy(m)).run()
        print(f"  {m:9s} loss={r['loss']:8.1f} delay={r['delay']:4d} pen={r['penalty']:7.1f} int={r['intervention']:6.1f} hold={r['holding']:6.1f} idle={r['idle']:5.0f} done={r['completed']} acts={[r[k] for k in ['n_alt','n_cape','n_air','n_ahead','n_defer','n_forced']]} {time.time()-t0:.2f}s")
