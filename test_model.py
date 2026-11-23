"""Verification tests: bounded stocks, conservation, boundary and extreme conditions, determinism."""
import sys, copy
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.psplib import parse, serial_sgs, resource_flow_arcs
from model.network import build_project, make_world, queue_path, exit_time, INF, World
from model.sim import Sim, make_policy, METHODS

INST = Path(__file__).resolve().parents[2] / "data" / "raw" / "psplib" / "j30" / "j301_1.sm"


@pytest.fixture(scope="module")
def proj():
    return build_project(parse(INST))


def test_sgs_feasible():
    inst = parse(INST); st = serial_sgs(inst)
    for i in range(inst.n):
        for j in inst.succ[i]:
            assert st[j] >= st[i] + inst.dur[i]
    T = int((st + inst.dur).max())
    for t in range(T):
        use = sum(inst.req[j] for j in range(inst.n) if st[j] <= t < st[j] + inst.dur[j])
        assert np.all(use <= inst.cap)


def test_queue_bounds():
    a = np.ones(100); a[10:30] = 0.2
    Q, cum = queue_path(a, np.full(100, 0.85), 1.1, 0.0)
    assert Q.min() >= 0 and np.all(np.diff(cum) >= 0)
    # surge multiplies the reduced capacity once a backlog exists: first day 0.65, then 0.85 - 0.22 per day
    assert Q[30] == pytest.approx(0.65 + 19 * (0.85 - 0.2 * 1.1), rel=1e-6)
    # backlog drains at amax - u per day once capacity is restored
    assert Q[31] == pytest.approx(Q[30] - (1.1 - 0.85), rel=1e-6)


def test_exit_time_infinite_when_capacity_zero():
    a = np.zeros(50); Q, cum = queue_path(a, np.full(50, 0.8), 1.0, 1.0)
    assert exit_time(Q, cum, 5) >= INF


def _calm_world(proj, seed=3):
    w = make_world(proj, "evergiven", seed)
    for n in w.names:  # remove every shock: nominal capacity, nominal intake
        w.a[n][:] = 1.0
        w.upath[n][:] = w.u[n]
        w.Q[n], w.cum[n] = queue_path(w.a[n], w.upath[n], w.amax[n], 0.0)
    w.divert[:] = 0.0; w.shocks = []
    return w


@pytest.mark.parametrize("m", METHODS)
def test_no_disruption_no_delay(proj, m):
    w = _calm_world(proj)
    r = Sim(proj, w, make_policy(m)).run()
    assert r["completed"]
    assert r["delay"] == 0
    assert r["idle"] == 0


@pytest.mark.parametrize("m", ["naive", "carp", "practice"])
def test_deterministic(proj, m):
    w = make_world(proj, "covid+ruua", 11)
    r1 = Sim(proj, w, make_policy(m)).run(); r2 = Sim(proj, w, make_policy(m)).run()
    assert r1 == r2


def test_lot_conservation(proj):
    w = make_world(proj, "covid", 5)
    s = Sim(proj, w, make_policy("carp")); r = s.run()
    assert all(l.ordered and l.t_arr < INF for l in s.lots)
    for l in s.lots:
        assert s.start[l.act] >= l.t_arr  # no activity starts before its material arrives
        assert l.t_qexit <= l.t_prod <= l.t_arr


def test_precedence_and_resources_respected(proj):
    w = make_world(proj, "covid+evergiven", 7)
    s = Sim(proj, w, make_policy("carp")); s.run()
    inst = proj.inst
    for i in range(inst.n):
        for j in inst.succ[i]:
            assert s.start[j] >= s.fin[i]
    T = max(s.fin.values())
    for t in range(proj.T0, T):
        use = sum(inst.req[j] for j in range(inst.n) if s.start[j] <= t < s.fin[j])
        assert np.all(use <= inst.cap)


def test_exit_forces_replacement(proj):
    w = make_world(proj, "ruua", 2)
    r = Sim(proj, w, make_policy("naive")).run()
    assert r["completed"]
