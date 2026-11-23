"""PSPLIB single-mode RCPSP parser, serial schedule generation (LFT rule) and resource-flow network."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Instance:
    name: str
    n: int                      # jobs incl. dummies
    dur: np.ndarray             # periods
    req: np.ndarray             # n x K
    cap: np.ndarray             # K
    succ: list                  # list of lists (0-based)
    duedate: int
    tardcost: float
    pred: list = field(default_factory=list)

    def __post_init__(self):
        self.pred = [[] for _ in range(self.n)]
        for i, ss in enumerate(self.succ):
            for j in ss:
                self.pred[j].append(i)


def parse(path: str | Path) -> Instance:
    txt = Path(path).read_text()
    lines = txt.splitlines()
    n = int(re.search(r"jobs \(incl\. supersource/sink \):\s*(\d+)", txt).group(1))
    i = [k for k, l in enumerate(lines) if l.startswith("PROJECT INFORMATION")][0]
    vals = lines[i + 2].split()
    duedate, tardcost = int(vals[3]), float(vals[4])
    i = [k for k, l in enumerate(lines) if l.startswith("PRECEDENCE RELATIONS")][0] + 2
    succ = []
    for k in range(n):
        v = list(map(int, lines[i + k].split()))
        succ.append([s - 1 for s in v[3:3 + v[2]]])
    i = [k for k, l in enumerate(lines) if l.startswith("REQUESTS/DURATIONS")][0] + 3
    dur, req = [], []
    for k in range(n):
        v = list(map(int, lines[i + k].split()))
        dur.append(v[2]); req.append(v[3:])
    i = [k for k, l in enumerate(lines) if l.startswith("RESOURCEAVAILABILITIES")][0] + 2
    cap = np.array(list(map(int, lines[i].split())))
    return Instance(Path(path).stem, n, np.array(dur), np.array(req), cap, succ, duedate, tardcost)


def topo_order(inst: Instance) -> list:
    indeg = [len(p) for p in inst.pred]
    order, stack = [], [j for j in range(inst.n) if indeg[j] == 0]
    while stack:
        j = stack.pop(); order.append(j)
        for s in inst.succ[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    return order


def cpm_lft(inst: Instance) -> np.ndarray:
    order = topo_order(inst)
    es = np.zeros(inst.n, int)
    for j in order:
        for s in inst.succ[j]:
            es[s] = max(es[s], es[j] + inst.dur[j])
    T = int((es + inst.dur).max())
    lf = np.full(inst.n, T, int)
    for j in reversed(order):
        for s in inst.succ[j]:
            lf[j] = min(lf[j], lf[s] - inst.dur[s])
    return lf


def serial_sgs(inst: Instance) -> np.ndarray:
    """Serial schedule generation with the minimum latest-finish-time priority rule (Kolisch 1996)."""
    lf = cpm_lft(inst)
    horizon = int(inst.dur.sum()) + 1
    use = np.zeros((horizon + 1, len(inst.cap)), int)
    start = np.full(inst.n, -1, int)
    done = set()
    while len(done) < inst.n:
        elig = [j for j in range(inst.n) if j not in done and all(p in done for p in inst.pred[j])]
        j = min(elig, key=lambda x: (lf[x], x))
        t = max([start[p] + inst.dur[p] for p in inst.pred[j]], default=0)
        d = inst.dur[j]
        while d > 0 and np.any(use[t:t + d] + inst.req[j] > inst.cap):
            t += 1
        if d > 0:
            use[t:t + d] += inst.req[j]
        start[j] = t; done.add(j)
    return start


def resource_flow_arcs(inst: Instance, start: np.ndarray) -> set:
    """Resource-flow network of a feasible schedule (chronological allocation, Artigues et al. 2003):
    an arc i->j when i passes at least one resource unit to j. Precedence plus flow arcs give a
    partial-order schedule whose every precedence-feasible execution is resource feasible."""
    finish = start + inst.dur
    K = len(inst.cap)
    arcs = set()
    order = sorted(range(inst.n), key=lambda j: (start[j], j))
    # per resource, units held by activities (by finish time); source holds all units at time 0
    for k in range(K):
        holders = [[-1, int(inst.cap[k]), 0]]  # [activity, units, finish]
        for j in order:
            r = int(inst.req[j, k])
            if r == 0:
                continue
            need = r
            # take from holders finished by start[j], prefer precedence predecessors, then latest finish
            cand = [h for h in holders if h[1] > 0 and h[2] <= start[j]]
            cand.sort(key=lambda h: (0 if h[0] in inst.pred[j] else 1, -h[2]))
            for h in cand:
                if need == 0:
                    break
                q = min(need, h[1]); h[1] -= q; need -= q
                if h[0] >= 0 and h[0] != j:
                    arcs.add((h[0], j))
            assert need == 0, (inst.name, j, k)
            holders.append([j, r, int(finish[j])])
    prec = {(i, j) for i in range(inst.n) for j in inst.succ[i]}
    return arcs - prec
