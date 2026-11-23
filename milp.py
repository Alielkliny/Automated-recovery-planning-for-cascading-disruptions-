"""Recovery-action MILP solved with HiGHS (highspy). One binary per lot and candidate action, one binary per
package and alternative region (qualification), one optional binary per activity deferral; scenario copies of the
activity start times over precedence and resource-flow arcs; expected delay penalty, holding cost and an optional
CVaR term on the delay penalty (Rockafellar and Uryasev 2000)."""
from __future__ import annotations
import numpy as np
import highspy
from scipy import sparse

BIG = 5000.0
_H = None


def _solver():
    """One HiGHS instance per process, reused across solves (constructing one costs about 20 ms)."""
    global _H
    if _H is None:
        _H = highspy.Highs()
        _H.setOptionValue("output_flag", False)
        _H.setOptionValue("threads", 1)
        _H.setOptionValue("random_seed", 0)
    return _H


class Builder:
    def __init__(self):
        self.lb, self.ub, self.c, self.integ = [], [], [], []
        self.rows, self.cols, self.vals, self.rlo, self.rhi = [], [], [], [], []
        self.nr = 0

    def var(self, lb=0.0, ub=np.inf, c=0.0, integer=False):
        self.lb.append(lb); self.ub.append(ub); self.c.append(c); self.integ.append(integer)
        return len(self.lb) - 1

    def row(self, terms, lo=-np.inf, hi=np.inf):
        for j, v in terms:
            self.rows.append(self.nr); self.cols.append(j); self.vals.append(v)
        self.rlo.append(lo); self.rhi.append(hi); self.nr += 1

    def solve(self, time_limit=5.0, gap=1e-3):
        n = len(self.lb)
        A = sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=(self.nr, n))
        lp = highspy.HighsLp()
        lp.num_col_ = n; lp.num_row_ = self.nr
        lp.col_cost_ = np.array(self.c, float)
        lp.col_lower_ = np.array(self.lb, float); lp.col_upper_ = np.array(self.ub, float)
        lp.row_lower_ = np.array(self.rlo, float); lp.row_upper_ = np.array(self.rhi, float)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        lp.integrality_ = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
                           for i in self.integ]
        h = _solver()
        h.clearModel()
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", float(gap))
        h.passModel(lp)
        h.run()
        sol = h.getSolution().col_value
        return np.array(sol) if len(sol) == n else None


def solve_recovery(t, proj, act_state, lot_opts, qual_cost, defer_cands, pen_day, K, lam=0.0, alpha=0.9,
                   time_limit=5.0, recourse=False, group=None):
    """act_state: dict with 'unstarted' (list), 'fin' (activity -> known finish day, absolute), 'arcs' (list of
    (i, j) precedence and active flow arcs), 'defer_arcs' (j -> list of deferrable successors).
    lot_opts: list of (lot, [(opt, cost, A_now (K,), A_delayed (K,))], hold_per_day, late_cost, need_rel).
    With recourse, the first option ('keep') means waiting one epoch: in each scenario a continuous recourse share
    then takes any option with its arrival when activated one epoch later (two-stage structure).
    Returns chosen option per lot index and the set of deferred activities."""
    B = Builder()
    U = act_state["unstarted"]; Uset = set(U)
    D = proj.dur
    C0rel = proj.T0 + proj.C0 - t
    S = {(j, k): B.var(0.0) for j in U for k in range(K)}
    T = [B.var(0.0, c=pen_day / K * (1.0 if lam == 0 else (1.0))) for k in range(K)]
    if lam > 0:
        eta = B.var(-np.inf, c=lam)
        v = [B.var(0.0, c=lam / ((1 - alpha) * K)) for _ in range(K)]
        for k in range(K):
            B.row([(v[k], 1.0), (T[k], -pen_day), (eta, 1.0)], lo=0.0)
    X = {}
    Y = {}       # recourse: (lot, option, scenario) -> continuous share, only under the wait option
    qual = {}
    for lot, opts, hold, *_ in lot_opts:
        xs = []
        for oi, row in enumerate(opts):
            opt, cost = row[0], row[1]
            x = B.var(0.0, 1.0, c=cost, integer=True)
            X[(lot.idx, oi)] = x; xs.append(x)
            if opt.startswith("alt:"):
                key = (lot.pkg, opt[4:])
                if key in qual_cost:
                    if key not in qual:
                        qual[key] = B.var(0.0, 1.0, c=qual_cost[key], integer=True)
                    B.row([(qual[key], 1.0), (x, -1.0)], lo=0.0)
        B.row([(x, 1.0) for x in xs], lo=1.0, hi=1.0)
        if recourse and len(opts) > 1 and len(opts[0]) > 3:
            xw = X[(lot.idx, 0)]                      # 'keep' becomes 'wait one epoch, then decide'
            grp = list(range(K)) if group is None else list(group)
            G = max(grp) + 1
            for g in range(G):
                ys = []
                share = grp.count(g) / K
                for oi, row in enumerate(opts):
                    opt, cost = row[0], row[1]
                    qc = qual_cost.get((lot.pkg, opt[4:]), 0.0) if opt.startswith("alt:") else 0.0
                    y = B.var(0.0, 1.0, c=(cost + qc) * share)
                    ys.append(y)
                    for k in range(K):
                        if grp[k] == g:
                            Y[(lot.idx, oi, k)] = y
                B.row([(y, 1.0) for y in ys] + [(xw, -1.0)], lo=0.0, hi=0.0)
    Z = {j: B.var(0.0, 1.0, c=cz, integer=True) for j, cz in defer_cands.items()}
    fin = act_state["fin"]
    for k in range(K):
        lbk = 0.0
        for (i, j) in act_state["arcs"]:
            if j not in Uset:
                continue
            if i in Uset:
                if j in act_state["defer_arcs"].get(i, ()) and i in Z:
                    B.row([(S[(j, k)], 1.0), (S[(i, k)], -1.0), (Z[i], BIG)], lo=float(D[i]))
                    B.row([(S[(i, k)], 1.0), (S[(j, k)], -1.0), (Z[i], -BIG)], lo=float(D[j]) - BIG)
                else:
                    B.row([(S[(j, k)], 1.0), (S[(i, k)], -1.0)], lo=float(D[i]))
            else:
                f = fin.get(i)
                if f is not None and f - t > 0:
                    B.row([(S[(j, k)], 1.0)], lo=float(f - t))
        for j in U:
            B.row([(T[k], 1.0), (S[(j, k)], -1.0)], lo=float(D[j]) - C0rel)
        known = [f - t for f in fin.values() if f is not None]
        if known:
            B.row([(T[k], 1.0)], lo=float(max(known)) - C0rel)
    for lot, opts, hold, late_c, need_rel in lot_opts:
        j = lot.act
        if j not in Uset:
            continue
        rec = recourse and len(opts) > 1 and len(opts[0]) > 3
        for k in range(K):
            arr_terms = []
            for oi, row in enumerate(opts):
                a_now = float(min(row[2][k], 4000.0))
                if rec and oi == 0:
                    for oj, row2 in enumerate(opts):
                        ad = float(min(row2[3][k], 4000.0))
                        arr_terms.append((Y[(lot.idx, oj, k)], ad))
                else:
                    arr_terms.append((X[(lot.idx, oi)], a_now))
            if late_c > 0:
                Lv = B.var(0.0, c=late_c / K)
                B.row([(Lv, 1.0)] + [(v, -a) for v, a in arr_terms], lo=-float(need_rel))
            B.row([(S[(j, k)], 1.0)] + [(v, -a) for v, a in arr_terms], lo=0.0)
            for v, a in arr_terms:
                B.c[v] -= hold * a / K
            B.c[S[(j, k)]] += hold / K
    sol = B.solve(time_limit=time_limit)
    if sol is None:
        return None, set()
    choice = {}
    for lot, opts, *_ in lot_opts:
        best = max(range(len(opts)), key=lambda oi: sol[X[(lot.idx, oi)]])
        choice[lot.idx] = opts[best][0]
    deferred = {j for j, z in Z.items() if sol[z] > 0.5}
    return choice, deferred


def _reduce_network(t, proj, act_state, lot_acts):
    """Exact reduction of the activity network to the activities that wait for imported lots (used when no
    deferral is possible, so the arcs are fixed). Returns the static release of each lot activity (relative days),
    the longest-path distance between lot activities (transitively reduced), each lot activity's longest path to
    completion, and the static lower bound on the project delay."""
    U = act_state["unstarted"]; Uset = set(U); D = proj.dur
    C0rel = proj.T0 + proj.C0 - t
    succ = {j: [] for j in U}; npred = {j: 0 for j in U}
    rel = {j: 0.0 for j in U}
    fin = act_state["fin"]
    for (i, j) in act_state["arcs"]:
        if j not in Uset:
            continue
        if i in Uset:
            succ[i].append(j); npred[j] += 1
        else:
            f = fin.get(i)
            if f is not None and f - t > rel[j]:
                rel[j] = float(f - t)
    order, stack = [], [j for j in U if npred[j] == 0]
    npc = dict(npred)
    while stack:
        i = stack.pop(); order.append(i)
        for j in succ[i]:
            npc[j] -= 1
            if npc[j] == 0:
                stack.append(j)
    for i in order:                                  # static earliest starts
        for j in succ[i]:
            if rel[i] + D[i] > rel[j]:
                rel[j] = rel[i] + D[i]
    tail = {j: 0.0 for j in U}                       # longest path from the start of j to completion
    for i in reversed(order):
        best = float(D[i])
        for j in succ[i]:
            if D[i] + tail[j] > best:
                best = D[i] + tail[j]
        tail[i] = best
    static_T = max([rel[j] + tail[j] for j in U] + [0.0]) - C0rel
    known = [f - t for f in fin.values() if f is not None]
    if known:
        static_T = max(static_T, max(known) - C0rel)
    pos = {j: k for k, j in enumerate(order)}
    L = sorted(set(lot_acts) & Uset, key=lambda j: pos[j])
    Lset = set(L)
    dist = {}
    for i in L:                                      # longest path from start of i to start of every successor
        d = {i: 0.0}
        for u in order[pos[i]:]:
            if u not in d:
                continue
            for v in succ[u]:
                if d[u] + D[u] > d.get(v, -1.0):
                    d[v] = d[u] + D[u]
        dist[i] = {j: d[j] for j in L if j != i and j in d}
    pairs = []
    for i in L:
        for j, dij in dist[i].items():
            implied = any(dist[i].get(k) is not None and dist[k].get(j) is not None and dist[i][k] + dist[k][j] >= dij
                          for k in L if k != i and k != j)
            if not implied:
                pairs.append((i, j, dij))
    return rel, pairs, tail, static_T, L


def solve_recovery_reduced(t, proj, act_state, lot_opts, qual_cost, pen_day, K, lam=0.0, alpha=0.9,
                           time_limit=5.0, recourse=False, group=None):
    """Same model as solve_recovery without deferral, on the reduced network (identical optimal objective)."""
    lot_acts = [lot.act for lot, *_ in lot_opts]
    rel, pairs, tail, static_T, L = _reduce_network(t, proj, act_state, lot_acts)
    B = Builder()
    C0rel = proj.T0 + proj.C0 - t
    S = {(j, k): B.var(rel[j]) for j in L for k in range(K)}
    T = [B.var(max(0.0, static_T), c=pen_day / K) for k in range(K)]
    if lam > 0:
        eta = B.var(-np.inf, c=lam)
        v = [B.var(0.0, c=lam / ((1 - alpha) * K)) for _ in range(K)]
        for k in range(K):
            B.row([(v[k], 1.0), (T[k], -pen_day), (eta, 1.0)], lo=0.0)
    X, Y, qual = {}, {}, {}
    for lot, opts, hold, *_ in lot_opts:
        xs = []
        for oi, row in enumerate(opts):
            opt, cost = row[0], row[1]
            x = B.var(0.0, 1.0, c=cost, integer=True)
            X[(lot.idx, oi)] = x; xs.append(x)
            if opt.startswith("alt:"):
                key = (lot.pkg, opt[4:])
                if key in qual_cost:
                    if key not in qual:
                        qual[key] = B.var(0.0, 1.0, c=qual_cost[key], integer=True)
                    B.row([(qual[key], 1.0), (x, -1.0)], lo=0.0)
        B.row([(x, 1.0) for x in xs], lo=1.0, hi=1.0)
        if recourse and len(opts) > 1 and len(opts[0]) > 3:
            xw = X[(lot.idx, 0)]
            grp = list(range(K)) if group is None else list(group)
            G = max(grp) + 1
            for g in range(G):
                ys = []
                share = grp.count(g) / K
                for oi, row in enumerate(opts):
                    opt, cost = row[0], row[1]
                    qc = qual_cost.get((lot.pkg, opt[4:]), 0.0) if opt.startswith("alt:") else 0.0
                    y = B.var(0.0, 1.0, c=(cost + qc) * share)
                    ys.append(y)
                    for k in range(K):
                        if grp[k] == g:
                            Y[(lot.idx, oi, k)] = y
                B.row([(y, 1.0) for y in ys] + [(xw, -1.0)], lo=0.0, hi=0.0)
    for k in range(K):
        for (i, j, dij) in pairs:
            B.row([(S[(j, k)], 1.0), (S[(i, k)], -1.0)], lo=float(dij))
        for j in L:
            B.row([(T[k], 1.0), (S[(j, k)], -1.0)], lo=float(tail[j]) - C0rel)
    Lset = set(L)
    for lot, opts, hold, late_c, need_rel in lot_opts:
        j = lot.act
        if j not in Lset:
            continue
        rec = recourse and len(opts) > 1 and len(opts[0]) > 3
        for k in range(K):
            arr_terms = []
            for oi, row in enumerate(opts):
                a_now = float(min(row[2][k], 4000.0))
                if rec and oi == 0:
                    for oj, row2 in enumerate(opts):
                        arr_terms.append((Y[(lot.idx, oj, k)], float(min(row2[3][k], 4000.0))))
                else:
                    arr_terms.append((X[(lot.idx, oi)], a_now))
            if late_c > 0:
                Lv = B.var(0.0, c=late_c / K)
                B.row([(Lv, 1.0)] + [(vv, -a) for vv, a in arr_terms], lo=-float(need_rel))
            B.row([(S[(j, k)], 1.0)] + [(vv, -a) for vv, a in arr_terms], lo=0.0)
            for vv, a in arr_terms:
                B.c[vv] -= hold * a / K
            B.c[S[(j, k)]] += hold / K
    sol = B.solve(time_limit=time_limit)
    if sol is None:
        return None, set()
    choice = {}
    for lot, opts, *_ in lot_opts:
        best = max(range(len(opts)), key=lambda oi: sol[X[(lot.idx, oi)]])
        choice[lot.idx] = opts[best][0]
    return choice, set()
