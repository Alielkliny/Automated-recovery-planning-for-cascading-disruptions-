"""Execution simulator (daily clock, weekly review epochs) and the recovery policies compared in the study."""
from __future__ import annotations
import copy
import numpy as np

from . import params as PR
from .network import INF, SUP, passage, node_names, Lot
from .forecast import Forecaster, Obs
from .milp import solve_recovery, solve_recovery_reduced

OPT_SLOT = {"std": 0, "cape": 1, "air": 2}
REG_SLOT = {r: 3 + i for i, r in enumerate(["EA", "SA", "EU", "TR", "OT"])}
METAL = {"structural steel", "reinforcing bars", "aluminium structures", "cables"}


class Sim:
    def __init__(self, proj, world, policy):
        self.p, self.w, self.pol = proj, world, policy
        self.lots = [copy.copy(l) for l in proj.lots]
        self.names = node_names(proj)
        inst = proj.inst
        self.prec = {(i, j) for i in range(inst.n) for j in inst.succ[i]}
        self.arcs = set(self.prec) | set(proj.flow)
        self.start, self.fin = {}, {}
        self.qualified = {}          # (pkg, region) -> day the qualification is complete
        self.q_at_order = {}
        self.costs = dict(alt=0.0, qual=0.0, cancel=0.0, cape=0.0, air=0.0, reseq=0.0)
        self.n_actions = dict(alt=0, cape=0, air=0, ahead=0, defer=0, forced=0)
        self.pending_defer = []
        self.deferred = set()
        self.approvals = []          # completion days of substitution approvals in progress
        self.ready_day = {}
        self.t = 0
        self.prio = sorted(range(inst.n), key=lambda j: (proj.start0[j], j))

    # ----------------------------------------------------------------- lot actions
    def _order(self, lot, day, slot=None):
        lot.ordered = True; lot.order_day = int(day)
        slot = OPT_SLOT[lot.route] if slot is None else slot
        sup = SUP(lot.region, lot.cat)
        self.q_at_order[lot.idx] = float(self.w.Q[sup][min(int(day), len(self.w.Q[sup]) - 1)])
        q, pr, sh, ar, dv = passage(self.w, lot, int(day), lot.region, lot.route, slot)
        lot.t_qexit, lot.t_prod, lot.t_ship, lot.t_arr, lot.diverted = q, pr, sh, ar, dv

    def act_alt(self, lot, region, forced=False):
        t = self.t
        if lot.ordered and lot.t_qexit <= t:
            return False
        key = (lot.pkg, region)
        esc = self.w.esc[t] if lot.cat in METAL else 1.0
        self.costs["alt"] += PR.ALT_PREMIUM * lot.value * esc
        if lot.ordered:
            self.costs["cancel"] += 0.02 * lot.value
        if key not in self.qualified:
            busy = sum(1 for d in self.approvals if d > t)
            extra = 7 * max(0, busy - 1) // 2
            ready = t + PR.APPROVAL[lot.cat] + PR.ACTIVATION + extra
            self.qualified[key] = ready
            self.approvals.append(t + PR.APPROVAL[lot.cat] + extra)
            self.costs["qual"] += PR.QUALIFY * sum(l.value for l in self.lots if l.pkg == lot.pkg)
        entry = max(t, self.qualified[key])
        lot.region, lot.route = region, "std"
        lot.switched += 1
        self._order(lot, entry, slot=REG_SLOT.get(region, 7))
        self.n_actions["forced" if forced else "alt"] += 1
        return True

    def act_route(self, lot, route):
        t = self.t
        if lot.route == route or (lot.ordered and lot.t_ship <= t):
            return False
        if route == "air" and lot.cat not in PR.AIR_OK:
            return False
        if route == "cape" and lot.region not in PR.SUEZ_REGIONS:
            return False
        lot.route = route
        self.costs[route] += (PR.CAPE_COST if route == "cape" else PR.AIR_COST) * lot.value
        self.n_actions[route] += 1
        if lot.ordered and lot.order_day <= t and lot.t_qexit <= t:
            q, pr, sh, ar, dv = passage(self.w, lot, lot.order_day, lot.region, route, OPT_SLOT[route],
                                        from_stage=(lot.t_qexit, lot.t_prod))
            lot.t_ship, lot.t_arr, lot.diverted = sh, ar, dv
        elif lot.ordered:
            self._order(lot, lot.order_day, slot=OPT_SLOT[route])
        return True

    def act_ahead(self, lot):
        if lot.ordered:
            return False
        lot.plan_order = self.t
        self._order(lot, self.t)
        self.n_actions["ahead"] += 1
        return True

    def act_defer(self, j):
        if j in self.start or j in self.deferred:
            return False
        self.pending_defer.append((self.t + PR.RESEQ_DELAY, j))
        self.deferred.add(j)
        self.costs["reseq"] += PR.RESEQ_COST * self.p.inst.tardcost
        self.n_actions["defer"] += 1
        return True

    def deferrable(self, j):
        out = []
        for (a, s) in list(self.arcs):
            if a != j or (a, s) in self.prec or s in self.start:
                continue
            if not self._reaches(j, s, skip=(j, s)):
                out.append(s)
        return out

    def _reaches(self, a, b, skip):
        succ = {}
        for (x, y) in self.arcs:
            if (x, y) != skip:
                succ.setdefault(x, []).append(y)
        seen, st = {a}, [a]
        while st:
            x = st.pop()
            for y in succ.get(x, ()):
                if y == b:
                    return True
                if y not in seen:
                    seen.add(y); st.append(y)
        return False

    def _apply_defer(self, j):
        """Resequencing: the deferred activity releases its outgoing resource-flow arcs, so the crews it would hand
        over can move on; it then starts when its material, predecessors and free resources allow."""
        if j in self.start:
            return
        succs = self.deferrable(j)
        preds = [p for (p, x) in self.arcs if x == j and (p, x) not in self.prec]
        for s in succs:
            self.arcs.discard((j, s))
            for p in preds:
                if p != s and not self._reaches(s, p, skip=None):
                    self.arcs.add((p, s))

    # ----------------------------------------------------------------- execution
    def _usage(self, d):
        u = np.zeros(len(self.p.inst.cap), int)
        for j, s in self.start.items():
            if s <= d < self.fin[j]:
                u += self.p.inst.req[j]
        return u

    def _start_ready(self, d):
        inst = self.p.inst
        pred = {}
        for (i, j) in self.arcs:
            pred.setdefault(j, []).append(i)
        started = False
        use = self._usage(d)
        for j in self.prio:
            if j in self.start:
                continue
            if j == 0:
                if d >= self.p.T0:
                    self.start[0] = self.p.T0; self.fin[0] = self.p.T0; started = True
                continue
            ok = all(i in self.fin and self.fin[i] <= d for i in pred.get(j, ()))
            if not ok:
                continue
            li = self.p.act_lot.get(j)
            free = not np.any(use + inst.req[j] > inst.cap)
            if li is not None:
                lot = self.lots[li]
                if not (lot.ordered and lot.t_arr <= d):
                    if free and j not in self.ready_day:
                        self.ready_day[j] = d
                    continue
            if not free:
                continue
            self.start[j] = d; self.fin[j] = d + int(self.p.dur[j]); started = True
            use = use + inst.req[j] if self.p.dur[j] > 0 else use
        return started

    def advance(self, t_end):
        d = self.t
        n = self.p.inst.n
        while d < t_end and len(self.start) < n:
            while self._start_ready(d):
                pass
            ev = [f for f in self.fin.values() if f > d]
            ev += [l.t_arr for l in self.lots if l.ordered and d < l.t_arr < INF]
            d = min(ev + [t_end])

    def forced_replacements(self):
        """A lot stuck at a supplier that has left the market is re-sourced once its activity is otherwise ready,
        in every method (the minimum response a contractor must make)."""
        t = self.t
        pred = {}
        for (i, j) in self.prec:
            pred.setdefault(j, []).append(i)
        for lot in self.lots:
            if lot.ordered and lot.t_arr >= INF:
                ready = all(i in self.fin and self.fin[i] <= t + 28 for i in pred.get(lot.act, ()))
                if ready and self.p.pkg_alts[lot.pkg]:
                    self.act_alt(lot, self.p.pkg_alts[lot.pkg][0], forced=True)

    def run(self):
        H = self.p.horizon - 10
        self.pol.setup(self)
        t = 0
        while t < H and len(self.start) < self.p.inst.n:
            self.t = t
            for (dday, j) in list(self.pending_defer):
                if dday <= t:
                    self._apply_defer(j); self.pending_defer.remove((dday, j))
            obs = Obs(self.w, t, self.names); obs._q_at_order = self.q_at_order
            self.pol.decide(self, obs)
            for lot in self.lots:
                if not lot.ordered and lot.plan_order < t + 7:
                    self._order(lot, max(lot.plan_order, t))
            self.forced_replacements()
            self.advance(t + 7)
            t += 7
        return self.metrics()

    def metrics(self):
        p = self.p
        done = len(self.start) == p.inst.n
        comp = max(self.fin.values()) if done else p.horizon
        delay = max(0, comp - (p.T0 + p.C0))
        hold = 0.0; idle = 0.0; ontime = 0
        pred = {}
        for (i, j) in self.prec:
            pred.setdefault(j, []).append(i)
        for lot in self.lots:
            s = self.start.get(lot.act, p.horizon)
            if lot.t_arr < INF:
                hold += PR.HOLD_RATE * lot.value * max(0, s - lot.t_arr)
            if lot.act in self.ready_day:
                idle += max(0, s - self.ready_day[lot.act])
            ontime += int(lot.t_arr <= lot.need)
        pen = p.pen_day * delay
        inter = sum(self.costs.values())
        standby = PR.STANDBY * p.pen_day * idle
        return dict(delay=delay, penalty=pen, intervention=inter, holding=hold, standby=standby,
                    loss=pen + inter + hold + standby,
                    idle=idle, ontime=ontime / max(1, len(self.lots)), completed=done,
                    **{f"n_{k}": v for k, v in self.n_actions.items()}, **{f"c_{k}": v for k, v in self.costs.items()})


# =========================================================================================== policies
class Policy:
    name = "base"

    def setup(self, sim):
        pass

    def decide(self, sim, obs):
        pass


class NoAction(Policy):
    name = "No recovery action"


class StaticBuffer(Policy):
    """Plan-time safety lead time on every order (static resilience buffer)."""
    name = "Static safety lead time"

    def __init__(self, buffer_days=28):
        self.b = buffer_days

    def setup(self, sim):
        for lot in sim.lots:
            lot.plan_order = max(0, lot.plan_order - self.b)


def _lot_quote(fc, sim, lot, opt="keep"):
    return float(np.mean(fc.arrival(lot, opt, sim.p, sim.qualified)))


class ReactiveExpedite(Policy):
    """Incumbent practice: act on a lot only when its quoted arrival misses the need date inside a short window;
    then take the fastest available remedy (air freight, else the alternative supplier with the earliest quote)."""
    name = "Reactive expediting"

    def __init__(self, window=28):
        self.window = window

    def decide(self, sim, obs):
        t = sim.t
        fc = Forecaster("det"); fc.prepare(obs, sim.names)
        for lot in sim.lots:
            if lot.ordered and lot.t_arr <= t:
                continue
            if lot.need - t > self.window:
                continue
            q = t + _lot_quote(fc, sim, lot)
            if q <= lot.need:
                continue
            if lot.cat in PR.AIR_OK and not (lot.ordered and lot.t_ship <= t) and lot.route != "air":
                sim.act_route(lot, "air"); continue
            if lot.ordered and lot.t_qexit <= t:
                continue
            best, bq = None, q - 7
            for r in sim.p.pkg_alts[lot.pkg]:
                qa = t + _lot_quote(fc, sim, lot, "alt:" + r)
                if qa < bq:
                    best, bq = r, qa
            if best:
                sim.act_alt(lot, best)


class FloatRescheduling(Policy):
    """Reactive schedule repair: defer an activity whose quoted material arrival exceeds its start plus total
    float, letting resource-flow successors proceed (no supply-side action)."""
    name = "Float-based rescheduling"

    def __init__(self, margin=0):
        self.margin = margin
        self.tf = None

    def setup(self, sim):
        p = sim.p; inst = p.inst
        succ = {}
        for (i, j) in sim.arcs:
            succ.setdefault(i, []).append(j)
        order = sorted(range(inst.n), key=lambda j: p.start0[j])
        ls = {j: p.C0 - p.dur[j] for j in range(inst.n)}
        for j in reversed(order):
            for s in succ.get(j, ()):
                ls[j] = min(ls[j], ls[s] - p.dur[j])
        self.tf = {j: ls[j] - p.start0[j] for j in range(inst.n)}

    def decide(self, sim, obs):
        t = sim.t
        fc = Forecaster("det"); fc.prepare(obs, sim.names)
        for lot in sim.lots:
            j = lot.act
            if j in sim.start or j in sim.deferred or (lot.ordered and lot.t_arr <= t):
                continue
            q = t + _lot_quote(fc, sim, lot)
            if q > lot.need + self.tf[j] + self.margin and sim.deferrable(j):
                sim.act_defer(j)


class MPC(Policy):
    """Rolling-horizon recovery planning: each week, forecast the arrival of every open lot under each feasible
    action, then choose actions and deferrals with the recovery MILP. The forecaster kind sets the method:
    det (deterministic MPC), iid (sample-average MPC with independent lead-time scenarios), onoff (network
    scenarios without recovery-dependent state) and carp (the cascade-aware recovery planner)."""

    def __init__(self, kind, name, K=12, lam=0.0, lookahead=240, sigma=0.3, defer=False, seed=0, plan_only=False,
                 hazard=0.0, recourse=None, rdelay=7, mis=None, ahead=None, groups=4, lead=0, late_w=1.0, margin=0):
        self.kind, self.name, self.K, self.lam, self.la, self.sigma = kind, name, K, lam, lookahead, sigma
        self.hazard = hazard
        self.rdelay = rdelay
        self.mis = mis
        self.ahead = PR.AHEAD_WINDOW if ahead is None else ahead
        self.recourse = (kind in ("iid", "onoff", "carp") and not plan_only) if recourse is None else recourse
        self.groups = groups if self.recourse else None
        self.lead, self.late_w = lead, late_w
        self.margin = margin          # need-date safety margin of the late-material term (amendment 17)
        self.defer, self.seed, self.plan_only = defer, seed, plan_only
        self.calls = 0

    def setup(self, sim):
        self.rng = np.random.default_rng([self.seed, sim.p.inst.n])
        for lot in sim.lots:          # plan-time safety lead (a tuned setting of every rolling-horizon planner)
            lot.plan_order = max(0, lot.plan_order - self.lead)

    def options(self, sim, lot):
        t = sim.t
        opts = ["keep"]
        if lot.ordered and lot.t_arr <= t:
            return []
        shipped = lot.ordered and lot.t_ship <= t
        in_prod = lot.ordered and lot.order_day <= t and lot.t_qexit <= t
        if self.plan_only:
            opts += ["shift:14", "shift:28", "shift:42"]
        elif not lot.ordered and lot.plan_order - t <= self.ahead:
            opts.append("ahead")
        if not shipped:
            if lot.region in PR.SUEZ_REGIONS and lot.route == "std":
                opts.append("cape")
            if lot.cat in PR.AIR_OK and lot.route != "air" and not self.plan_only:
                opts.append("air")
        if not in_prod:
            for r in sim.p.pkg_alts[lot.pkg]:
                if r != lot.region:
                    opts.append("alt:" + r)
        return opts

    def decide(self, sim, obs):
        if self.plan_only and sim.t > 0:
            return
        t = sim.t
        fc = Forecaster(self.kind, K=self.K, seed=int(self.rng.integers(1 << 30)), sigma=self.sigma,
                        hazard_scale=self.hazard, mis=self.mis, groups=self.groups)
        fc.prepare(obs, sim.names)
        K = fc.K
        lot_opts = []
        threat = False
        for lot in sim.lots:
            if lot.act in sim.start:
                continue
            if lot.need - t > self.la and not lot.ordered:
                continue
            opts = self.options(sim, lot)
            if not opts:
                continue
            rows = []
            for o in opts:
                A = np.minimum(fc.arrival(lot, o, sim.p, sim.qualified), 4000.0)
                Ad = np.minimum(fc.arrival(lot, o, sim.p, sim.qualified, delay=self.rdelay), 4000.0) if self.recourse else None
                esc = obs.esc if lot.cat in METAL else 1.0
                cost = 0.0
                if o.startswith("alt:"):
                    cost = PR.ALT_PREMIUM * lot.value * esc + (0.02 * lot.value if lot.ordered else 0.0)
                elif o == "cape":
                    cost = PR.CAPE_COST * lot.value
                elif o == "air":
                    cost = PR.AIR_COST * lot.value
                rows.append((o, cost, A, Ad) if self.recourse else (o, cost, A))
                if o == "keep" and np.max(A) > lot.need - t - self.margin:
                    threat = True
            lot_opts.append((lot, rows, PR.HOLD_RATE * lot.value, self.late_w * PR.STANDBY * sim.p.pen_day, lot.need - t - self.margin))
        if not lot_opts or (not threat and not self.plan_only):
            return
        qual_cost = {}
        for lot, rows, *_ in lot_opts:
            for o, *_ in rows:
                if o.startswith("alt:") and (lot.pkg, o[4:]) not in sim.qualified:
                    qual_cost[(lot.pkg, o[4:])] = PR.QUALIFY * sum(l.value for l in sim.lots if l.pkg == lot.pkg)
        unstarted = [j for j in range(sim.p.inst.n) if j not in sim.start]
        defer_c, darcs = {}, {}
        if self.defer and not self.plan_only:
            for lot, rows, *_ in lot_opts:
                j = lot.act
                if j in sim.deferred:
                    continue
                ds = sim.deferrable(j)
                if ds:
                    defer_c[j] = PR.RESEQ_COST * sim.p.inst.tardcost; darcs[j] = ds
        state = dict(unstarted=unstarted, fin={i: f for i, f in sim.fin.items()},
                     arcs=list(sim.arcs), defer_arcs=darcs)
        # activities already running have known finish; started-but-unfinished included in fin
        if defer_c:
            choice, deferred = solve_recovery(t, sim.p, state, lot_opts, qual_cost, defer_c, sim.p.pen_day, K,
                                              lam=self.lam, recourse=self.recourse, group=fc.group)
        else:   # fixed arcs: exact reduction to the lot activities (same model, fewer variables)
            choice, deferred = solve_recovery_reduced(t, sim.p, state, lot_opts, qual_cost, sim.p.pen_day, K,
                                                      lam=self.lam, recourse=self.recourse, group=fc.group)
        self.calls += 1
        if choice is None:
            return
        for lot, rows, *_ in lot_opts:
            o = choice[lot.idx]
            if o == "keep":
                continue
            if o == "ahead":
                sim.act_ahead(lot)
            elif o.startswith("shift:"):
                lot.plan_order = max(0, lot.plan_order - int(o[6:]))
            elif o in ("cape", "air"):
                sim.act_route(lot, o)
            elif o.startswith("alt:"):
                sim.act_alt(lot, o[4:])
        for j in deferred:
            sim.act_defer(j)


def make_policy(name, cfg=None):
    cfg = cfg or {}
    if name == "naive":
        return NoAction()
    if name == "practice":
        return ReactiveExpedite(**cfg)
    if name == "buffer":
        return StaticBuffer(**cfg)
    if name == "float":
        return FloatRescheduling(**cfg)
    if name == "saaplan":
        return MPC("onoff", "Two-stage stochastic plan", plan_only=True, **cfg)
    if name == "detmpc":
        return MPC("det", "Deterministic rolling-horizon MPC", K=1, **cfg)
    if name == "saampc":
        return MPC("iid", "Sample-average rolling-horizon MPC", **cfg)
    if name == "ablation":
        return MPC("onoff", "Planner without recovery-dependent state", **cfg)
    if name == "carp":
        return MPC("carp", "Cascade-aware recovery planner", **cfg)
    raise ValueError(name)


METHODS = ["naive", "practice", "buffer", "float", "saaplan", "detmpc", "saampc", "ablation", "carp"]
