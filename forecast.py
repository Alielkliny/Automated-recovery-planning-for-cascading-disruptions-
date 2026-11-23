"""Planner-side arrival forecasts. Every method observes the same weekly state (node backlogs and capacity levels up
to today, current diversion share, price escalation, lot stages). The models differ in how that state is carried
forward:

  carp   recovery-dependent propagation: per scenario, the current outage continues for a remaining time drawn
         conditional on its elapsed duration, capacity ramps back, new outages arrive at stationary hazards on
         shared nodes, and each node's backlog evolves as a fluid queue from today's backlog, so unresolved backlog
         is carried into later shocks;
  onoff  the same conditional, network-correlated scenarios, but each node delay is the frozen current quote plus
         the remaining outage at entry (on/off disruption, no backlog accumulation or drain);
  det    the frozen current quote plus the expected remaining outage (one deterministic scenario);
  iid    the det forecast times independent lognormal lead-time noise per lot and scenario.
"""
from __future__ import annotations
import numpy as np
from . import params as PR
from .network import queue_path, INF, SUP

PER_CLASS = True   # design A4; False reproduces design A3
MOMENTUM = False    # design A5: excess intake projected with the AR(2) dynamics of the GSCPI; False keeps AR(1) decay

U_NOM = {"S": 0.85, "LANE": PR.LANE_U, "CANAL": PR.CANAL_U, "PORT": PR.PORT_U}
AMAX_NOM = {"S": 1.10, "LANE": PR.LANE_AMAX, "CANAL": 1.0, "PORT": 1.0}


def _kind(n):
    return "S" if n.startswith("S|") else n.split("|")[0]


DEEP_LEVEL = 0.72   # development library: production outages run at 0.2 to 0.7 of capacity, curtailments at 0.75 to 0.9


def prior_duration(kind, rng, size, level=None):
    if kind == "S":
        if level is not None and PER_CLASS:
            # per-class library (design A4): deep production outages lasted 28 to 63 days, shallow curtailments 90 to 180
            return rng.uniform(28, 63, size) if level < DEEP_LEVEL else rng.uniform(90, 180, size)
        long = rng.uniform(size=size) < 0.25
        return np.where(long, rng.uniform(90, 180, size), rng.uniform(28, 63, size))
    if kind == "CANAL":
        return rng.uniform(5, 9, size)
    return rng.uniform(180, 900, size)   # congestion of ports and lanes


def cond_remaining(kind, elapsed, rng, size, level=None):
    """Remaining outage given its elapsed duration: rejection from the prior, then a Lindy tail beyond it."""
    out = np.empty(size)
    for i in range(size):
        for _ in range(50):
            d = prior_duration(kind, rng, 1, level)[0]
            if d > elapsed:
                out[i] = d - elapsed; break
        else:
            out[i] = elapsed * (1.0 / max(rng.uniform(), 0.05) - 1.0)
    return out


def expected_remaining(kind, elapsed, level=None):
    rng = np.random.default_rng(0)
    return float(np.mean(cond_remaining(kind, elapsed, rng, 64, level)))


from numba import njit


@njit(cache=True)
def _runlen(a):
    K, H = a.shape
    R = np.zeros((K, H + 1))
    for k in range(K):
        for h in range(H - 1, -1, -1):
            R[k, h] = R[k, h + 1] + 1 if a[k, h] < 0.999 else 0.0
    return R


def _ar2_path(e1, e0, H):
    """Daily excess-intake path from monthly AR(2) iteration (GSCPI coefficients), linear within months."""
    a1, a2 = PR.AR2
    months = int(np.ceil(H / 30)) + 1
    m = [e1]; prev, cur = e0, e1
    for _ in range(months):
        nxt = a1 * cur + a2 * prev; m.append(nxt); prev, cur = cur, nxt
    days = np.arange(H) / 30.0
    return np.interp(days, np.arange(len(m)), np.array(m))


@njit(cache=True)
def _exit_batch(CF, QF, ti, remaining, inf):
    K, H1 = CF.shape
    out = np.empty(K)
    for k in range(K):
        t = ti[k]
        target = CF[k, t] + (remaining if remaining >= 0 else QF[k, t]) - 1e-9
        j = np.searchsorted(CF[k], target)
        out[k] = j if j <= H1 - 1 else inf
    return out


class Obs:
    """What every planner sees at day t."""

    def __init__(self, world, t, names):
        self.t = t
        self.Q = {n: float(world.Q[n][t]) for n in names}
        self.a = {n: float(world.a[n][t]) for n in names}
        self.exit = {n: bool(world.a[n][t] == 0 and np.all(world.a[n][max(0, t - 30):t + 1] == 0) and
                             any(s.kind == "exit" and s.start <= t and n.startswith(f"S|{s.region}|")
                                 for s in world.shocks)) for n in names}
        self.elapsed, self.ramping = {}, {}
        for n in names:
            h = world.a[n][: t + 1]
            if h[-1] >= 1.0:
                self.elapsed[n] = 0; self.ramping[n] = False; continue
            k = np.where(h >= 1.0)[0]
            self.elapsed[n] = t - (k[-1] if len(k) else 0)
            self.ramping[n] = len(h) > 2 and h[-1] > h[-2] + 1e-9
        self.divert = float(world.divert[t])
        dv = world.divert[: t + 1]
        k = np.where(dv <= 0)[0]
        self.div_elapsed = t - (k[-1] if len(k) else 0) if self.divert > 0 else 0
        self.esc = float(world.esc[t])
        # excess order intake inferred from the last two weeks of backlog and output (observable to the buyer)
        self.excess = {}
        self.excess_prev = {}
        w = 14
        for n in names:
            if t >= w and world.Q[n][t - w] > 0.5 and world.Q[n][t] > 0.5:
                out = world.cum[n][t] - world.cum[n][t - w]
                intake = (world.Q[n][t] - world.Q[n][t - w] + out) / w
                self.excess[n] = float(intake - U_NOM[_kind(n)])
                # the same estimate one month earlier, for the momentum projection
                t1 = t - 30
                if t1 >= w and world.Q[n][t1 - w] > 0.5 and world.Q[n][t1] > 0.5:
                    out1 = world.cum[n][t1] - world.cum[n][t1 - w]
                    self.excess_prev[n] = float((world.Q[n][t1] - world.Q[n][t1 - w] + out1) / w - U_NOM[_kind(n)])
        self.cum_hist = {n: world.cum[n] for n in names}   # realised capacity up to t (used only for <= t)


class Forecaster:
    def __init__(self, kind, K=12, H=420, seed=0, sigma=0.3, hazard_scale=1.0, mis=None, groups=None):
        self.kind, self.K, self.H, self.sigma = kind, (1 if kind == "det" else K), H, sigma
        # information groups (design A6): scenarios within a group share the outage and backlog path, which a week
        # reveals, and differ only in draws a week does not reveal (carrier diversion, lead-time noise)
        self.G = self.K if groups is None else max(1, min(groups, self.K))
        self.group = np.arange(self.K) * self.G // self.K
        self.rng = np.random.default_rng(seed)
        self.hz = hazard_scale
        mis = mis or {}
        # planner-internal constants (perturbed only in the misspecification analysis)
        self.u_nom = dict(U_NOM); self.amax_nom = dict(AMAX_NOM)
        self.u_nom["S"] = mis.get("u_sup", U_NOM["S"]); self.amax_nom["S"] = mis.get("amax_sup", AMAX_NOM["S"])
        self.phi = mis.get("phi", PR.PHI_MONTHLY); self.ramp = int(mis.get("ramp", PR.RAMP))
        self.fleet = mis.get("fleet", PR.FLEET_RESPONSE); self.round_trip = mis.get("round_trip", PR.ROUND_TRIP)

    # ----------------------------------------------------------------- scenario capacity paths (carp, onoff)
    def _scenarios(self, obs, names):
        a, rem, lev = self._scenarios_raw(obs, names, self.G)
        if self.G == self.K:
            return a, rem, lev
        idx = self.group
        return ({n: v[idx] for n, v in a.items()}, {n: v[idx] for n, v in rem.items()}, lev)

    def _scenarios_raw(self, obs, names, K):
        H, rng = self.H, self.rng
        a = {n: np.ones((K, H)) for n in names}
        rem = {n: np.zeros(K) for n in names}
        lev = {}
        for n in names:
            kd = _kind(n); a0 = obs.a[n]
            if obs.exit[n]:
                a[n][:] = 0.0; rem[n][:] = INF; lev[n] = 0.0; continue
            if a0 < 1.0:
                if obs.ramping[n] and kd == "S":
                    r = np.zeros(K)
                else:
                    r = cond_remaining(kd, obs.elapsed[n], rng, K, a0)
                rem[n] = r; lev[n] = a0
                for k in range(K):
                    e = int(min(H, round(r[k])))
                    a[n][k, :e] = a0
                    if kd == "S":
                        ramp = self.ramp
                        seg = a0 + (1 - a0) * np.arange(1, ramp + 1) / ramp
                        a[n][k, e:e + ramp] = seg[: max(0, min(ramp, H - e))]
            else:
                lev[n] = 1.0
        # Red Sea lane effect from the observed diversion share
        if obs.divert > 0:
            adiv = 1.0 / (1.0 + obs.divert * 2 * PR.CAPE_EXTRA / self.round_trip)
            rec = np.minimum(1.0, (obs.div_elapsed + np.arange(H)) / self.fleet)
            for n in ("LANE|EA", "LANE|SA"):
                if n in a:
                    a[n] = np.minimum(a[n], adiv + (1 - adiv) * rec)
        # new outages at stationary hazards, shared by all supplier nodes of a region
        regions = sorted({n.split("|")[1] for n in names if n.startswith("S|")}) if self.hz > 0 else []
        for r in regions:
            for k in range(K):
                t = 0
                while True:
                    t += rng.exponential(1095.0 / self.hz)
                    if t >= H:
                        break
                    d = int(prior_duration("S", rng, 1)[0]); lv = rng.uniform(0.2, 0.6); s0 = int(t)
                    for n in names:
                        if n.startswith(f"S|{r}|") and not obs.exit[n]:
                            a[n][k, s0:s0 + d] = np.minimum(a[n][k, s0:s0 + d], lv)
                            ramp = self.ramp; e = min(H, s0 + d)
                            seg = lv + (1 - lv) * np.arange(1, ramp + 1) / ramp
                            m = max(0, min(ramp, H - e))
                            a[n][k, e:e + m] = np.minimum(a[n][k, e:e + m], seg[:m])
        if "CANAL" in names and self.hz > 0:
            for k in range(K):
                t = rng.exponential(1460.0 / self.hz)
                if t < H:
                    s0 = int(t); d = int(rng.uniform(5, 9)); a["CANAL"][k, s0:s0 + d] = 0.0
        return a, rem, lev

    def prepare(self, obs, names):
        """Build the per-node forecast structures for this epoch."""
        self.obs, self.names = obs, names
        self.t = obs.t
        if self.kind == "carp":
            a, _, _ = self._scenarios(obs, names)
            self.QF, self.CF = {}, {}
            phi_d = self.phi ** (1 / 30.0)
            decay = phi_d ** np.arange(self.H)
            for n in names:
                kd = _kind(n)
                e1 = obs.excess.get(n, 0.0)
                if MOMENTUM and n in obs.excess_prev:
                    upath = self.u_nom[kd] + _ar2_path(e1, obs.excess_prev[n], self.H)
                else:
                    upath = self.u_nom[kd] + e1 * decay
                QF = np.empty((self.K, self.H + 1)); CF = np.empty((self.K, self.H + 1))
                for k in range(self.K):
                    QF[k], CF[k] = queue_path(a[n][k], upath, self.amax_nom[kd], obs.Q[n])
                self.QF[n], self.CF[n] = QF, CF
        elif self.kind == "onoff":
            a, rem, lev = self._scenarios(obs, names)
            self.a_sc = a
            # remaining outage length at each future day: count of consecutive days with a < 1 from that day
            self.R = {}
            for n in names:
                self.R[n] = _runlen(a[n])
        else:
            self.quote, self.er = {}, {}
            for n in names:
                kd = _kind(n)
                if obs.exit[n]:
                    self.quote[n] = INF; self.er[n] = 0.0; continue
                er = expected_remaining(kd, obs.elapsed[n], obs.a[n]) if obs.a[n] < 1 else 0.0
                self.er[n] = er
                self.quote[n] = obs.Q[n] + U_NOM[kd] * er * (1 - obs.a[n])

    # ----------------------------------------------------------------- node exit forecast (relative days, K-vector)
    def node_exit(self, n, tau_rel, remaining=None):
        """Forecast day (relative to t) when a lot entering node n at tau_rel (or already queued with `remaining`
        work ahead) leaves the queue."""
        K, H = self.K, self.H
        tau = np.clip(np.asarray(tau_rel, float), 0, H)
        if self.kind == "carp":
            ti = (tau.astype(np.int64) if tau.ndim else np.full(K, int(tau), np.int64))
            return _exit_batch(self.CF[n], self.QF[n], ti, -1.0 if remaining is None else float(remaining), float(INF))
        if self.kind == "onoff":
            if self.obs.exit[n]:
                return np.full(K, INF)
            kd = _kind(n)
            ti = np.clip(np.round(tau).astype(int) if tau.ndim else np.full(K, int(tau)), 0, H)
            Rk = self.R[n][np.arange(K), ti]
            lvl = self.a_sc[n][np.arange(K), np.minimum(ti, H - 1)]
            base = remaining if remaining is not None else self.obs.Q[n]
            return ti + base + U_NOM[kd] * Rk * (1 - lvl)
        q = self.quote[n]
        if q >= INF:
            return np.full(K, INF)
        # current backlog quoted as it stands; the outage adds the intake it will block between entry and its
        # expected end, so a lot entering after the announced end sees no outage component
        kd = _kind(n)
        rem_out = np.maximum(self.er[n] - tau, 0.0)
        if remaining is not None:
            base = remaining + rem_out * (1 - self.obs.a[n])
        else:
            base = self.obs.Q[n] + U_NOM[kd] * rem_out * (1 - self.obs.a[n])
        out = tau + base
        if self.kind == "iid":
            out = tau + base * np.exp(self.sigma * self.rng.normal(size=K) - 0.5 * self.sigma ** 2) \
                if np.ndim(out) else out
        return np.broadcast_to(out, (K,)).astype(float).copy()

    # ----------------------------------------------------------------- lot arrival forecast
    def arrival(self, lot, opt, proj, qualified, delay=0):
        """Forecast arrival (relative days, K-vector) of `lot` under option opt in
        ('keep', 'ahead', 'cape', 'air', 'alt:<region>'). With delay > 0 the option starts `delay` days later; in a
        scenario where the lot ships (route options) or starts production (re-sourcing) before then, the delayed
        option is no longer available and the lot keeps its current plan."""
        if delay > 0 and opt != "keep" and not opt.startswith("shift:"):
            arr_d, _ = self._arrival(lot, opt, proj, qualified, delay)
            arr_keep, prod_keep = self._arrival(lot, "keep", proj, qualified, 0)
            if opt in ("cape", "air"):
                gone = prod_keep <= delay                      # shipped (or about to ship) before the next epoch
            elif opt.startswith("alt:"):
                gone = (prod_keep - PR.PI[lot.cat]) <= delay   # production started before the next epoch
            else:
                gone = np.zeros(self.K, bool)
            return np.where(gone, arr_keep, arr_d)
        return self._arrival(lot, opt, proj, qualified, delay)[0]

    def _arrival(self, lot, opt, proj, qualified, delay=0):
        t, K = self.t, self.K
        region, route = lot.region, lot.route
        if opt.startswith("alt:"):
            region = opt[4:]; route = "std"
            act = 0 if (lot.pkg, region) in qualified else PR.APPROVAL[lot.cat] + PR.ACTIVATION
            stage = ("enter", np.full(K, float(act + delay)))
        elif lot.ordered and lot.t_arr <= t:
            return np.zeros(K), np.full(K, -1.0)
        elif lot.ordered and lot.order_day > t:
            stage = ("enter", np.full(K, float(lot.order_day - t)))
        elif lot.ordered and lot.t_ship <= t:
            stage = ("shipped", None)
        elif lot.ordered and lot.t_prod <= t:
            stage = ("produced", np.full(K, float(lot.t_prod - t)))
        elif lot.ordered and lot.t_qexit <= t:
            stage = ("produced", np.full(K, float(lot.t_qexit + PR.PI[lot.cat] - t)))
        elif lot.ordered:
            sup = SUP(lot.region, lot.cat)
            ch = self.obs.cum_hist[sup]
            ahead = max(0.0, float(proj_q_at(lot, sup, self.obs)) - float(ch[t] - ch[lot.order_day]))
            stage = ("queued", ahead)
        else:
            if opt == "ahead":
                enter = float(min(delay, max(0, lot.plan_order - t)))
            elif opt.startswith("shift:"):
                enter = float(max(0, lot.plan_order - int(opt[6:]) - t))
            else:
                enter = float(max(0, lot.plan_order - t))
            stage = ("enter", np.full(K, enter))
        if opt == "cape":
            route = "cape"
        elif opt == "air":
            route = "air"
        # supplier
        sup = SUP(region, lot.cat)
        if stage[0] == "enter":
            prod = self.node_exit(sup, stage[1]) + PR.PI[lot.cat]
        elif stage[0] == "queued":
            prod = self.node_exit(sup, np.zeros(K), remaining=stage[1]) + PR.PI[lot.cat]
        elif stage[0] == "produced":
            prod = stage[1]
        else:
            prod = None
        if stage[0] == "shipped":
            # at sea: the carrier's ETA from the departure day and the routing known at departure
            ship = float(lot.t_ship - t)
            if route == "air":
                return np.full(K, max(0.0, ship + PR.AIR_TIME)), np.full(K, -1.0)
            v = PR.VOYAGE[region]
            if region in PR.SUEZ_REGIONS and route != "cape" and not lot.diverted:
                canal = self.node_exit("CANAL", np.full(K, max(0.0, ship + 0.6 * v))) + 1
                port = np.maximum(canal + 0.4 * v, 0.0)
            elif region in PR.SUEZ_REGIONS:
                port = np.full(K, ship + v + PR.CAPE_EXTRA)
            else:
                port = np.full(K, ship + v)
            arr = self.node_exit("PORT", np.maximum(port, 0.0)) + PR.PORT_SERVICE
            return np.maximum(arr, 0.0), np.full(K, -1.0)
        prod = np.where(prod >= INF / 2, INF, prod)
        if route == "air":
            return prod + PR.AIR_TIME, prod
        if region in PR.SUEZ_REGIONS:
            ship = self.node_exit("LANE|" + region, prod)
            v = PR.VOYAGE[region]
            if route == "cape":
                port = ship + v + PR.CAPE_EXTRA
            else:
                f = self.obs.divert
                if self.kind == "det":
                    canal = self.node_exit("CANAL", ship + 0.6 * v) + 1
                    port = f * (ship + v + PR.CAPE_EXTRA) + (1 - f) * (canal + 0.4 * v)
                else:
                    div = self.rng.uniform(size=K) < f
                    canal = self.node_exit("CANAL", ship + 0.6 * v) + 1
                    port = np.where(div, ship + v + PR.CAPE_EXTRA, canal + 0.4 * v)
        else:
            port = prod + PR.VOYAGE[region]
        arr = self.node_exit("PORT", port) + PR.PORT_SERVICE
        return np.where((prod >= INF / 2) | (arr >= INF / 2), INF, arr), prod


def proj_q_at(lot, sup, obs):
    """Backlog the lot found at its order day (known to the buyer from the supplier's acknowledgement)."""
    return obs._q_at_order.get(lot.idx, 0.0) if hasattr(obs, "_q_at_order") else 0.0
