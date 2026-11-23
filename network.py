"""Multilayer supply network: fluid-queue nodes (suppliers, carrier lanes, the Suez canal, the destination port),
disruption episodes, true capacity trajectories and lot passage times.

Each node n has a capacity multiplier a_n(t) <= 1 set by shocks, a background utilisation u_n and a backlog
Q_n(t) in days of nominal capacity: Q(t+1) = max(0, Q(t) + u - a(t) * (amax if Q(t) > 0 else 1)).
A lot entering n at day tau leaves the queue when the cumulative effective capacity after tau covers Q(tau)
(first in, first out), then spends the node's service time. Background flows dominate the node, so a project's
own lots do not change Q (a small-customer assumption)."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
import numpy as np
from numba import njit

from . import params as PR
from .psplib import Instance, serial_sgs, resource_flow_arcs

INF = 10 ** 7


@njit(cache=True)
def queue_path(a, u, amax, q0):
    """u is an array of daily utilisation (order intake over nominal capacity)."""
    T = a.shape[0]
    Q = np.empty(T + 1); cum = np.empty(T + 1)
    Q[0] = q0; cum[0] = 0.0
    for t in range(T):
        eff = a[t] * (amax if Q[t] > 1e-9 else 1.0)
        Q[t + 1] = max(0.0, Q[t] + u[t] - eff)
        cum[t + 1] = cum[t] + eff
    return Q, cum


def exit_time(Q, cum, tau):
    """First day t >= tau with cum[t] - cum[tau] >= Q[tau]; INF when capacity never covers it."""
    tau = int(min(max(tau, 0), len(Q) - 1))
    target = cum[tau] + Q[tau]
    k = int(np.searchsorted(cum, target - 1e-9, side="left"))
    if k >= len(cum):
        return INF
    return max(k, tau)


# ------------------------------------------------------------------------------------------ project setup
@dataclass
class Lot:
    idx: int
    act: int
    pkg: int
    cat: str
    value: float
    need: int            # absolute day of the baseline start
    plan_order: int      # absolute day of the planned order
    region: str          # current supplier region
    route: str = "std"   # std, cape, air
    order_day: int = -1  # day the current order enters the supplier queue (after approval if re-sourced)
    ordered: bool = False
    t_qexit: int = INF
    t_prod: int = INF
    t_ship: int = INF
    t_arr: int = INF
    diverted: bool = False
    switched: int = 0


@dataclass
class Project:
    inst: Instance
    start0: np.ndarray        # baseline start, days from construction start
    dur: np.ndarray           # days
    flow: set
    C0: int                   # baseline completion, days
    T0: int                   # absolute day of construction start
    lots: list
    pkg_cat: list
    pkg_alts: list            # alternative regions per package
    act_lot: dict             # activity -> lot index
    pen_day: float
    horizon: int


def _rng_for(name: str, salt: str = "") -> np.random.Generator:
    h = int(hashlib.sha256((name + salt).encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def nominal_lead(cat: str, region: str, route: str = "std") -> int:
    v = PR.VOYAGE[region]
    if route == "air":
        return PR.PI[cat] + PR.AIR_TIME
    if route == "cape":
        v += PR.CAPE_EXTRA
    canal = 1 if (region in PR.SUEZ_REGIONS and route == "std") else 0
    return PR.PI[cat] + v + canal + PR.PORT_SERVICE


def build_project(inst: Instance, mat_fraction: float = None) -> Project:
    mat_fraction = PR.MAT_FRACTION if mat_fraction is None else mat_fraction
    rng = _rng_for(inst.name, "structure")
    st = serial_sgs(inst)
    flow = resource_flow_arcs(inst, st)
    D = inst.dur * PR.DAYS_PER_PERIOD
    start0 = st * PR.DAYS_PER_PERIOD
    real = [j for j in range(inst.n) if inst.dur[j] > 0]
    m = max(1, int(round(mat_fraction * len(real))))
    chosen = sorted(rng.choice(real, size=m, replace=False).tolist(), key=lambda j: (start0[j], j))
    cats = rng.choice(len(PR.CATS), size=m, p=PR.CAT_WEIGHT)
    T0 = PR.PROCUREMENT_LEAD
    lots, pkg_cat, pkg_alts, act_lot = [], [], [], {}
    open_pkg = {}
    for j, ci in zip(chosen, cats):
        c = PR.CATS[ci]
        if c not in open_pkg or len(open_pkg[c][1]) >= PR.LOTS_PER_PACKAGE:
            sh = PR.SHARES[c]
            reg = PR.REGIONS[int(rng.choice(len(PR.REGIONS), p=sh / sh.sum()))]
            order = [PR.REGIONS[k] for k in np.argsort(-sh) if PR.REGIONS[k] != reg and sh[k] > 0.02
                     and PR.REGIONS[k] != "BS"]
            alts = order[:2] if order else []
            if "EU" not in alts and reg != "EU" and len(alts) < 2:
                alts.append("EU")
            pid = len(pkg_cat); pkg_cat.append(c); pkg_alts.append(alts)
            open_pkg[c] = (pid, [], reg)
        pid, members, reg = open_pkg[c]
        members.append(j)
        need = T0 + int(start0[j])
        lead = nominal_lead(c, reg)
        lot = Lot(len(lots), j, pid, c, PR.BASE_VALUE * PR.VALUE_MULT[c], need,
                  max(0, need - PR.PLAN_FLOAT - lead), reg)
        act_lot[j] = lot.idx
        lots.append(lot)
    C0 = int((start0 + D).max())
    horizon = T0 + 5 * C0 + 900
    return Project(inst, start0, D, flow, C0, T0, lots, pkg_cat, pkg_alts, act_lot,
                   inst.tardcost / PR.DAYS_PER_PERIOD, horizon)


# ------------------------------------------------------------------------------------------ episodes
SUP = lambda r, c: f"S|{r}|{c}"


def node_names(proj: Project) -> list:
    names = {"LANE|EA", "LANE|SA", "CANAL", "PORT"}
    for lot in proj.lots:
        names.add(SUP(lot.region, lot.cat))
        for r in proj.pkg_alts[lot.pkg]:
            names.add(SUP(r, lot.cat))
    return sorted(names)


@dataclass
class Shock:
    kind: str        # supplier, canal, exit, congestion, redsea
    region: str      # supplier region or '' ; for congestion/redsea the affected group
    cats: tuple      # categories affected ('' = all)
    start: int
    dur: int         # INF for permanent
    level: float
    ramp: int


@dataclass
class World:
    """True realisation of one replication: node trajectories and per-lot random numbers (common across methods)."""
    names: list
    a: dict
    u: dict
    upath: dict
    amax: dict
    Q: dict
    cum: dict
    divert: np.ndarray       # diversion share of Suez sailings per day
    esc: np.ndarray          # metal price escalation per day
    shocks: list
    lot_u: np.ndarray        # per lot, per option uniform for diversion
    lot_z: np.ndarray        # per lot, per option normal for voyage noise
    episode: str


EPISODES_DEV = ["covid", "evergiven", "ruua", "covid+evergiven", "covid+ruua"]
EPISODES_TEST = ["redsea", "redsea+production"]


def _supplier_shock(rng, region, start, dmin, dmax, lmin, lmax, cats=("",)):
    return Shock("supplier", region, tuple(cats), int(start), int(rng.integers(dmin, dmax + 1)),
                 float(rng.uniform(lmin, lmax)), int(rng.integers(14, 36)))


def sample_episode(proj: Project, episode: str, seed: int, onset: int = None, spacing: int = None,
                   components: tuple = None) -> list:
    """Shock list for one replication. components restricts a compound episode to some of its parts (matched
    experiments): ('A',), ('B',) or ('A','B'); onset and spacing fix the timing."""
    rng = np.random.default_rng(seed)
    T0, C0 = proj.T0, proj.C0
    o = int(rng.integers(T0 - 60, T0 + int(0.5 * C0))) if onset is None else onset
    dlt = int(rng.integers(60, 301)) if spacing is None else spacing
    parts = episode.split("+")
    comps = ("A", "B")[:len(parts)] if components is None else components
    shocks = []
    for k, part in enumerate(parts):
        tag = "A" if k == 0 else "B"
        t = o if k == 0 else o + dlt
        prng = np.random.default_rng([seed, k, 7])
        if tag not in comps:
            continue
        if part == "covid":
            shocks.append(_supplier_shock(prng, "EA", t, 35, 63, 0.2, 0.4))
            t2 = t + int(prng.integers(28, 50))
            for r in ("EU", "TR"):
                shocks.append(_supplier_shock(prng, r, t2, 28, 49, 0.5, 0.7))
            shocks.append(Shock("congestion", "", ("",), t, 30 * len(PR.GSCPI_PATH), PR.CONGESTION_DEPTH, 0))
        elif part == "evergiven":
            shocks.append(Shock("canal", "", ("",), t, int(prng.integers(6, 9)), 0.0, 0))
        elif part == "ruua":
            shocks.append(Shock("exit", "BS", ("",), t, INF, 0.0, 0))
            for r in ("EU", "TR"):
                shocks.append(_supplier_shock(prng, r, t, 90, 180, 0.75, 0.9, cats=tuple(PR.ENERGY_INTENSIVE)))
            shocks.append(Shock("escalation", "", ("",), t, INF, 0.0, 0))
        elif part == "redsea":
            shocks.append(Shock("redsea", "", ("",), t, INF, 0.0, 0))
        elif part == "production":
            shocks.append(_supplier_shock(prng, "EA", t, 28, 56, 0.3, 0.5))
        else:
            raise ValueError(part)
    return shocks


def apply_shocks(names, horizon, shocks, rng_params):
    a = {n: np.ones(horizon) for n in names}
    divert = np.zeros(horizon); esc = np.ones(horizon)
    for s in shocks:
        if s.kind in ("supplier", "exit"):
            for n in names:
                if not n.startswith("S|"):
                    continue
                _, r, c = n.split("|")
                if r != s.region or (s.cats != ("",) and c not in s.cats):
                    continue
                _shape(a[n], s.start, s.dur, s.level, s.ramp)
        elif s.kind == "canal":
            _shape(a["CANAL"], s.start, s.dur, s.level, 0)
        elif s.kind == "congestion":
            g = np.repeat(PR.GSCPI_PATH, 30)
            for n in ("PORT", "LANE|EA", "LANE|SA"):
                end = min(horizon, s.start + len(g))
                if s.start < horizon:
                    seg = 1 - s.level * g[: end - max(s.start, 0)] if s.start >= 0 else 1 - s.level * g[-s.start: end - s.start]
                    a[n][max(s.start, 0):end] = np.minimum(a[n][max(s.start, 0):end], seg)
        elif s.kind == "redsea":
            f = np.repeat(PR.REDSEA_DIVERT, 30)
            seg = np.concatenate([f, np.full(max(0, horizon), f[-1])])[: max(0, horizon - s.start)]
            divert[s.start:] = np.maximum(divert[s.start:], seg)
            for n in ("LANE|EA", "LANE|SA"):
                adiv = 1.0 / (1.0 + seg * 2 * PR.CAPE_EXTRA / PR.ROUND_TRIP)
                rec = np.minimum(1.0, np.arange(len(seg)) / PR.FLEET_RESPONSE)
                a[n][s.start:] = np.minimum(a[n][s.start:], adiv + (1 - adiv) * rec)
        elif s.kind == "escalation":
            e = np.repeat(PR.METAL_ESC, 30)
            seg = np.concatenate([e, np.full(horizon, e[-1])])[: max(0, horizon - s.start)]
            esc[s.start:] = np.maximum(esc[s.start:], seg)
    return a, divert, esc


def _shape(arr, start, dur, level, ramp):
    H = len(arr)
    s0 = max(0, start); e = H if dur >= INF else min(H, start + dur)
    if e > s0:
        arr[s0:e] = np.minimum(arr[s0:e], level)
    if dur < INF and ramp > 0 and e < H:
        r_end = min(H, e + ramp)
        seg = level + (1 - level) * (np.arange(1, r_end - e + 1) / ramp)
        arr[e:r_end] = np.minimum(arr[e:r_end], seg)


def node_params(name, rng):
    if name.startswith("S|"):
        return float(rng.uniform(PR.SUP_U_LO, PR.SUP_U_HI)), float(rng.uniform(1.05, 1.15))
    if name.startswith("LANE"):
        return PR.LANE_U, PR.LANE_AMAX
    if name == "CANAL":
        return PR.CANAL_U, 1.0
    return PR.PORT_U, 1.0


def make_world(proj: Project, episode: str, seed: int, **kw) -> World:
    names = node_names(proj)
    shocks = sample_episode(proj, episode, seed, **kw)
    prng = np.random.default_rng([seed, 99])
    a, divert, esc = apply_shocks(names, proj.horizon, shocks, prng)
    u, amax, Q, cum, up = {}, {}, {}, {}, {}
    H = proj.horizon
    for n in names:
        u[n], amax[n] = node_params(n, np.random.default_rng([seed, int(hashlib.sha256(n.encode()).hexdigest()[:8], 16)]))
        up[n] = demand_path(n, u[n], shocks, H)
        Q[n], cum[n] = queue_path(a[n], up[n], amax[n], 0.0)
    lrng = np.random.default_rng([seed, 5])
    L = len(proj.lots)
    return World(names, a, u, up, amax, Q, cum, divert, esc, shocks, lrng.uniform(size=(L, 8)),
                 lrng.normal(size=(L, 8)), episode)


def demand_path(n, u0, shocks, H):
    """Order intake over nominal capacity. The COVID episode raises intake along the GSCPI shape; the exit of
    Black Sea suppliers shifts demand to other suppliers of energy-intensive categories, decaying with the
    measured GSCPI persistence."""
    up = np.full(H, u0)
    for s in shocks:
        if s.kind == "congestion":
            g = np.repeat(PR.GSCPI_PATH, 30)
            beta = PR.DEMAND_SURGE_SUP if n.startswith("S|") else PR.DEMAND_SURGE_LOG
            e = min(H, s.start + len(g))
            if e > s.start:
                up[s.start:e] *= (1 + beta * g[: e - s.start])
        elif s.kind == "exit" and n.startswith("S|"):
            _, r, c = n.split("|")
            if r != s.region and c in PR.ENERGY_INTENSIVE:
                phi = PR.PHI_MONTHLY ** (1 / 30.0)
                h = np.arange(H - s.start)
                up[s.start:] += PR.DEMAND_SHIFT_RUUA * phi ** h
    return up


# ------------------------------------------------------------------------------------------ true passage
def passage(world: World, lot: Lot, order_day: int, region: str, route: str, opt_slot: int, from_stage=None):
    """True stage times of a lot ordered at order_day from region via route. Returns (qexit, prod, ship, arr)."""
    sup = SUP(region, lot.cat)
    if from_stage is None:
        q = exit_time(world.Q[sup], world.cum[sup], order_day)
        if q >= INF:
            return INF, INF, INF, INF, False
        prod = q + PR.PI[lot.cat]
    else:
        q, prod = from_stage
    z = world.lot_z[lot.idx, opt_slot]; uu = world.lot_u[lot.idx, opt_slot]
    noise = float(np.exp(0.08 * z))
    if route == "air":
        return q, prod, prod, int(prod + round(PR.AIR_TIME * noise)), False
    diverted = False
    if region in PR.SUEZ_REGIONS:
        lane = "LANE|" + region
        ship = exit_time(world.Q[lane], world.cum[lane], prod)
        if ship >= INF:
            return q, prod, INF, INF, False
        v = PR.VOYAGE[region] * noise
        if route == "cape" or uu < world.divert[min(ship, len(world.divert) - 1)]:
            diverted = route != "cape"
            t_port = ship + v + PR.CAPE_EXTRA
        else:
            t_canal = int(ship + 0.6 * v)
            t_c = exit_time(world.Q["CANAL"], world.cum["CANAL"], t_canal) + 1
            t_port = t_c + 0.4 * v
    else:
        ship = prod
        t_port = ship + PR.VOYAGE[region] * noise
    t_port = int(round(t_port))
    arr = exit_time(world.Q["PORT"], world.cum["PORT"], t_port) + PR.PORT_SERVICE
    return q, prod, ship, int(arr), diverted
