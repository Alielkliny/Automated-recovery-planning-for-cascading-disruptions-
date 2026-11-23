"""Conditions that change the preferred action, computed from the model primitives for one imported lot
(structural steel from East Asia, alternative region Europe). An outage (capacity 0.3 for 49 days, 21-day ramp)
starts at day 0 and is detected at day 7. Over a grid of (days from detection to the need date) x (activation delay
of the alternative: approval plus capacity activation), the lot's arrival and loss are computed ex post for keeping
the order and for switching at detection. Regions: unnecessary (kept order on time), beneficial (switching lowers
the loss), too late (the kept order is late and switching does not lower the loss). Panel (b) repeats the grid when
an earlier outage of the same supplier, 110 days before, has left an unresolved backlog."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.network import queue_path
from model import params as PR
ROOT = Path(__file__).resolve().parents[1]
cat = "structural steel"; V = PR.BASE_VALUE * PR.VALUE_MULT[cat]
pen_day = 15.0 / 7.0                       # median J30 tardiness cost per period, per day
late_rate = pen_day * (1 + PR.STANDBY)     # delay penalty plus crew standby per day late (critical activity)
cost_alt = PR.ALT_PREMIUM * V + PR.QUALIFY * V + 0.02 * V
H = 900; t_det = 7


OFF = 300   # time axis starts 300 days before the outage


def supplier(earlier):
    a = np.ones(H)
    shocks = [(OFF, 49)] + ([(OFF - 110, 49)] if earlier else [])
    for s0, d in shocks:
        a[s0:s0 + d] = np.minimum(a[s0:s0 + d], 0.3)
        a[s0 + d:s0 + d + 21] = np.minimum(a[s0 + d:s0 + d + 21], 0.3 + 0.7 * np.arange(1, 22) / 21)
    return queue_path(a, np.full(H, PR.SUP_U), PR.SUP_AMAX, 0.0)


def first_exit(Q, cum, tau):
    target = cum[tau] + Q[tau]; return int(np.searchsorted(cum, target - 1e-9))


lead_keep = PR.VOYAGE["EA"] + 1 + PR.PORT_SERVICE
lead_alt = PR.VOYAGE["EU"] + PR.PORT_SERVICE
X = np.arange(0, 301, 5)        # days from detection to need date
Y = np.arange(7, 92, 3)         # activation delay of the alternative
out = {}
for tag, earlier in [("single", False), ("unresolved", True)]:
    Q, cum = supplier(earlier)
    out[tag + "_backlog_at_onset"] = float(Q[OFF])
    region = np.zeros((len(Y), len(X)), int); saving = np.zeros((len(Y), len(X)))
    for ix, x in enumerate(X):
        need = t_det + x
        plan = need - PR.PLAN_FLOAT - (PR.PI[cat] + lead_keep)
        qe = first_exit(Q, cum, OFF + plan) - OFF
        arr_keep = qe + PR.PI[cat] + lead_keep
        late_keep = max(0, arr_keep - need)
        for iy, y in enumerate(Y):
            arr_alt = t_det + y + PR.PI[cat] + lead_alt
            late_alt = max(0, arr_alt - need)
            loss_keep = late_rate * late_keep; loss_alt = cost_alt + late_rate * late_alt
            saving[iy, ix] = loss_keep - loss_alt
            if late_keep == 0:
                region[iy, ix] = 0          # unnecessary
            elif loss_alt < loss_keep:
                region[iy, ix] = 1          # beneficial
            else:
                region[iy, ix] = 2          # too late or not worth it
    out[tag] = dict(region=region.tolist(), saving=saving.tolist())
out.update(X=X.tolist(), Y=Y.tolist(), pen_day=pen_day, cost_alt=cost_alt, category=cat, outage="0.3 for 49 days, 21-day ramp",
           earlier_outage="0.3 for 49 days starting 110 days before", detection_day=t_det)
for tag in ("single", "unresolved"):
    r = np.array(out[tag]["region"])
    out[tag + "_shares_pct"] = {"unnecessary": round(100 * float((r == 0).mean()), 1), "beneficial": round(100 * float((r == 1).mean()), 1),
                                "too_late": round(100 * float((r == 2).mean()), 1)}
    xs = np.array(out["X"]); ben = (r == 1).any(axis=0)
    out[tag + "_beneficial_need_days"] = [int(xs[ben].min()), int(xs[ben].max())] if ben.any() else None
(ROOT / "results" / "conditions.json").write_text(json.dumps(out))
for tag in ("single", "unresolved"):
    r = np.array(out[tag]["region"]); print(tag, "share unnecessary/beneficial/too late:", [round(float((r == k).mean()), 3) for k in range(3)])
