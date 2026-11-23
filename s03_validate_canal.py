"""Validation of the chokepoint queue against observed behaviour: the canal node, driven by the observed
closure days of March 2021, predicts the post-reopening surge and backlog clearance, compared with PortWatch daily
transits (total vessels) for 30 March to 20 April 2021. Also checks the Red Sea diversion share used by the model."""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.network import queue_path
from model import params as PR
ROOT = Path(__file__).resolve().parents[1]
pw = pd.read_csv(ROOT / "data/raw/portwatch_chokepoints_daily.csv", encoding="utf-8-sig"); pw["date"] = pd.to_datetime(pw["date"])
s = pw[pw.portid == "chokepoint1"].set_index("date").sort_index()["n_total"]
pre = s["2021-02-20":"2021-03-22"].mean()
peak = s["2021-03-30":"2021-04-20"].max()
kappa = float(peak)                    # nominal throughput capacity (vessels/day)
u = float(pre) / kappa                 # utilisation (normal demand over capacity)
win = s["2021-03-15":"2021-04-20"]
obs = win.values.astype(float)
dates = win.index
closed = (win.index >= "2021-03-23") & (win.index <= "2021-03-29")
a = np.ones(len(obs)); a[closed] = obs[closed] / kappa   # observed residual throughput while blocked
Q, cum = queue_path(a, np.full(len(obs), u), 1.0, 0.0)
served = np.minimum(a, Q[:-1] + u) * kappa
post = dates >= "2021-03-30"
rmse = float(np.sqrt(np.mean((served[post] - obs[post]) ** 2)))
naive_rmse = float(np.sqrt(np.mean((pre - obs[post]) ** 2)))   # benchmark: demand returns at the normal rate, no backlog
clear_model = int(np.argmax((Q[:-1] < 0.5) & post & (np.arange(len(obs)) > np.argmax(closed)))) - int(np.where(post)[0][0])
lost = float(((pre - obs) * closed).sum())
cum_excess = np.cumsum((obs - pre) * post)
clear_obs = int(np.argmax(cum_excess >= lost)) - int(np.where(post)[0][0]) if (cum_excess >= lost).any() else None
w14 = post & (dates <= "2021-04-12")
rmse14 = float(np.sqrt(np.mean((served[w14] - obs[w14]) ** 2)))
rmse14_nb = float(np.sqrt(np.mean((pre - obs[w14]) ** 2)))
mae14 = float(np.mean(np.abs(served[w14] - obs[w14]))); mae14_nb = float(np.mean(np.abs(pre - obs[w14])))
sd_pre = float(s["2021-02-20":"2021-03-22"].std())
elev_obs = int(((obs > pre + sd_pre) & post).sum()); elev_model = int(((served > pre + sd_pre) & post).sum())
excess_obs14 = float((obs[w14] - pre).sum()); excess_model14 = float((served[w14] - pre).sum())
out = dict(rmse_14d=rmse14, rmse_14d_no_backlog=rmse14_nb, mae_14d=mae14, mae_14d_no_backlog=mae14_nb,
           elevated_days_observed=elev_obs, elevated_days_model=elev_model, pre_sd=sd_pre,
           excess_transits_14d_observed=excess_obs14, excess_transits_14d_model=excess_model14,
           kappa=kappa, u=u, pre_mean=float(pre), rmse_post=rmse, rmse_no_backlog=naive_rmse,
           backlog_vessels_at_reopening=float(Q[np.where(post)[0][0]] * kappa), clearance_days_model=clear_model,
           transits_lost=lost, clearance_days_observed=clear_obs,
           served=[round(float(x), 1) for x in served], observed=[float(x) for x in obs],
           dates=[str(d.date()) for d in dates],
           redsea_divert_mean_2024=float(np.mean(PR.REDSEA_DIVERT[1:13])),
           input_sha256=hashlib.sha256((ROOT / "data/raw/portwatch_chokepoints_daily.csv").read_bytes()).hexdigest())
(ROOT / "results" / "validation_canal.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if not isinstance(v, list)})
