"""Persistence check: do disruption effects persist beyond their trigger long enough that the state
carried at a weekly planning epoch informs the next weeks? Measured on real public series only."""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

g = pd.read_excel(RAW / "gscpi_data.xlsx", sheet_name="GSCPI Monthly Data", usecols=[0, 1])
g.columns = ["date", "gscpi"]
g = g.dropna()
g["date"] = pd.to_datetime(g["date"], format="%d-%b-%Y")
g = g.set_index("date")["gscpi"].astype(float)

# AR(1) persistence of the monthly GSCPI, 1998 onward, with HAC standard errors
y = g.values
X = sm.add_constant(y[:-1]); res = sm.OLS(y[1:], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
phi = float(res.params[1]); lo, hi = [float(v) for v in res.conf_int(alpha=0.05)[1]]
half_life_months = float(np.log(0.5) / np.log(phi))

# state carried into the onset month of the 2022 episode
gscpi_feb2022 = float(g.loc["2022-02"].iloc[0])
gscpi_peak = float(g.max()); peak_date = str(g.idxmax().date())

# maritime: Suez cargo transits, Red Sea episode persistence and Ever Given backlog clearance
pw = pd.read_csv(RAW / "portwatch_chokepoints_daily.csv", encoding="utf-8-sig"); pw["date"] = pd.to_datetime(pw["date"])
suez = pw[pw.portid == "chokepoint1"].set_index("date").sort_index()
cape = pw[pw.portid == "chokepoint7"].set_index("date").sort_index()
pre = suez.loc["2023-01-01":"2023-11-30", "n_cargo"].mean()
post = suez.loc["2024-03-01":"2024-12-31", "n_cargo"].mean()
cape_pre = cape.loc["2023-01-01":"2023-11-30", "n_cargo"].mean(); cape_post = cape.loc["2024-03-01":"2024-12-31", "n_cargo"].mean()
m = suez["n_cargo"].resample("MS").mean()
below = (m.loc["2023-12-01":] < 0.75 * pre)
months_below_75 = int(below.sum())
# Ever Given: closure 2021-03-23..2021-03-29; transits above the pre-closure mean afterwards
evg_pre = suez.loc["2021-02-20":"2021-03-22", "n_total"].mean()
after = suez.loc["2021-03-30":"2021-05-15", "n_total"]
lost = float((evg_pre - suez.loc["2021-03-23":"2021-03-29", "n_total"]).sum())
cum_excess = (after - evg_pre).cumsum()
days_to_clear = int(np.argmax(cum_excess.values >= lost) + 1) if (cum_excess.values >= lost).any() else None

out = {
  "gscpi": {"ar1_phi": phi, "ar1_ci95": [lo, hi], "half_life_months": half_life_months,
            "n_months": int(len(y)), "value_2022_02": gscpi_feb2022, "peak": gscpi_peak, "peak_date": peak_date},
  "suez_red_sea": {"cargo_transits_per_day_2023_jan_nov": float(pre), "cargo_transits_per_day_2024_mar_dec": float(post),
                   "ratio_post_pre": float(post / pre), "months_below_75pct_since_2023_12": months_below_75,
                   "cape_cargo_per_day_2023_jan_nov": float(cape_pre), "cape_cargo_per_day_2024_mar_dec": float(cape_post)},
  "ever_given": {"pre_closure_total_transits_per_day": float(evg_pre), "transits_lost_during_closure": lost,
                 "days_after_reopening_to_clear_backlog": days_to_clear},
  "premise_rule": "ar1_ci95 lower bound above 0.5 means a half-life above one month, longer than the four-week planning look-ahead, so the disruption state at an epoch carries information about the following weeks",
  "inputs": {"gscpi_data.xlsx": sha(RAW / "gscpi_data.xlsx"), "portwatch_chokepoints_daily.csv": sha(RAW / "portwatch_chokepoints_daily.csv")},
}
out["premise_ar1_ci_low"] = lo
(ROOT / "probe" / "premise.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
