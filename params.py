"""Model parameters with provenance: REAL (measured from a downloaded public file), CITED (from a
verified source), ASSUMPTION (author setting with a sensitivity test) or DERIVED (computed from REAL inputs).
Calibration series are read from the raw public files at import and hashed in the run manifests."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

PROVENANCE = []  # rows: name, value, unit, tag, source, sensitivity


def P(name, value, unit, tag, source, sens=""):
    PROVENANCE.append(dict(name=name, value=value if np.isscalar(value) else str(value), unit=unit, tag=tag,
                           source=source, sensitivity=sens))
    return value


# ---------------------------------------------------------------- calendar and project layer
DAYS_PER_PERIOD = P("days_per_period", 7, "days", "ASSUMPTION", "one PSPLIB period read as one working week", "5 days")
PROCUREMENT_LEAD = P("procurement_lead", 210, "days", "ASSUMPTION", "procurement opens 210 days before construction start, enough for the longest nominal lead time")
MAT_FRACTION = P("material_fraction", 0.40, "share of activities", "ASSUMPTION",
                 "share of non-dummy activities that need an imported package lot", "0.25, 0.55")
LOTS_PER_PACKAGE = P("lots_per_package", 3, "lots", "ASSUMPTION", "activities served by one package")
PLAN_FLOAT = P("plan_float", 7, "days", "ASSUMPTION", "planned arrival ahead of the baseline start")

# ---------------------------------------------------------------- categories (Comtrade HS codes)
CATS = ["structural steel", "reinforcing bars", "cables", "transformers", "air conditioning units",
        "aluminium structures", "valves", "lifting equipment"]
PI = dict(zip(CATS, [35, 21, 35, 120, 56, 49, 42, 90]))  # manufacturing lead time, days
P("manufacturing_lead_time", str(PI), "days", "ASSUMPTION", "nominal manufacturing time per category", "x0.75, x1.25")
VALUE_MULT = dict(zip(CATS, [1.5, 0.8, 1.0, 3.0, 1.5, 1.2, 0.5, 2.0]))
P("lot_value_multiplier", str(VALUE_MULT), "x base", "ASSUMPTION", "relative lot value per category")
AIR_OK = {"cables", "air conditioning units", "valves"}
P("air_eligible", str(sorted(AIR_OK)), "-", "ASSUMPTION", "categories light enough for air freight")
APPROVAL = dict(zip(CATS, [21, 14, 21, 42, 28, 28, 21, 42]))
P("approval_time", str(APPROVAL), "days", "ASSUMPTION", "substitution submittal review time", "x0.5, x1.5")
ACTIVATION = P("capacity_activation", 14, "days", "ASSUMPTION", "alternative supplier capacity activation")
ENERGY_INTENSIVE = {"structural steel", "reinforcing bars", "aluminium structures", "cables"}

# ---------------------------------------------------------------- regions and logistics layer
REGIONS = ["EA", "SA", "EU", "TR", "BS", "OT"]
VOYAGE = dict(EA=24, SA=14, EU=6, TR=4, BS=5, OT=22)
P("voyage_time", str(VOYAGE), "days", "ASSUMPTION", "port-to-port time to an Eastern Mediterranean port", "x0.8, x1.2")
SUEZ_REGIONS = {"EA", "SA"}
CAPE_EXTRA = P("cape_extra", 14, "days", "ASSUMPTION", "added sailing time when routed via the Cape of Good Hope",
               "10, 18")
AIR_TIME = P("air_time", 6, "days", "ASSUMPTION", "air freight door to site incl. clearance")
PORT_SERVICE = P("port_service", 5, "days", "ASSUMPTION", "destination port clearance")
PORT_U = P("port_utilisation", 0.75, "-", "ASSUMPTION", "destination port utilisation")
SUP_U = P("supplier_utilisation", 0.85, "-", "ASSUMPTION", "supplier background utilisation", "0.80 to 0.90 per run")
SUP_U_LO, SUP_U_HI = 0.80, 0.90
SUP_AMAX = P("supplier_surge", 1.10, "x nominal", "ASSUMPTION", "supplier surge capacity while backlog exists",
             "1.05 to 1.15 per run")
LANE_U = P("lane_utilisation", 0.88, "-", "ASSUMPTION", "carrier lane utilisation")
LANE_AMAX = P("lane_surge", 1.05, "x nominal", "ASSUMPTION", "carrier surge capacity while backlog exists")
RAMP = P("capacity_ramp", 21, "days", "ASSUMPTION", "linear capacity recovery after a shock ends", "14 to 35 per run")
ROUND_TRIP = P("lane_round_trip", 77, "days", "ASSUMPTION", "Asia to Mediterranean service round trip")
FLEET_RESPONSE = P("fleet_response", 240, "days", "ASSUMPTION", "days for carriers to restore lane capacity")

# ---------------------------------------------------------------- costs (cost units, CU)
BASE_VALUE = P("base_lot_value", 100.0, "CU", "ASSUMPTION", "base lot value; PSPLIB tardcost gives the delay penalty",
               "50, 200")
ALT_PREMIUM = P("alt_premium", 0.15, "x lot value", "ASSUMPTION", "price premium of an alternative supplier")
QUALIFY = P("qualification_cost", 0.05, "x lot value", "ASSUMPTION", "once per package and alternative region")
CAPE_COST = P("cape_booking_cost", 0.04, "x lot value", "ASSUMPTION", "premium for a booked Cape routing")
AIR_COST = P("air_cost", 0.30, "x lot value", "ASSUMPTION", "air freight premium")
HOLD_RATE = P("holding_rate", 0.18 / 365, "x lot value per day", "ASSUMPTION", "inventory carrying rate of 18% a year")
STANDBY = P("standby_rate", 0.5, "x daily delay penalty per idle activity-day", "ASSUMPTION",
            "crew standby while an activity waits for its material", "0.25, 1.0")
RESEQ_COST = P("resequencing_cost", 0.5, "x weekly tardcost", "ASSUMPTION", "crew remobilisation per deferral")
AHEAD_WINDOW = P("order_ahead_window", 35, "days", "ASSUMPTION", "an order can be advanced by at most 35 days")
RESEQ_DELAY = P("resequencing_delay", 7, "days", "ASSUMPTION", "replanning time before a deferral takes effect")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


INPUT_HASHES = {}


def _load_series():
    g = pd.read_excel(RAW / "gscpi_data.xlsx", sheet_name="GSCPI Monthly Data", usecols=[0, 1])
    g.columns = ["date", "v"]; g = g.dropna(); g["date"] = pd.to_datetime(g["date"], format="%d-%b-%Y")
    g = g.set_index("date")["v"].astype(float)
    gpath = g.loc["2020-03":"2022-12"].clip(lower=0).values
    gpath = gpath / gpath.max()
    pw = pd.read_csv(RAW / "portwatch_chokepoints_daily.csv", encoding="utf-8-sig"); pw["date"] = pd.to_datetime(pw["date"])
    s = pw[pw.portid == "chokepoint1"].set_index("date").sort_index()
    base = s.loc["2023-01-01":"2023-11-30", "n_container"].mean()
    m = s["n_container"].resample("MS").mean().loc["2023-12":"2025-12"]
    divert = np.clip(1 - m.values / base, 0, 1)
    x = pd.ExcelFile(RAW / "CMO-Historical-Data-Monthly.xlsx")
    ind = pd.read_excel(x, "Monthly Indices", header=None).iloc[9:]
    ind.index = ind[0]
    bm = pd.to_numeric(ind[15], errors="coerce")
    esc = (bm.loc["2022M02":"2023M12"] / bm.loc["2021M01":"2021M12"].mean()).values.astype(float)
    evg = s.loc["2021-03-15":"2021-04-20", "n_total"]
    for f in ["gscpi_data.xlsx", "portwatch_chokepoints_daily.csv", "CMO-Historical-Data-Monthly.xlsx",
              "comtrade_egypt_imports.csv", "comtrade_partnerAreas.json"]:
        INPUT_HASHES[f] = _sha(RAW / f)
    return gpath, divert, np.maximum(esc, 1.0), evg


GSCPI_PATH, REDSEA_DIVERT, METAL_ESC, EVERGIVEN_OBS = _load_series()
P("gscpi_path", "GSCPI 2020-03 to 2022-12, floored at 0, scaled to peak 1", "-", "REAL", "NY Fed GSCPI monthly file")
P("redsea_divert", "1 - Suez container transits / Jan-Nov 2023 mean, monthly from 2023-12", "-", "REAL",
  "IMF PortWatch daily chokepoint file")
P("metal_escalation", "Base metals index / 2021 mean, monthly from 2022-02", "-", "REAL", "World Bank Pink Sheet")
CANAL_U = P("canal_utilisation", 57.0 / 88.0, "-", "DERIVED",
            "mean daily Suez transits 20 Feb to 22 Mar 2021 (57) over the post-closure peak (88), PortWatch")
DEMAND_SURGE_SUP = P("demand_surge_supplier", 0.30, "x utilisation at GSCPI peak", "ASSUMPTION",
                     "supplier order intake above normal at the GSCPI peak, following the GSCPI shape", "0.2, 0.4")
DEMAND_SURGE_LOG = P("demand_surge_logistics", 0.15, "x utilisation at GSCPI peak", "ASSUMPTION",
                     "carrier lane and port demand above normal at the GSCPI peak", "0.1, 0.2")
DEMAND_SHIFT_RUUA = P("demand_shift_ruua", 0.15, "utilisation added", "ASSUMPTION",
                      "buyers leaving Black Sea suppliers raise demand at other energy-intensive suppliers")
PHI_MONTHLY = P("pressure_persistence", 0.933, "AR(1) per month", "REAL",
                "AR(1) coefficient of the monthly GSCPI, 1998 to 2026 (probe/premise.json)")
_prem = __import__("json").load(open(ROOT / "probe" / "premise.json"))
AR2 = P("pressure_ar2", (_prem["gscpi_ar2"]["a1"], _prem["gscpi_ar2"]["a2"]), "AR(2) per month", "REAL",
        "second-order autoregression of the monthly GSCPI (probe/premise.json), lower AIC than AR(1)")
CONGESTION_DEPTH = P("covid_congestion_depth", 0.15, "capacity loss at GSCPI peak", "ASSUMPTION",
                     "port and lane capacity loss at the GSCPI peak", "0.10, 0.25")


def sourcing_shares():
    """Mean import-value share by supplier region per category, Comtrade 2019, 2021, 2023 (REAL)."""
    import json
    df = pd.read_csv(RAW / "comtrade_egypt_imports.csv")
    ref = {r["PartnerCode"]: r.get("PartnerCodeIsoAlpha3", "") for r in
           json.load(open(RAW / "comtrade_partnerAreas.json", encoding="utf-8-sig"))["results"]}
    df["iso3"] = df.partnerCode.map(ref)
    df = df[df.partnerCode != 0]
    EA = {"CHN", "HKG", "KOR", "JPN", "TWN", "VNM", "THA", "MYS", "IDN", "SGP", "PHL"}
    SA = {"IND", "PAK", "BGD", "LKA"}
    BS = {"RUS", "UKR", "BLR", "GEO", "KAZ", "MDA"}
    EU = {"ITA", "DEU", "FRA", "ESP", "GRC", "BGR", "ROU", "POL", "CZE", "AUT", "NLD", "BEL", "GBR", "SWE", "FIN",
          "PRT", "SVN", "SVK", "HUN", "CHE", "DNK", "IRL", "NOR", "HRV", "LTU", "LVA", "EST", "LUX", "SRB"}
    df["region"] = df.iso3.map(lambda i: "EA" if i in EA else "SA" if i in SA else "BS" if i in BS else
                               "EU" if i in EU else "TR" if i == "TUR" else "OT")
    out = {}
    pk = {"structural steel": "structural steel", "reinforcing bars": "reinforcing bars", "cables": "cables",
          "transformers": "transformers", "air conditioning units": "air conditioning units",
          "aluminium structures": "aluminium structures", "valves": "valves", "lifting equipment": "lifting equipment"}
    shares = []
    for yr in [2019, 2021, 2023]:
        t = df[df.year == yr].groupby(["package", "region"]).primaryValue.sum().unstack(fill_value=0)
        shares.append(t.div(t.sum(1), axis=0))
    sh = sum(shares) / 3
    sh = sh.reindex(columns=REGIONS, fill_value=0)
    val = df.groupby("package").primaryValue.sum()
    for c in CATS:
        out[c] = sh.loc[pk[c]].values.astype(float)
    w = np.array([val[pk[c]] for c in CATS], float)
    return out, w / w.sum()


SHARES, CAT_WEIGHT = sourcing_shares()
P("sourcing_shares", "mean import share by region, 2019, 2021, 2023", "-", "REAL", "UN Comtrade, reporter Egypt")
P("category_weight", "import value share by category", "-", "REAL", "UN Comtrade, reporter Egypt")


def provenance_table():
    return pd.DataFrame(PROVENANCE)
