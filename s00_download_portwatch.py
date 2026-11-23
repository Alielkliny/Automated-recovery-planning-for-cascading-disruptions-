"""Download IMF PortWatch daily chokepoint transit data (public ArcGIS feature service)."""
import json, sys, time, hashlib
from pathlib import Path
import requests
import pandas as pd

BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "portwatch_chokepoints_daily.csv"

def fetch(where):
    rows, offset = [], 0
    while True:
        p = {"where": where, "outFields": "*", "resultOffset": offset, "resultRecordCount": 1000,
             "f": "json"}
        for attempt in range(8):
            r = requests.get(BASE, params=p, timeout=120); r.raise_for_status()
            d = r.json()
            if "error" not in d:
                break
            time.sleep(2 + 2 * attempt)  # the service intermittently answers 'Invalid URL'; retry
        else:
            raise RuntimeError(f"{where} offset {offset}: {d['error']}")
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        if not d.get("exceededTransferLimit") and len(feats) < 1000:
            break
        offset += len(feats)
        time.sleep(0.3)
    return rows

if __name__ == "__main__":
    ids = [f"chokepoint{i}" for i in range(1, 29)]
    allrows = []
    for cid in ids:
        rows = []
        for yr in range(2017, 2027):
            rows += fetch(f"portid='{cid}' AND year={yr}")
        if rows:
            print(cid, rows[0]["portname"], len(rows))
        allrows += rows
    df = pd.DataFrame(allrows)
    df.to_csv(OUT, index=False)
    print("saved", OUT, len(df), hashlib.sha256(OUT.read_bytes()).hexdigest())
