"""UN Comtrade public preview API: annual imports by partner for construction material HS codes (no key)."""
import time, json, hashlib
from pathlib import Path
import requests, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "data" / "raw" / "comtrade_egypt_imports.csv"
HS = {"7308": "structural steel", "7214": "reinforcing bars", "8544": "cables", "8504": "transformers",
      "8415": "air conditioning units", "7610": "aluminium structures", "8481": "valves", "8428": "lifting equipment"}
rows = []
for yr in [2019, 2021, 2023]:
    for code in HS:
        u = f"https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=818&period={yr}&cmdCode={code}&flowCode=M"
        for a in range(5):
            r = requests.get(u, timeout=60)
            if r.status_code == 200: break
            time.sleep(3 + 3 * a)
        d = r.json().get("data", [])
        for x in d:
            rows.append({"year": yr, "hs": code, "package": HS[code], "partnerCode": x["partnerCode"],
                         "primaryValue": x["primaryValue"], "netWgt": x.get("netWgt")})
        print(yr, code, len(d)); time.sleep(1.2)
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
print(OUT, len(df), hashlib.sha256(OUT.read_bytes()).hexdigest())
