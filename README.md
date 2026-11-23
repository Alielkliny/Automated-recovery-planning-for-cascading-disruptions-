# Supply-network simulation and weekly decision model

Weekly decisions that protect deliveries of imported construction materials when one disruption sets off others along a supply network of suppliers, carrier lanes, a canal chokepoint and the destination port. A daily execution simulator with weekly review epochs runs the project schedule, planner-side forecasts and a recovery-action MILP solved with HiGHS, against the compared recovery policies under common random numbers. Disruption persistence and chokepoint behaviour are calibrated on public series.

## Scope of this repository

This repository holds the code of the method, the comparison methods and the evaluation, with the split,
protocol and settings files that the scripts read. It does not hold the input data, run outputs, figures or tables.

## Repository layout

| Path | Contents |
|---|---|
| `code/model/network.py` | Fluid-queue supply network, disruption episodes and lot flows |
| `code/model/forecast.py` | Planner-side arrival forecasts |
| `code/model/milp.py` | Recovery-action MILP (HiGHS) |
| `code/model/sim.py` | Execution simulator and the recovery policies compared |
| `code/model/params.py` | Model parameters with their provenance |
| `code/model/psplib.py` | PSPLIB parser, serial schedule generation and resource-flow network |
| `code/runner.py`, `experiment.py` | Parallel runner and staged experiments (tune, dev, test1, test2) |
| `code/s11_select.py` | Configuration selection on the tuning split |
| `code/analyze.py` | Pair-level aggregation and paired tests |
| `code/s00_download_portwatch.py`, `s02_download_comtrade.py` | Downloads of the public series |
| `code/s01_premise.py`, `s03_validate_canal.py` | Disruption persistence and validation of the chokepoint queue |
| `code/s20_amplification.py`, `s21_conditions.py`, `s22_misspec.py`, `s23_sensitivity.py` | Compound-shock amplification, decision conditions, misspecification and sensitivity |
| `code/tests/` | Verification and leakage tests |
| `probe/spec_tuning*.json`, `probe/probe_instances.json`, `results/tuned_configs.json` | Tuning grids, instances held back from the splits and the selected settings |

## Environment

The code was run with Python 3.13 on Windows 11. Install the packages with

```
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` are those of the Python environment the code was run in.

## Input data

The input data are public and are not redistributed here. Place them as follows.

- PSPLIB J30 instances under `data/raw/psplib/j30/`.
- IMF PortWatch daily chokepoint transits (`code/s00_download_portwatch.py`) and UN Comtrade imports (`code/s02_download_comtrade.py`), both saved to `data/raw/`.
- NY Fed Global Supply Chain Pressure Index `data/raw/gscpi_data.xlsx` and the World Bank commodity price file `CMO-Historical-Data-Monthly.xlsx` read by `code/model/params.py`.

## Running the code

```
cd code
python s00_download_portwatch.py
python s02_download_comtrade.py
python s01_premise.py
python s03_validate_canal.py
python experiment.py tune
python s11_select.py
python experiment.py dev
python experiment.py test1
python experiment.py test2
python analyze.py
python s20_amplification.py
python s21_conditions.py
python s22_misspec.py dev
python s23_sensitivity.py
python -m pytest -q tests
```

`code/model/params.py` reads `probe/premise.json`, which `s01_premise.py` writes.
