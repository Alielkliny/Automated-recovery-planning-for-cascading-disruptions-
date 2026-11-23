"""Staged experiments. Usage: python experiment.py tune|dev|test1|test2
Writes results/<stage>.jsonl and results/<stage>_manifest.json (code and input SHA-256, configs, splits)."""
import sys, json, glob, random, hashlib, datetime, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import run
from model.psplib import parse
from model.network import EPISODES_DEV, EPISODES_TEST

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; RES.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes():
    files = sorted(glob.glob(str(ROOT / "code" / "model" / "*.py"))) + [str(ROOT / "code" / f) for f in
                                                                         ("runner.py", "experiment.py")]
    return {Path(f).name: sha(f) for f in files}


def instances(set_name, n, seed, exclude=()):
    fs = sorted(glob.glob(str(ROOT / "data" / "raw" / "psplib" / set_name / "*.sm")))
    fs = [f for f in fs if parse(f).tardcost > 0 and Path(f).name not in exclude]
    return sorted(random.Random(seed).sample(fs, n))


DEV_METHODS = ["naive", "practice", "buffer", "float", "saaplan", "detmpc", "saampc", "ablation", "carp"]

if __name__ == "__main__":
    stage = sys.argv[1]
    probe = [Path(p).name for p in json.load(open(ROOT / "probe" / "probe_instances.json"))]
    if stage in ("tune7", "tune8", "tune9", "tune10", "tune11"):   # tune8: edge extension of the tune7 grids (amendment 14)
        spec = json.load(open(ROOT / "probe" / f"spec_tuning{stage[4:]}.json"))
        insts = [str(ROOT / "data/raw/psplib/j30" / p) for p in probe]
        eps, seeds = EPISODES_DEV + EPISODES_TEST, range(1, 4)
        methods = [tuple(m) if isinstance(m, list) else m for m in spec["methods"]]
        cfg = spec["cfg"]
    elif stage in ("tune", "tune2", "tune3", "tune4", "tune5", "tune6"):
        spec = json.load(open(ROOT / "probe" / {"tune": "spec_tuning.json", "tune2": "spec_tuning2.json", "tune3": "spec_tuning3.json", "tune4": "spec_tuning4.json", "tune5": "spec_tuning5.json", "tune6": "spec_tuning6.json"}[stage]))
        insts = [str(ROOT / "data/raw/psplib/j30" / p) for p in probe]
        eps, seeds = EPISODES_DEV, range(1, 4)
        methods = [tuple(m) if isinstance(m, list) else m for m in spec["methods"]]
        cfg = spec["cfg"]
    else:
        tuned = json.load(open(RES / "tuned_configs.json"))
        cfg = {m: tuned.get(m, {}) for m in DEV_METHODS}
        methods = list(DEV_METHODS)
        if stage == "dev":
            insts = instances("j30", 30, 2026110201, exclude=probe)
            eps, seeds = EPISODES_DEV + EPISODES_TEST, range(101, 111)   # amendment 13: Red Sea joins development
        elif stage == "test1":
            insts = instances("j60", 30, 2026110202); eps, seeds = EPISODES_TEST, range(201, 206)
        elif stage == "test3":
            design = sys.argv[2] if len(sys.argv) > 2 else ""
            if not design or not (RES / f"test_lock_{design}.json").exists():
                raise SystemExit("Test 3 is sealed")
            used = set(json.load(open(RES / "test1_manifest.json"))["instances"])
            insts = instances("j60", 30, 2026110204, exclude=used); eps, seeds = EPISODES_TEST, range(401, 406)
        elif stage == "test2":
            # guard: the J90 split stays sealed until a named design passes its development check (protocol amendment 12)
            design = sys.argv[2] if len(sys.argv) > 2 else ""
            lock = RES / f"test_lock_{design}.json"
            if not design or not lock.exists():
                raise SystemExit("Test 2 is sealed: run as `experiment.py test2 <design>` after writing results/test_lock_<design>.json")
            insts = instances("j90", 20, 2026110203); eps, seeds = EPISODES_TEST, range(301, 306)
        else:
            raise SystemExit(stage)
    if stage in ("test2", "test3"):
        stage_name = stage
    out = RES / f"{stage}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists: results are never overwritten")
    manifest = dict(stage=stage, started=datetime.datetime.now().isoformat(timespec="seconds"),
                    instances=[Path(i).name for i in insts], episodes=list(eps), seeds=list(seeds),
                    methods=[m if isinstance(m, str) else list(m) for m in methods], configs=cfg,
                    code_sha256=code_hashes(), python=platform.python_version(), smoke=False,
                    inputs_sha256={Path(f).name: sha(f) for f in insts})
    (RES / f"{stage}_manifest.json").write_text(json.dumps(manifest, indent=1))
    tasks = [(i, e, s, methods, cfg, {}) for i in insts for e in eps for s in seeds]
    run(tasks, out, workers=9)
    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest["n_rows"] = sum(1 for _ in open(out))
    (RES / f"{stage}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", stage, manifest["n_rows"])
