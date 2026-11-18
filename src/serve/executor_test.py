"""Executor test: every model on the 10 held-out folders (clean Kubicka
originals, fetched by `python -m originals.held_out data/corpus
data/corpus/executor_test`).

Usage: uv run python -m serve.executor_test <root holding only executor_test>
       then, per model: uv run python -m benchmark <same root> results/executor_test <model>

Routes go to results/executor_test/<model>/routes/executor_test/, timings to
results/executor_test/timing.csv. The benchmark needs a corpus root with
only this split (a folder with a symlink to data/corpus/executor_test), so
the published train/student_test results are not touched.
"""
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from dataset import Corpus
from serve.models import run_batch
OUT = ROOT / "results" / "executor_test"

def job(a):
    model, folder = a
    g = folder.graph(); tr = folder.track(g); t0 = time.perf_counter()
    fixes = list(zip(tr.lon.tolist(), tr.lat.tolist(), tr.t.tolist()))
    try:
        r = run_batch(model, str(folder.path), str(ROOT / "results"), fixes, None)
        target = OUT / model / "routes" / "executor_test" / f"{folder.name()}.route"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(f"{x}\n" for x in r["route"]))
        return model, folder.name(), round(time.perf_counter() - t0, 1), ""
    except Exception as e:
        return model, folder.name(), round(time.perf_counter() - t0, 1), repr(e)[:200]

def main():
    folders = Corpus.from_root(Path(sys.argv[1])).folders("executor_test")
    jobs = [(m, f) for m in ["learned", "segment", "mdp", "hmm", "mcm"] for f in folders]
    with Pool(14) as p:
        rows = p.map(job, jobs, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "timing.csv", "w", newline="") as h:
        w = csv.writer(h); w.writerow(["model", "id", "seconds", "error"]); w.writerows(rows)
    print("errors:", [r for r in rows if r[3]]); print("slowest:", sorted(rows, key=lambda r: -r[2])[:3])


if __name__ == "__main__":
    main()
