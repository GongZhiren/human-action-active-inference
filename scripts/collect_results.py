"""Aggregate results across seeds and print the Table-1 summary."""
import os, sys, json, glob
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

NAME = {"car": "Car-Following", "nba": "NBA SportVU", "ddxplus": "DDXPlus (URTI)",
        "atari_berzerk": "Atari-Berzerk"}


def gather(dataset, tag=""):
    pat = f"{dataset}_s*" + (f"_{tag}" if tag else "")
    runs = sorted(glob.glob(os.path.join(ROOT, pat)))
    accs, hhars, lats, cts, pms, rcs = [], [], [], [], [], []
    for r in runs:
        f = os.path.join(r, "results.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        fin = d["final"]
        accs.append([fin["acc@1"], fin["acc@3"], fin["acc@5"]])
        hhars.append(fin["hhar"]); lats.append(fin["latency_ms"])
        cts.append(d.get("convergence_time_h", 0)); pms.append(d.get("peak_mem_mb", 0))
        rcs.append(d.get("n_rules", 0))
    return dict(accs=np.array(accs), hhar=np.array(hhars), lat=np.array(lats),
               ct=np.array(cts), pm=np.array(pms), rc=np.array(rcs), n=len(accs))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"{'Dataset':<16} {'Acc@1':>13} {'Acc@3':>13} {'Acc@5':>13} {'HHAR':>9} "
          f"{'Lat(ms)':>9} {'CT(h)':>7} {'PM(MB)':>8} {'RC':>4}")
    print("-" * 100)
    n = 0
    for ds in ["nba", "car", "ddxplus", "atari_berzerk"]:
        g = gather(ds, tag)
        if g["n"] == 0:
            print(f"{NAME[ds]:<16}  (no runs found)")
            continue
        n = g["n"]
        a = g["accs"]; am = a.mean(0); asd = a.std(0)
        def fmt(m, s):
            return f"{m:5.2f}±{s:4.2f}"
        row = f"{NAME[ds]:<16} "
        row += f"{fmt(am[0],asd[0]):>13} {fmt(am[1],asd[1]):>13} {fmt(am[2],asd[2]):>13} "
        row += f"{g['hhar'].mean():9.2f} {g['lat'].mean():9.2f} {g['ct'].mean():7.2f} "
        row += f"{g['pm'].mean():8.0f} {int(g['rc'].mean()):4d}"
        print(row)
    print("-" * 100)
    print(f"(mean±std over {n} seed(s); Acc@k = mean per-step accuracy over the next k steps)")


if __name__ == "__main__":
    main()
