"""Training dynamics (Fig 5/6): parse per-epoch F/VFE/EFE, RC, latency, Acc@k,
HHAR from the training logs and dump per-dataset curves."""
import os, re, json, glob, sys
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
NAME = {"car": "Car-Following", "nba": "NBA SportVU", "ddxplus": "DDXPlus", "atari_berzerk": "Atari-Berzerk"}

pat = re.compile(
    r"\[ep\s*(\d+)\].*?loss=([\d.]+).*?vfe=([\d.]+).*?efe=([\d.]+).*?"
    r"acc@1=([\d.]+)\s*acc@3=([\d.]+)\s*acc@5=([\d.]+)\s*hhar=([\d.]+)\s*RC=(\d+)\s*lat=([\d.]+)")

out = {}
for ds in ["car", "nba", "ddxplus", "atari_berzerk"]:
    f = os.path.join(ROOT, f"{ds}_s0", "log.txt")
    if not os.path.exists(f):
        continue
    rows = []
    for line in open(f):
        m = pat.search(line)
        if m:
            ep, loss, vfe, efe, a1, a3, a5, hh, rc, lat = m.groups()
            rows.append(dict(epoch=int(ep), F=float(loss), VFE=float(vfe), EFE=float(efe),
                             acc1=float(a1), acc3=float(a3), acc5=float(a5),
                             hhar=float(hh), RC=int(rc), latency=float(lat)))
    if rows:
        out[ds] = rows
        print(f"=== {NAME.get(ds, ds)} training dynamics ({len(rows)} epochs) ===")
        print("  ep:  F     VFE   EFE   | acc@1/3/5      HHAR  RC  lat")
        for r in rows[::max(1, len(rows)//6)] + [rows[-1]]:
            print(f"  {r['epoch']:2d}: {r['F']:5.2f} {r['VFE']:5.2f} {r['EFE']:5.2f} | "
                  f"{r['acc1']:.1f}/{r['acc3']:.1f}/{r['acc5']:.1f}  {r['hhar']:.1f}  {r['RC']:3d}  {r['latency']:.1f}")

json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "experiments", "training_dynamics.json"), "w"), indent=2)
print("\nsaved experiments/training_dynamics.json")
