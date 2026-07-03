"""Rule-performance trade-off (Figure 3).

Grows rule banks of increasing size (RC) from a trained model and evaluates
accuracy / HHAR / rule-hit-rate / latency at each size: as the rule bank grows,
more decisions are handled by rule lookup (bypassing EFE planning), so latency
drops while accuracy stays stable.
"""
import os, sys, json, argparse, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rgai.models import WorldModel
from rgai.rules import RuleLibrary
from rgai.metrics import evaluate
from rgai.engine import harvest_stats
from scripts.configs import CONFIGS, ROOT
from scripts.run import load_datasets
from torch.utils.data import DataLoader, Subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = CONFIGS[args.dataset](0)
    meta = os.path.join(cfg.data_dir, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        cfg.model.vocab_size = m.get("vocab_size", 0); cfg.model.extra_dim = m.get("extra_dim", 0)
        cfg.model.n_actions = m.get("n_actions", cfg.model.n_actions); cfg.model.n_mental = m.get("n_mental", cfg.model.n_mental)
    tr, va = load_datasets(cfg)
    run = os.path.join(ROOT, "runs", f"{cfg.name}_s0" + (("_"+args.tag) if args.tag else ""))
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location=cfg.train.device)
    mo = WorldModel(cfg.model).to(cfg.train.device); mo.load_state_dict(ck["model"], strict=False); mo.eval()
    loader = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=tr.collate)
    dev = cfg.train.device

    # harvest a pool of (S, m, a, fe) once
    hl = DataLoader(tr, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=tr.collate)
    Ss, Ms, As, Fs = [], [], [], []
    for i, b in enumerate(hl):
        b = {k: v.to(dev) for k, v in b.items()}
        S, M, A, Fe = harvest_stats(mo, b, cfg.train)
        Ss.append(S); Ms.append(M); As.append(A); Fs.append(Fe)
        if i + 1 >= 60:
            break
    S = torch.cat(Ss); M = torch.cat(Ms); A = torch.cat(As); Fe = torch.cat(Fs)

    print(f"\n=== Rule-performance trade-off: {cfg.name} ===")
    print(f"{'RC':>5} {'Acc@3':>7} {'Acc@5':>7} {'HHAR':>7} {'RHR':>6} {'Lat(ms)':>8}")
    # RC=0 baseline (no rules)
    r0 = evaluate(mo, None, loader, cfg.train, dev, use_rules=False, use_plan=True)
    print(f"{0:5d} {r0['acc@3']:7.2f} {r0['acc@5']:7.2f} {r0['hhar']:7.2f} {0.0:6.1f} {r0['latency_ms']:8.2f}")
    rows = [dict(rc=0, **{k: r0[k] for k in ["acc@1", "acc@3", "acc@5", "hhar", "rule_hit_rate", "latency_ms"]})]
    for mr in [2, 4, 8, 16, 32, 64, 128]:
        c = copy.deepcopy(cfg.rule); c.max_rules = mr
        rules = RuleLibrary(c, cfg.model.state_dim, cfg.model.n_actions, cfg.model.n_mental, dev)
        rules.build_from_stats(S, M, A, Fe)
        r = evaluate(mo, rules if len(rules) else None, loader, cfg.train, dev, True, True)
        print(f"{len(rules):5d} {r['acc@3']:7.2f} {r['acc@5']:7.2f} {r['hhar']:7.2f} {r['rule_hit_rate']:6.1f} {r['latency_ms']:8.2f}")
        rows.append(dict(rc=len(rules), **{k: r[k] for k in ["acc@1", "acc@3", "acc@5", "hhar", "rule_hit_rate", "latency_ms"]}))
    os.makedirs(os.path.join(ROOT, "runs", "tradeoff"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "runs", "tradeoff", f"{cfg.name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
